/// hold_detection_service.dart
///
/// On-device climbing hold detection using a TFLite model bundled in assets.
/// Drop-in replacement for the previous HTTP-based service — the public API
/// (detectHoldsFromBytes, DetectionResult, DetectedHold, BoundingBox) is
/// identical, so create_route_screen.dart needs only one line changed.
///
/// Supported model architectures (auto-detected from output tensor names):
///   CenterNet  — outputs named 'heatmap', 'wh', 'offset'
///   SSD / EfficientDet — one or more 4-D tensors (1, gH, gW, anchors*(5+C))
///
/// Asset path: assets/model.tflite  (configure via [modelAssetPath])

import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

// ─────────────────────────────────────────────
// Public data classes  (same shape as before)
// ─────────────────────────────────────────────

class BoundingBox {
  final double x1, y1, x2, y2;
  const BoundingBox(this.x1, this.y1, this.x2, this.y2);

  double get width  => x2 - x1;
  double get height => y2 - y1;
}

class DetectedHold {
  final BoundingBox bbox;
  final double confidence;

  const DetectedHold({required this.bbox, required this.confidence});

  /// Centre of the bounding box in image-pixel coordinates.
  ({double x, double y}) get center => (
    x: (bbox.x1 + bbox.x2) / 2,
    y: (bbox.y1 + bbox.y2) / 2,
  );
}

class DetectionResult {
  final List<DetectedHold> holds;
  final int imageWidth;
  final int imageHeight;

  const DetectionResult({
    required this.holds,
    required this.imageWidth,
    required this.imageHeight,
  });
}

// ─────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────

class HoldDetectionService {
  /// Asset path to the bundled TFLite model.
  final String modelAssetPath;

  /// Minimum confidence to keep a detection. Default matches the old server default.
  final double confidenceThreshold;

  /// Input size the model was trained with (width, height).
  final ({int width, int height}) inputSize;

  /// Number of threads for inference. 2–4 is a good default on mobile.
  final int numThreads;

  IsolateInterpreter? _isolateInterpreter;
  Interpreter?        _interpreter;
  bool                _initialized = false;

  HoldDetectionService({
    this.modelAssetPath      = 'assets/model.tflite',
    this.confidenceThreshold = 0.8,
    this.inputSize           = (width: 320, height: 320),
    this.numThreads          = 2,
    // ── Ignored parameters kept for API compatibility with the old HTTP service ──
    String? baseUrl,
  });

  // ── Initialisation ──────────────────────────────────────────────────────

  Future<void> _ensureInitialized() async {
    if (_initialized) return;

    final options = InterpreterOptions()..threads = numThreads;

    // Load model bytes from Flutter assets
    final modelData = await rootBundle.load(modelAssetPath);
    final modelBytes = modelData.buffer.asUint8List(
      modelData.offsetInBytes,
      modelData.lengthInBytes,
    );

    _interpreter = Interpreter.fromBuffer(modelBytes, options: options);

    final inputTensor = _interpreter!.getInputTensor(0);
    print('Input type: ${inputTensor.type}');
    print('Input shape: ${inputTensor.shape}');

    // Wrap in IsolateInterpreter so inference never blocks the UI thread
    _isolateInterpreter = await IsolateInterpreter.create(
      address: _interpreter!.address,
    );

    _initialized = true;
  }

  /// Kept for API compatibility. Always returns true for on-device inference.
  Future<bool> healthCheck() async {
    try {
      await _ensureInitialized();
      return true;
    } catch (e) {
      return false;
    }
  }

  // ── Public detection API ────────────────────────────────────────────────

  /// Detect climbing holds in [imageBytes] (any format flutter/image can decode).
  ///
  /// Returns a [DetectionResult] with bounding boxes in *original* image-pixel
  /// coordinates — identical to what the HTTP service used to return.
  Future<DetectionResult> detectHoldsFromBytes(Uint8List imageBytes) async {
    await _ensureInitialized();

    // Decode to RGBA, record original dimensions
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) throw Exception('Could not decode image bytes.');
    final origW = decoded.width;
    final origH = decoded.height;

    // Resize to model input size
    final resized = img.copyResize(
      decoded,
      width:  inputSize.width,
      height: inputSize.height,
      interpolation: img.Interpolation.linear,
    );

    // Build float32 input tensor: shape [1, H, W, 3], values in [0, 1]
    final inputTensor = _imageToFloat32List(resized);

    // Collect output tensor info
    final interpreter  = _interpreter!;
    final outputCount  = interpreter.getOutputTensors().length;
    final outputNames  = {
      for (var i = 0; i < outputCount; i++)
        interpreter.getOutputTensor(i).name: i
    };

    // ── Run inference in isolate ────────────────────────────────────────
    List<DetectedHold> holds;

    if (outputNames.containsKey('heatmap')) {
      // CenterNet path
      holds = await _runCenterNet(inputTensor, outputNames, origW, origH);
    } else {
      // SSD / EfficientDet path
      holds = await _runAnchorBased(inputTensor, outputCount, origW, origH);
    }

    return DetectionResult(holds: holds, imageWidth: origW, imageHeight: origH);
  }

  // ── CenterNet inference ─────────────────────────────────────────────────

  Future<List<DetectedHold>> _runCenterNet(
    List<List<List<List<double>>>> inputTensor,
    Map<String, int> outputNames,
    int origW,
    int origH,
  ) async {
    final interpreter = _interpreter!;

    // Allocate output buffers
    final hmTensor  = interpreter.getOutputTensor(outputNames['heatmap']!);
    final whtTensor = interpreter.getOutputTensor(outputNames['wh']!);
    final offTensor = interpreter.getOutputTensor(outputNames['offset']!);

    final oH = hmTensor.shape[1];
    final oW = hmTensor.shape[2];

    final hmOut  = List.generate(1, (_) =>
                   List.generate(oH, (_) =>
                   List.generate(oW, (_) => List<double>.filled(1, 0))));
    final whOut  = List.generate(1, (_) =>
                   List.generate(oH, (_) =>
                   List.generate(oW, (_) => List<double>.filled(2, 0))));
    final offOut = List.generate(1, (_) =>
                   List.generate(oH, (_) =>
                   List.generate(oW, (_) => List<double>.filled(2, 0))));

    final outputs = <int, Object>{
      outputNames['heatmap']!: hmOut,
      outputNames['wh']!:      whOut,
      outputNames['offset']!:  offOut,
    };

    await _isolateInterpreter!.runForMultipleInputs([inputTensor], outputs);

    return _decodeCenterNet(
      hmOut, whOut, offOut,
      oH: oH, oW: oW,
      origW: origW, origH: origH,
    );
  }

  List<DetectedHold> _decodeCenterNet(
    List hmOut, List whOut, List offOut, {
    required int oH, required int oW,
    required int origW, required int origH,
  }) {
    final List<DetectedHold> results = [];

    for (var y = 0; y < oH; y++) {
      for (var x = 0; x < oW; x++) {
        final conf = (hmOut[0][y][x][0] as num).toDouble();
        if (conf < confidenceThreshold) continue;

        // Simple 3×3 local maximum check
        bool isLocalMax = true;
        for (var dy = -1; dy <= 1 && isLocalMax; dy++) {
          for (var dx = -1; dx <= 1 && isLocalMax; dx++) {
            if (dx == 0 && dy == 0) continue;
            final ny = (y + dy).clamp(0, oH - 1);
            final nx = (x + dx).clamp(0, oW - 1);
            if ((hmOut[0][ny][nx][0] as num) > conf) isLocalMax = false;
          }
        }
        if (!isLocalMax) continue;

        final offX = (offOut[0][y][x][0] as num).toDouble();
        final offY = (offOut[0][y][x][1] as num).toDouble();
        final wNorm = (whOut[0][y][x][0] as num).toDouble();
        final hNorm = (whOut[0][y][x][1] as num).toDouble();

        // Map to original image coordinates
        final cxImg = ((x + offX) / oW) * origW;
        final cyImg = ((y + offY) / oH) * origH;
        final wImg  = wNorm * origW;
        final hImg  = hNorm * origH;

        results.add(DetectedHold(
          bbox: BoundingBox(
            cxImg - wImg / 2, cyImg - hImg / 2,
            cxImg + wImg / 2, cyImg + hImg / 2,
          ),
          confidence: conf,
        ));
      }
    }

    return _nms(results);
  }

  // ── SSD / EfficientDet inference ────────────────────────────────────────

  Future<List<DetectedHold>> _runAnchorBased(
    List<List<List<List<double>>>> inputTensor,
    int outputCount,
    int origW,
    int origH,
  ) async {
    final interpreter = _interpreter!;

    // Build output buffers for all scale heads
    final outputs = <int, Object>{};
    final shapes  = <int, List<int>>{};
    for (var i = 0; i < outputCount; i++) {
      final t = interpreter.getOutputTensor(i);
      shapes[i] = t.shape;   // [1, gH, gW, anchors*(5+C)]
      outputs[i] = _allocateBuffer(t.shape);
    }

    await _isolateInterpreter!.runForMultipleInputs([inputTensor], outputs);

    final List<DetectedHold> all = [];
    for (var i = 0; i < outputCount; i++) {
      final shape = shapes[i]!;
      if (shape.length != 4) continue;
      all.addAll(
        _decodeAnchorScale(outputs[i]!, shape, origW: origW, origH: origH),
      );
    }
    return _nms(all);
  }

  List<DetectedHold> _decodeAnchorScale(
    Object rawOutput, List<int> shape, {
    required int origW, required int origH,
    int numAnchors = 6,
  }) {
    // shape: [1, gH, gW, numAnchors*(5+C)]
    final gH = shape[1];
    final gW = shape[2];

    // Flatten to a 4D Dart list for indexing
    // rawOutput is List<List<List<List<double>>>> from allocateBuffer
    final out = rawOutput as List;

    final List<DetectedHold> results = [];

    for (var gy = 0; gy < gH; gy++) {
      for (var gx = 0; gx < gW; gx++) {
        for (var a = 0; a < numAnchors; a++) {
          final base = a * (5 + 1); // 5 + num_classes (1 for single-class)

          final rawCx  = _idx(out, 0, gy, gx, base + 0);
          final rawCy  = _idx(out, 0, gy, gx, base + 1);
          final rawW   = _idx(out, 0, gy, gx, base + 2);
          final rawH   = _idx(out, 0, gy, gx, base + 3);
          final rawObj = _idx(out, 0, gy, gx, base + 4);

          final objConf = _sigmoid(rawObj);
          if (objConf < confidenceThreshold) continue;

          // Decode grid-relative cx/cy → normalised image coords
          final cxNorm = (_sigmoid(rawCx) + gx) / gW;
          final cyNorm = (_sigmoid(rawCy) + gy) / gH;
          final wNorm  = math.exp(rawW) / gW;
          final hNorm  = math.exp(rawH) / gH;

          final cxImg  = cxNorm * origW;
          final cyImg  = cyNorm * origH;
          final wImg   = wNorm  * origW;
          final hImg   = hNorm  * origH;

          results.add(DetectedHold(
            bbox: BoundingBox(
              cxImg - wImg / 2, cyImg - hImg / 2,
              cxImg + wImg / 2, cyImg + hImg / 2,
            ),
            confidence: objConf,
          ));
        }
      }
    }
    return results;
  }

  // ── NMS ─────────────────────────────────────────────────────────────────

  List<DetectedHold> _nms(List<DetectedHold> detections,
      // 0.30 matches the --nms-iou default in train_holds.py.
      // Lower to 0.20 if duplicates still appear; raise toward 0.45 only if
      // legitimate nearby holds are being incorrectly merged.
      {double iouThreshold = 0.30}) {
    if (detections.isEmpty) return [];

    final sorted = List<DetectedHold>.from(detections)
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    final List<DetectedHold> kept = [];
    final List<bool> suppressed = List.filled(sorted.length, false);

    for (var i = 0; i < sorted.length; i++) {
      if (suppressed[i]) continue;
      kept.add(sorted[i]);
      for (var j = i + 1; j < sorted.length; j++) {
        if (suppressed[j]) continue;
        if (_iou(sorted[i].bbox, sorted[j].bbox) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
    return kept;
  }

  static double _iou(BoundingBox a, BoundingBox b) {
    final ix1 = math.max(a.x1, b.x1);
    final iy1 = math.max(a.y1, b.y1);
    final ix2 = math.min(a.x2, b.x2);
    final iy2 = math.min(a.y2, b.y2);
    final iw  = math.max(0.0, ix2 - ix1);
    final ih  = math.max(0.0, iy2 - iy1);
    final inter = iw * ih;
    final union = (a.x2 - a.x1) * (a.y2 - a.y1) +
                  (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return union <= 0 ? 0 : inter / union;
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  /// Convert a decoded [img.Image] (RGBA) to a [1, H, W, 3] float32 list.
  List<List<List<List<double>>>> _imageToFloat32List(img.Image image) {
    final h = image.height;
    final w = image.width;
    return List.generate(1, (_) =>
      List.generate(h, (y) =>
        List.generate(w, (x) {
          final pixel = image.getPixel(x, y);
          return [
            pixel.r / 255.0,
            pixel.g / 255.0,
            pixel.b / 255.0,
          ];
        })
      )
    );
  }

  /// Allocate a nested Dart list matching [shape] for use as a TFLite output buffer.
  Object _allocateBuffer(List<int> shape) {
    if (shape.length == 1) return List<double>.filled(shape[0], 0.0);
    return List.generate(shape[0], (_) => _allocateBuffer(shape.sublist(1)));
  }

  /// Index into a nested 4-D list without casting at every level.
  double _idx(List out, int b, int y, int x, int c) =>
      ((out[b] as List)[y] as List)[x][c] as double;

  static double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

  // ── Cleanup ──────────────────────────────────────────────────────────────

  void dispose() {
    _isolateInterpreter?.close();
    _interpreter?.close();
    _initialized = false;
  }
}