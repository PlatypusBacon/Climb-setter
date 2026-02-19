/// hold_detection_service.dart
///
/// On-device climbing hold detection using a TFLite model bundled in assets.
/// Drop-in replacement for the previous HTTP-based service — the public API
/// (detectHoldsFromBytes, DetectionResult, DetectedHold, BoundingBox) is
/// identical, so create_route_screen.dart needs only one line changed.
///
/// Supported model architectures (auto-detected from output tensor shapes):
///   CenterNet      — 3 outputs at 80×80: [1,80,80,1], [1,80,80,2], [1,80,80,2]
///   SSD / EfficientDet — one or more 4-D tensors (1, gH, gW, anchors*(5+C))
///
/// Asset path: assets/model.tflite  (configure via [modelAssetPath])

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:developer';

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
// Internal struct for CenterNet tensor indices
// ─────────────────────────────────────────────

class _CenterNetIndices {
  final int heatmapIdx;
  final int whIdx;
  final int offsetIdx;
  final int outputH;
  final int outputW;

  const _CenterNetIndices({
    required this.heatmapIdx,
    required this.whIdx,
    required this.offsetIdx,
    required this.outputH,
    required this.outputW,
  });
}

// ─────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────

class HoldDetectionService {
  /// Asset path to the bundled TFLite model.
  final String modelAssetPath;

  /// Minimum confidence to keep a detection.
  final double confidenceThreshold;

  /// Input size the model was trained with (width, height).
  final ({int width, int height}) inputSize;

  /// Number of threads for inference. 2–4 is a good default on mobile.
  final int numThreads;

  IsolateInterpreter? _isolateInterpreter;
  Interpreter?        _interpreter;
  bool                _initialized = false;
  bool                _inputIsUint8 = false;

  /// Populated during init if the model is CenterNet; null for SSD/EfficientDet.
  _CenterNetIndices?  _centerNetIndices;

  HoldDetectionService({
    this.modelAssetPath      = 'assets/model.tflite',
    this.confidenceThreshold = 0.7,
    this.inputSize           = (width: 320, height: 320),
    this.numThreads          = 2,
    // Ignored — kept for API compatibility with the old HTTP service
    String? baseUrl,
  });

  // ── Initialisation ──────────────────────────────────────────────────────

  Future<void> _ensureInitialized() async {
    if (_initialized) return;

    final options = InterpreterOptions()..threads = numThreads;
    final modelData  = await rootBundle.load(modelAssetPath);
    final modelBytes = modelData.buffer.asUint8List(
      modelData.offsetInBytes, modelData.lengthInBytes,
    );

    _interpreter = Interpreter.fromBuffer(modelBytes, options: options);

    // Detect input type (float32 vs full-INT8-quantised uint8)
    final inputType = _interpreter!.getInputTensor(0).type;
    _inputIsUint8 = (inputType == TensorType.uint8);
    log('[HoldDetection] Input type: $inputType  uint8=$_inputIsUint8');
    log('[HoldDetection] Input shape: ${_interpreter!.getInputTensor(0).shape}');

    // Inspect outputs to identify architecture
    final outputCount = _interpreter!.getOutputTensors().length;
    log('[HoldDetection] Output tensor count: $outputCount');

    int? heatmapIdx, whIdx, offsetIdx, oH, oW;

    for (var i = 0; i < outputCount; i++) {
      final t = _interpreter!.getOutputTensor(i);
      log('[HoldDetection] Output $i: "${t.name}" shape=${t.shape}');

      // CenterNet outputs are all [1, oH, oW, C] where oH==oW (e.g. 80×80)
      if (t.shape.length == 4 && t.shape[1] == t.shape[2]) {
        final h = t.shape[1];
        final w = t.shape[2];
        final c = t.shape[3];

        // In _ensureInitialized, replace the c==2 block:
        if (c == 1 && heatmapIdx == null) {
          heatmapIdx = i; oH = h; oW = w;
        } else if (c == 2) {
          final name = t.name.toLowerCase();
          if (name.contains('wh') && whIdx == null) {
            whIdx = i;
          } else if ((name.contains('offset') || name.contains('off')) && offsetIdx == null) {
            offsetIdx = i;
          } else if (whIdx == null) {
            whIdx = i;      // fallback: first c==2 tensor
          } else if (offsetIdx == null) {
            offsetIdx = i;  // fallback: second c==2 tensor
          }
        }
      }
    }

    if (heatmapIdx != null && whIdx != null && offsetIdx != null &&
        oH != null && oW != null) {
      _centerNetIndices = _CenterNetIndices(
        heatmapIdx: heatmapIdx,
        whIdx:      whIdx,
        offsetIdx:  offsetIdx,
        outputH:    oH,
        outputW:    oW,
      );
      log('[HoldDetection] Architecture: CenterNet '
          '(heatmap=$heatmapIdx wh=$whIdx offset=$offsetIdx ${oH}x$oW)');
    } else {
      log('[HoldDetection] Architecture: SSD / EfficientDet');
    }

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
      log('[HoldDetection] healthCheck failed: $e');
      return false;
    }
  }

  // ── Public detection API ────────────────────────────────────────────────

  Future<DetectionResult> detectHoldsFromBytes(Uint8List imageBytes) async {
    await _ensureInitialized();

    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) throw Exception('Could not decode image bytes.');
    final origW = decoded.width;
    final origH = decoded.height;

    final resized = img.copyResize(
      decoded,
      width:  inputSize.width,
      height: inputSize.height,
      interpolation: img.Interpolation.linear,
    );

    final inputTensor = _buildInputTensor(resized);

    final List<DetectedHold> holds;
    
    if (_centerNetIndices != null) {
      holds = await _runCenterNet(inputTensor, origW, origH);
    } else {
      holds = await _runAnchorBased(inputTensor, origW, origH);
    }

    return DetectionResult(holds: holds, imageWidth: origW, imageHeight: origH);
  }

  // ── CenterNet inference ─────────────────────────────────────────────────

  Future<List<DetectedHold>> _runCenterNet(
    Object inputTensor,
    int origW,
    int origH,
  ) async {
    final cn  = _centerNetIndices!;
    final oH  = cn.outputH;
    final oW  = cn.outputW;

    // Allocate nested-list output buffers — tflite_flutter fills these in-place.
    // Each must exactly match the tensor shape: [1, oH, oW, C].
    final hmOut = List.generate(1, (_) =>
        List.generate(oH, (_) =>
        List.generate(oW, (_) => List<double>.filled(1, 0.0))));

    final whOut = List.generate(1, (_) =>
        List.generate(oH, (_) =>
        List.generate(oW, (_) => List<double>.filled(2, 0.0))));

    final offOut = List.generate(1, (_) =>
        List.generate(oH, (_) =>
        List.generate(oW, (_) => List<double>.filled(2, 0.0))));

    final outputs = <int, Object>{
      cn.heatmapIdx: hmOut,
      cn.whIdx:      whOut,
      cn.offsetIdx:  offOut,
    };

    await _isolateInterpreter!.runForMultipleInputs([inputTensor], outputs);

    return _decodeCenterNet(
      hmOut, whOut, offOut,
      oH: oH, oW: oW,
      origW: origW, origH: origH,
    );
  }

  Future<void> debugCenterNetOutputs(Uint8List imageBytes) async {
    await _ensureInitialized();
    final cn = _centerNetIndices!;

    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) {
      print('debugCenterNetOutputs: Failed to decode image bytes');
      return; // or throw a custom exception
    }
    final resized = img.copyResize(decoded,
        width: inputSize.width, height: inputSize.height);
    final inputTensor = _buildInputTensor(resized);

    final hmOut = List.generate(1, (_) => List.generate(cn.outputH, (_) =>
        List.generate(cn.outputW, (_) => List<double>.filled(1, 0.0))));
    final whOut = List.generate(1, (_) => List.generate(cn.outputH, (_) =>
        List.generate(cn.outputW, (_) => List<double>.filled(2, 0.0))));
    final offOut = List.generate(1, (_) => List.generate(cn.outputH, (_) =>
        List.generate(cn.outputW, (_) => List<double>.filled(2, 0.0))));

    await _isolateInterpreter!.runForMultipleInputs([inputTensor], {
      cn.heatmapIdx: hmOut,
      cn.whIdx:      whOut,
      cn.offsetIdx:  offOut,
    });

    // Flatten heatmap to find global max/min/mean
    double maxConf = 0, sum = 0;
    int count = 0;
    int peaksAbove01 = 0, peaksAbove05 = 0;
    for (var y = 0; y < cn.outputH; y++) {
      for (var x = 0; x < cn.outputW; x++) {
        final v = (hmOut[0][y][x][0] as num).toDouble();
        if (v > maxConf) maxConf = v;
        sum += v;
        count++;
        if (v > 0.1) peaksAbove01++;
        if (v > 0.5) peaksAbove05++;
      }
    }
    final mean = sum / count;

    log('[CenterNet DEBUG] Heatmap: max=$maxConf  mean=${mean.toStringAsFixed(4)}'
        '  peaks>0.1=$peaksAbove01  peaks>0.5=$peaksAbove05');
    log('[CenterNet DEBUG] confidenceThreshold=$confidenceThreshold');
    log('[CenterNet DEBUG] Your threshold would find: '
        '${peaksAbove05} detections at 0.5, '
        '${peaksAbove01} at 0.1');
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

        // 3×3 local-maximum suppression — keeps only peak heatmap responses.
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

        final offX  = (offOut[0][y][x][0] as num).toDouble();
        final offY  = (offOut[0][y][x][1] as num).toDouble();
        final wNorm = (whOut[0][y][x][0] as num).toDouble();
        final hNorm = (whOut[0][y][x][1] as num).toDouble();

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
    Object inputTensor,
    int origW,
    int origH,
  ) async {
    final interpreter = _interpreter!;
    final outputCount = interpreter.getOutputTensors().length;

    final outputs = <int, Object>{};
    final shapes  = <int, List<int>>{};

    for (var i = 0; i < outputCount; i++) {
      final t = interpreter.getOutputTensor(i);
      shapes[i]  = t.shape;
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
    // shape: [1, gH, gW, numAnchors*(5+numClasses)]
    final gH         = shape[1];
    final gW         = shape[2];
    final innerSize  = shape[3]; // should be numAnchors * (5 + numClasses)
    final numClasses = (innerSize ~/ numAnchors) - 5;

    if (numClasses < 1) {
      log('[HoldDetection] Skipping tensor with unexpected inner size $innerSize');
      return [];
    }

    final out = rawOutput as List;
    final List<DetectedHold> results = [];

    for (var gy = 0; gy < gH; gy++) {
      for (var gx = 0; gx < gW; gx++) {
        for (var a = 0; a < numAnchors; a++) {
          final base   = a * (5 + numClasses);
          final rawCx  = _idx(out, 0, gy, gx, base + 0);
          final rawCy  = _idx(out, 0, gy, gx, base + 1);
          final rawW   = _idx(out, 0, gy, gx, base + 2);
          final rawH   = _idx(out, 0, gy, gx, base + 3);
          final rawObj = _idx(out, 0, gy, gx, base + 4);

          final objConf = _sigmoid(rawObj);
          if (objConf < confidenceThreshold) continue;

          final cxNorm = (_sigmoid(rawCx) + gx) / gW;
          final cyNorm = (_sigmoid(rawCy) + gy) / gH;
          final wNorm  = math.exp(rawW) / gW;
          final hNorm  = math.exp(rawH) / gH;

          final cxImg = cxNorm * origW;
          final cyImg = cyNorm * origH;
          final wImg  = wNorm  * origW;
          final hImg  = hNorm  * origH;

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
      {double iouThreshold = 0.30}) {
    if (detections.isEmpty) return [];

    final sorted = List<DetectedHold>.from(detections)
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    final kept       = <DetectedHold>[];
    final suppressed = List.filled(sorted.length, false);

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
    final ix1   = math.max(a.x1, b.x1);
    final iy1   = math.max(a.y1, b.y1);
    final ix2   = math.min(a.x2, b.x2);
    final iy2   = math.min(a.y2, b.y2);
    final iw    = math.max(0.0, ix2 - ix1);
    final ih    = math.max(0.0, iy2 - iy1);
    final inter = iw * ih;
    final union = (a.x2 - a.x1) * (a.y2 - a.y1) +
                  (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return union <= 0 ? 0 : inter / union;
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  Object _buildInputTensor(img.Image image) {
    final h = image.height;
    final w = image.width;

    if (_inputIsUint8) {
      return List.generate(1, (_) =>
        List.generate(h, (y) =>
          List.generate(w, (x) {
            final pixel = image.getPixel(x, y);
            return [pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt()];
          })));
    } else {
      return List.generate(1, (_) =>
        List.generate(h, (y) =>
          List.generate(w, (x) {
            final pixel = image.getPixel(x, y);
            return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
          })));
    }
  }

  /// Recursively allocate a nested Dart list matching [shape].
  Object _allocateBuffer(List<int> shape) {
    if (shape.length == 1) return List<double>.filled(shape[0], 0.0);
    return List.generate(shape[0], (_) => _allocateBuffer(shape.sublist(1)));
  }

  /// Safe index into a nested 4-D list.
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