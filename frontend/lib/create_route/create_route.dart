import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:typed_data';
import '../data/climbing_models.dart';
import '../utils/hold_detection_service.dart';
import '../widgets/hold_marker_painter.dart';
import '../widgets/editor_toolbar.dart';
import '../widgets/hold_role_selector.dart';
import '../widgets/editing_hold_panel.dart';
import '../save_route_screen.dart';
import '../editor/editor_mode.dart';
import '../editor/route_editor_gesture_mixin.dart';
import '../data/quotes.dart';

class CreateRouteScreen extends StatefulWidget {
  final Function(ClimbingRoute) onRouteSaved;

  const CreateRouteScreen({super.key, required this.onRouteSaved});

  @override
  State<CreateRouteScreen> createState() => _CreateRouteScreenState();
}

class _CreateRouteScreenState extends State<CreateRouteScreen>
    with RouteEditorGestureMixin {
  // ── Services ──────────────────────────────────────────────────────────────

  final HoldDetectionService _detectionService = HoldDetectionService(
    confidenceThreshold: 0.755,
    inputSize: (width: 320, height: 320),
    numThreads: 2,
  );
  final ImagePicker _picker = ImagePicker();

  @override
  final TransformationController transformationController =
      TransformationController();

  // ── Image state ───────────────────────────────────────────────────────────

  File? _selectedImage;
  Uint8List? _selectedImageBytes;

  @override
  Size? imageSize;

  bool _isAnalyzing = false;
  String? _errorMessage;

  // ── Hold / editor state (consumed by mixin via @override getters) ─────────

  @override
  List<ClimbingHold> detectedHolds = [];

  @override
  HoldRole currentSelectionMode = HoldRole.middle;

  @override
  bool isEditingMode = false;

  @override
  bool isAddingHold = false;

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    _checkServerHealth();
  }

  @override
  void dispose() {
    transformationController.dispose();
    super.dispose();
  }

  Future<void> _checkServerHealth() async {
    final isHealthy = await _detectionService.healthCheck();
    if (!isHealthy && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Warning: Detection server is not available'),
          backgroundColor: Colors.orange,
          duration: Duration(seconds: 3),
        ),
      );
    }
  }

  // ── Image picking ─────────────────────────────────────────────────────────

  Future<void> _selectImage() => _pickImage(ImageSource.gallery);
  Future<void> _takePicture() => _pickImage(ImageSource.camera);

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? picked = await _picker.pickImage(
        source: source,
        maxWidth: 2048,
        maxHeight: 2048,
      );
      if (picked == null) return;

      setState(() {
        _isAnalyzing = true;
        _errorMessage = null;
        detectedHolds = [];
      });

      final bytes = await picked.readAsBytes();
      setState(() {
        _selectedImageBytes = bytes;
        if (!kIsWeb) _selectedImage = File(picked.path);
      });

      await _detectHolds();
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load image: $e';
        _isAnalyzing = false;
      });
    }
  }

  // ── Detection ─────────────────────────────────────────────────────────────

  Future<void> _detectHolds() async {
    if (_selectedImageBytes == null) return;

    try {
      final result =
          await _detectionService.detectHoldsFromBytes(_selectedImageBytes!);

      final rawHolds = result.holds.map((detected) {
        return ClimbingHold(
          id: 'hold_${detected.center.x.toInt()}_${detected.center.y.toInt()}',
          position: Offset(detected.center.x, detected.center.y),
          confidence: detected.confidence,
          width: detected.bbox.width,
          height: detected.bbox.height,
        );
      }).toList();

      final holds = _applyNMS(rawHolds);

      setState(() {
        detectedHolds = holds;
        imageSize = Size(
          result.imageWidth.toDouble(),
          result.imageHeight.toDouble(),
        );
        _isAnalyzing = false;
        _errorMessage = null;
      });

      if (mounted) {
        final removed = rawHolds.length - holds.length;
        final suffix = removed > 0 ? ' ($removed overlapping removed)' : '';
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Detected ${holds.length} climbing holds!$suffix'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Detection failed: $e';
        _isAnalyzing = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Detection failed: $e'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 5),
          ),
        );
      }
    }
  }

  /// Secondary NMS pass on top of the service-level NMS.
  List<ClimbingHold> _applyNMS(List<ClimbingHold> holds,
      {double overlapThreshold = 0.70}) {
    final sorted = [...holds]
      ..sort((a, b) => (b.width * b.height).compareTo(a.width * a.height));

    final kept = <ClimbingHold>[];
    for (final candidate in sorted) {
      bool suppressed = false;
      for (final keeper in kept) {
        final intersection = _intersectionArea(candidate, keeper);
        final area = candidate.width * candidate.height;
        if (area > 0 && intersection / area >= overlapThreshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) kept.add(candidate);
    }
    return kept;
  }

  double _intersectionArea(ClimbingHold a, ClimbingHold b) {
    final w = (a.position.dx + a.width / 2 < b.position.dx + b.width / 2
            ? a.position.dx + a.width / 2
            : b.position.dx + b.width / 2) -
        (a.position.dx - a.width / 2 > b.position.dx - b.width / 2
            ? a.position.dx - a.width / 2
            : b.position.dx - b.width / 2);
    final h = (a.position.dy + a.height / 2 < b.position.dy + b.height / 2
            ? a.position.dy + a.height / 2
            : b.position.dy + b.height / 2) -
        (a.position.dy - a.height / 2 > b.position.dy - b.height / 2
            ? a.position.dy - a.height / 2
            : b.position.dy - b.height / 2);
    return (w > 0 && h > 0) ? w * h : 0.0;
  }

  // ── Navigation ────────────────────────────────────────────────────────────

  void _createRoute() {
    final selectedHolds =
        detectedHolds.where((hold) => hold.isSelected).toList();
    if (selectedHolds.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select at least one hold')),
      );
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => SaveRouteScreen(
          imagePath: _selectedImage?.path,
          imageBytes: _selectedImageBytes,
          imageSize: imageSize,
          selectedHolds: selectedHolds,
          onSave: widget.onRouteSaved,
        ),
      ),
    );
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Create Route'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          if (_selectedImageBytes != null && !_isAnalyzing)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _detectHolds,
              tooltip: 'Re-analyze',
            ),
        ],
      ),
      body: _selectedImageBytes == null
          ? _buildEmptyState()
          : _buildImageAnalysis(),
    );
  }

  // ── Empty state ───────────────────────────────────────────────────────────

  Widget _buildEmptyState() {
    return Container(
      decoration: const BoxDecoration(
        image: DecorationImage(
          image: AssetImage('assets/background.png'),
          fit: BoxFit.cover,
        ),
      ),
      child: Center(
        child: Container(
          width: 220,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                getRandomMessage(),
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _selectImage,
                  style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14)),
                  child: const Text('Gallery'),
                ),
              ),
              const SizedBox(height: 10),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _takePicture,
                  style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14)),
                  child: const Text('Camera'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Image + gesture area ──────────────────────────────────────────────────

  Widget _buildImageAnalysis() {
    return Column(
      children: [
        if (_errorMessage != null) _buildErrorBanner(),
        Expanded(child: LayoutBuilder(builder: _buildGestureArea)),
        _buildBottomPanel(),
      ],
    );
  }

  Widget _buildErrorBanner() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      color: Colors.red[100],
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: Colors.red),
          const SizedBox(width: 8),
          Expanded(
            child: Text(_errorMessage!,
                style: const TextStyle(color: Colors.red)),
          ),
          IconButton(
            icon: const Icon(Icons.close, color: Colors.red),
            onPressed: () => setState(() => _errorMessage = null),
          ),
        ],
      ),
    );
  }

  Widget _buildGestureArea(BuildContext context, BoxConstraints constraints) {
    return GestureDetector(
      onTapDown: !_isAnalyzing && detectedHolds.isNotEmpty
          ? (d) => handleTap(d.localPosition, constraints)
          : null,

      // Pan — only while drawing a new hold (IV disabled)
      onPanStart: isAddingHold && !_isAnalyzing
          ? (d) => handlePanStart(d.localPosition, constraints)
          : null,
      onPanUpdate: isAddingHold && !_isAnalyzing
          ? (d) => handlePanUpdate(d.localPosition, constraints)
          : null,
      onPanEnd: isAddingHold && !_isAnalyzing
          ? (_) => handleGestureEnd(_commitNewHold)
          : null,

      // Scale — edit/select mode; multi-finger passes to IV for pinch-zoom
      onScaleStart:
          !isAddingHold && !_isAnalyzing && detectedHolds.isNotEmpty
              ? (d) => handleScaleStart(d, constraints)
              : null,
      onScaleUpdate:
          !isAddingHold && !_isAnalyzing && detectedHolds.isNotEmpty
              ? (d) => handleScaleUpdate(d, constraints)
              : null,
      onScaleEnd: !isAddingHold && !_isAnalyzing && detectedHolds.isNotEmpty
          ? (_) => handleGestureEnd(_commitNewHold)
          : null,

      behavior: HitTestBehavior.translucent,
      child: InteractiveViewer(
        transformationController: transformationController,
        minScale: 0.5,
        maxScale: 5.0,
        boundaryMargin: const EdgeInsets.all(100),
        panEnabled: !isAddingHold,
        scaleEnabled: !isAddingHold,
        child: Stack(
          fit: StackFit.expand,
          children: [
            Center(
              child: kIsWeb || _selectedImage == null
                  ? Image.memory(_selectedImageBytes!, fit: BoxFit.contain)
                  : Image.file(_selectedImage!, fit: BoxFit.contain),
            ),
            if (!_isAnalyzing && detectedHolds.isNotEmpty)
              CustomPaint(
                size: Size(constraints.maxWidth, constraints.maxHeight),
                painter: HoldMarkerPainter(
                  holds: detectedHolds,
                  imageSize: imageSize ?? Size.zero,
                  canvasSize:
                      Size(constraints.maxWidth, constraints.maxHeight),
                  editingHold: editingHold,
                  isEditingMode: isEditingMode,
                  isAddingHold: isAddingHold,
                  newHoldStart: newHoldStart,
                  newHoldEnd: lastDragPosition,
                  transformationController: transformationController,
                ),
              ),
            if (_isAnalyzing)
              Container(
                color: Colors.black54,
                child: const Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircularProgressIndicator(color: Colors.white),
                      SizedBox(height: 16),
                      Text('Analyzing holds...',
                          style:
                              TextStyle(color: Colors.white, fontSize: 18)),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  /// Called by [handleGestureEnd] when a valid new-hold box has been drawn.
  void _commitNewHold() {
    if (newHoldStart == null || lastDragPosition == null) return;
    final hold = ClimbingHold(
      id: 'manual_${DateTime.now().millisecondsSinceEpoch}',
      position: Offset(
        (newHoldStart!.dx + lastDragPosition!.dx) / 2,
        (newHoldStart!.dy + lastDragPosition!.dy) / 2,
      ),
      confidence: 1.0,
      width: (lastDragPosition!.dx - newHoldStart!.dx).abs(),
      height: (lastDragPosition!.dy - newHoldStart!.dy).abs(),
      isSelected: true,
      role: currentSelectionMode,
    );
    setState(() => detectedHolds.add(hold));
  }

  // ── Bottom panel ──────────────────────────────────────────────────────────

  Widget _buildBottomPanel() {
    final selectedCount = detectedHolds.where((h) => h.isSelected).length;
    final startCount = detectedHolds
        .where((h) => h.isSelected && h.role == HoldRole.start)
        .length;
    final finishCount = detectedHolds
        .where((h) => h.isSelected && h.role == HoldRole.finish)
        .length;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 8,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          EditorToolbar(
            isEditingMode: isEditingMode,
            isAddingHold: isAddingHold,
            onToggleEdit: () => setState(() {
              isEditingMode = !isEditingMode;
              isAddingHold = false;
              editingHold = null;
              newHoldStart = null;
              gestureMode = EditorGestureMode.idle;
            }),
            onToggleAdd: () => setState(() {
              isAddingHold = !isAddingHold;
              editingHold = null;
              newHoldStart = null;
              gestureMode = EditorGestureMode.idle;
            }),
            onZoomIn: () => setState(() => zoomBy(1.3)),
            onZoomOut: () => setState(() => zoomBy(1 / 1.3)),
            onZoomReset: () => setState(
                () => transformationController.value = Matrix4.identity()),
          ),

          if (isAddingHold) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue),
              ),
              child: const Text(
                'Drag on the image to draw a bounding box for the new hold',
                style:
                    TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
                textAlign: TextAlign.center,
              ),
            ),
          ],

          if (!isEditingMode) ...[
            const SizedBox(height: 12),
            HoldRoleSelector(
              currentRole: currentSelectionMode,
              onRoleChanged: (role) =>
                  setState(() => currentSelectionMode = role),
            ),
          ],

          if (isEditingMode && editingHold != null && !isAddingHold) ...[
            const SizedBox(height: 12),
            EditingHoldPanel(
              hold: editingHold!,
              onDelete: () => setState(() {
                detectedHolds.remove(editingHold);
                editingHold = null;
              }),
              onDeselect: () => setState(() => editingHold = null),
            ),
          ],

          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Total: ${detectedHolds.length}',
                  style:
                      const TextStyle(fontSize: 12, color: Colors.grey)),
              Text(
                'Start: $startCount | Middle: ${selectedCount - startCount - finishCount} | Finish: $finishCount',
                style: const TextStyle(
                    fontSize: 12, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: _selectImage,
                  child: const Text('Change Image'),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: FilledButton(
                  onPressed: selectedCount > 0 &&
                          startCount > 0 &&
                          finishCount > 0 &&
                          !_isAnalyzing
                      ? _createRoute
                      : null,
                  child: const Text('Create Route'),
                ),
              ),
            ],
          ),

          if (selectedCount > 0 && (startCount == 0 || finishCount == 0))
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'Select at least one start and one finish hold',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.orange[700],
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
        ],
      ),
    );
  }
}