// Updated: NMS moved entirely into hold_detection_service.dart
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:typed_data';
import 'climbing_models.dart';
import 'hold_detection_service.dart';
import 'save_route_screen.dart';
import 'quotes.dart';

class CreateRouteScreen extends StatefulWidget {
  final Function(ClimbingRoute) onRouteSaved;

  const CreateRouteScreen({super.key, required this.onRouteSaved});

  @override
  State<CreateRouteScreen> createState() => _CreateRouteScreenState();
}

class _CreateRouteScreenState extends State<CreateRouteScreen> {
  final HoldDetectionService _detectionService = HoldDetectionService(
    confidenceThreshold: 0.755,
    inputSize: (width: 320, height: 320),
    numThreads: 2,
  );

  File? _selectedImage;
  Uint8List? _selectedImageBytes;
  List<ClimbingHold> _detectedHolds = [];
  bool _isAnalyzing = false;
  String? _errorMessage;
  Size? _imageSize;
  HoldRole _currentSelectionMode = HoldRole.middle;
  bool _isEditingMode = false;
  bool _isAddingHold = false;
  ClimbingHold? _editingHold;
  String? _editingAction; // 'move', 'resize'
  Offset? _lastDragPosition;
  Offset? _newHoldStart;

  // True only when a single-finger gesture started ON a hold (edit/add).
  // Multi-finger gestures always set this false so the IV handles zoom.
  bool _panIsEditGesture = false;

  final TransformationController _transformationController =
      TransformationController();
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _checkServerHealth();
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

  Future<void> _selectImage() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 2048,
        maxHeight: 2048,
      );
      if (pickedFile == null) return;

      setState(() {
        _isAnalyzing = true;
        _errorMessage = null;
        _detectedHolds = [];
      });

      final bytes = await pickedFile.readAsBytes();
      setState(() {
        _selectedImageBytes = bytes;
        if (!kIsWeb) _selectedImage = File(pickedFile.path);
      });

      await _detectHolds();
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load image: $e';
        _isAnalyzing = false;
      });
    }
  }

  Future<void> _takePicture() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 2048,
        maxHeight: 2048,
      );
      if (pickedFile == null) return;

      setState(() {
        _isAnalyzing = true;
        _errorMessage = null;
        _detectedHolds = [];
      });

      final bytes = await pickedFile.readAsBytes();
      setState(() {
        _selectedImageBytes = bytes;
        if (!kIsWeb) _selectedImage = File(pickedFile.path);
      });

      await _detectHolds();
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to take picture: $e';
        _isAnalyzing = false;
      });
    }
  }

  // ---------------------------------------------------------------------------
  // NMS (secondary pass on top of service-level NMS)
  // ---------------------------------------------------------------------------
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

      final holds = _applyNMS(rawHolds, overlapThreshold: 0.70);

      setState(() {
        _detectedHolds = holds;
        _imageSize = Size(
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

  void _createRoute() {
    final selectedHolds =
        _detectedHolds.where((hold) => hold.isSelected).toList();
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
          imageSize: _imageSize,
          selectedHolds: selectedHolds,
          onSave: widget.onRouteSaved,
        ),
      ),
    );
  }

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
              Text(getRandomMessage(),
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.titleMedium),
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

  Widget _buildImageAnalysis() {
    return Column(
      children: [
        if (_errorMessage != null)
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            color: Colors.red[100],
            child: Row(
              children: [
                const Icon(Icons.error_outline, color: Colors.red),
                const SizedBox(width: 8),
                Expanded(
                    child: Text(_errorMessage!,
                        style: const TextStyle(color: Colors.red))),
                IconButton(
                  icon: const Icon(Icons.close, color: Colors.red),
                  onPressed: () => setState(() => _errorMessage = null),
                ),
              ],
            ),
          ),
        Expanded(
          child: LayoutBuilder(
            builder: (context, constraints) {
              // ── Gesture routing strategy ─────────────────────────────────
              //
              // ROOT CAUSE of broken pinch-to-zoom:
              //   GestureDetector(onPan*) wins the gesture arena over
              //   InteractiveViewer for ALL gestures, including pinch.
              //
              // FIX: Replace onPan* with onScale*.
              //   ScaleStartDetails.pointerCount distinguishes finger count:
              //     1 finger on a hold  → edit gesture (we handle movement)
              //     1 finger on space   → _panIsEditGesture=false, IV pans
              //     2+ fingers (pinch)  → _panIsEditGesture=false, IV zooms
              //
              // When _panIsEditGesture is false, onScaleUpdate is a no-op,
              // so the InteractiveViewer processes the event unobstructed.

              // ── Gesture routing ──────────────────────────────────────
              // Two modes need different recognizers:
              //
              // ADD mode  (IV disabled):
              //   onPan* — reliable single-finger drag for box drawing.
              //   No conflict with IV since it is disabled.
              //
              // EDIT / SELECT mode (IV enabled):
              //   onScale* — single-finger for hold move/resize,
              //   multi-finger passes through to IV for pinch-zoom.
              //
              // A GestureDetector cannot have both onPan* and onScale*
              // simultaneously, so we switch based on mode.

              return GestureDetector(
                onTapDown: !_isAnalyzing && _detectedHolds.isNotEmpty
                    ? (d) => _handleTap(d.localPosition, constraints)
                    : null,

                // ── Pan callbacks: used ONLY in add-hold mode ─────────────
                onPanStart: _isAddingHold && !_isAnalyzing
                    ? (d) => _handlePanStart(d.localPosition, constraints)
                    : null,
                onPanUpdate: _isAddingHold && !_isAnalyzing
                    ? (d) => _handlePanUpdate(d.localPosition, constraints)
                    : null,
                onPanEnd: _isAddingHold && !_isAnalyzing
                    ? (_) => _handleScaleEnd()
                    : null,

                // ── Scale callbacks: used in edit/select mode ─────────────
                onScaleStart: !_isAddingHold && !_isAnalyzing && _detectedHolds.isNotEmpty
                    ? (d) => _handleScaleStart(d, constraints)
                    : null,
                onScaleUpdate: !_isAddingHold && !_isAnalyzing && _detectedHolds.isNotEmpty
                    ? (d) => _handleScaleUpdate(d, constraints)
                    : null,
                onScaleEnd: !_isAddingHold && !_isAnalyzing && _detectedHolds.isNotEmpty
                    ? (_) => _handleScaleEnd()
                    : null,

                behavior: HitTestBehavior.translucent,

                child: InteractiveViewer(
                  transformationController: _transformationController,
                  minScale: 0.5,
                  maxScale: 5.0,
                  boundaryMargin: const EdgeInsets.all(100),
                  // Disabled while adding a hold so GestureDetector owns all touches
                  panEnabled: !_isAddingHold,
                  scaleEnabled: !_isAddingHold,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Center(
                        child: kIsWeb || _selectedImage == null
                            ? Image.memory(_selectedImageBytes!,
                                fit: BoxFit.contain)
                            : Image.file(_selectedImage!,
                                fit: BoxFit.contain),
                      ),
                      if (!_isAnalyzing && _detectedHolds.isNotEmpty)
                        CustomPaint(
                          size: Size(
                              constraints.maxWidth, constraints.maxHeight),
                          painter: HoldMarkerPainter(
                            holds: _detectedHolds,
                            imageSize: _imageSize ?? Size.zero,
                            canvasSize: Size(
                                constraints.maxWidth, constraints.maxHeight),
                            editingHold: _editingHold,
                            isEditingMode: _isEditingMode,
                            isAddingHold: _isAddingHold,
                            newHoldStart: _newHoldStart,
                            newHoldEnd: _lastDragPosition,
                            transformationController: _transformationController,
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
                                    style: TextStyle(
                                        color: Colors.white, fontSize: 18)),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
        _buildBottomPanel(),
      ],
    );
  }

  // ── Gesture handlers ───────────────────────────────────────────────────────

  void _handleTap(Offset tapPosition, BoxConstraints constraints) {
    if (_imageSize == null || _detectedHolds.isEmpty) return;

    final imageCoords = _screenToImageCoordinates(tapPosition, constraints);
    if (imageCoords == null) return;

    ClimbingHold? tappedHold;
    for (final hold in _detectedHolds) {
      if (_isPointInHold(imageCoords, hold)) {
        tappedHold = hold;
        break;
      }
    }

    if (tappedHold != null) {
      setState(() {
        if (_isEditingMode) {
          _editingHold = tappedHold;
        } else {
          if (!tappedHold!.isSelected) {
            tappedHold.isSelected = true;
            tappedHold.role = _currentSelectionMode;
          } else if (tappedHold.role == _currentSelectionMode) {
            tappedHold.isSelected = false;
          } else {
            tappedHold.role = _currentSelectionMode;
          }
        }
      });
    }
  }

  // ── Pan handlers: exclusively for add-hold box drawing ──────────────────
  // Safe to use onPan* here because the IV is disabled in add mode,
  // so there is no gesture arena conflict.

  void _handlePanStart(Offset position, BoxConstraints constraints) {
    if (_imageSize == null) return;
    final imageCoords = _screenToImageCoordinates(position, constraints);
    if (imageCoords == null) return;
    setState(() {
      _panIsEditGesture = true;
      _newHoldStart = imageCoords;
      _lastDragPosition = imageCoords;
    });
  }

  void _handlePanUpdate(Offset position, BoxConstraints constraints) {
    if (!_panIsEditGesture) return;
    final imageCoords = _screenToImageCoordinates(position, constraints);
    if (imageCoords == null) return;
    setState(() => _lastDragPosition = imageCoords);
  }

  // ── Scale handlers: for edit/select mode (hold move/resize + pinch-zoom) ──

  void _handleScaleStart(
      ScaleStartDetails details, BoxConstraints constraints) {
    // Multi-finger (pinch) → always let the InteractiveViewer handle it.
    if (details.pointerCount > 1) {
      _panIsEditGesture = false;
      return;
    }

    if (_imageSize == null) return;
    final imageCoords =
        _screenToImageCoordinates(details.focalPoint, constraints);
    if (imageCoords == null) return;

    // Add-hold mode: any single-finger drag draws a new box.
    if (_isAddingHold) {
      setState(() {
        _panIsEditGesture = true;
        _newHoldStart = imageCoords;
        _lastDragPosition = imageCoords;
      });
      return;
    }

    // Edit mode: single-finger drag on a hold = move/resize.
    if (_isEditingMode) {
      for (final hold in _detectedHolds) {
        if (_isPointInHold(imageCoords, hold)) {
          final distToLeft =
              (imageCoords.dx - (hold.position.dx - hold.width / 2)).abs();
          final distToRight =
              (imageCoords.dx - (hold.position.dx + hold.width / 2)).abs();
          final distToTop =
              (imageCoords.dy - (hold.position.dy - hold.height / 2)).abs();
          final distToBottom =
              (imageCoords.dy - (hold.position.dy + hold.height / 2)).abs();
          const edgeThreshold = 20.0;

          setState(() {
            _panIsEditGesture = true;
            _editingHold = hold;
            _lastDragPosition = imageCoords;
            _editingAction = (distToLeft < edgeThreshold ||
                    distToRight < edgeThreshold ||
                    distToTop < edgeThreshold ||
                    distToBottom < edgeThreshold)
                ? 'resize'
                : 'move';
          });
          return;
        }
      }
    }

    // Single finger on empty space → IV pans, we do nothing.
    _panIsEditGesture = false;
  }

  void _handleScaleUpdate(
      ScaleUpdateDetails details, BoxConstraints constraints) {
    // Multi-finger or non-edit → don't interfere; IV handles zoom/pan.
    if (!_panIsEditGesture || details.pointerCount > 1) return;

    final imageCoords =
        _screenToImageCoordinates(details.focalPoint, constraints);
    if (imageCoords == null) return;

    if (_isAddingHold && _newHoldStart != null) {
      setState(() => _lastDragPosition = imageCoords);
      return;
    }

    if (_isEditingMode && _editingHold != null && _lastDragPosition != null) {
      final delta = imageCoords - _lastDragPosition!;
      setState(() {
        if (_editingAction == 'move') {
          _editingHold!.position = Offset(
            _editingHold!.position.dx + delta.dx,
            _editingHold!.position.dy + delta.dy,
          );
        } else if (_editingAction == 'resize') {
          _editingHold!.width =
              (_editingHold!.width + delta.dx * 2).clamp(20.0, 200.0);
          _editingHold!.height =
              (_editingHold!.height + delta.dy * 2).clamp(20.0, 200.0);
        }
        _lastDragPosition = imageCoords;
      });
    }
  }

  void _handleScaleEnd() {
    if (!_panIsEditGesture) {
      _panIsEditGesture = false;
      return;
    }

    if (_isAddingHold && _newHoldStart != null && _lastDragPosition != null) {
      final width = (_lastDragPosition!.dx - _newHoldStart!.dx).abs();
      final height = (_lastDragPosition!.dy - _newHoldStart!.dy).abs();

      if (width > 10 && height > 10) {
        final newHold = ClimbingHold(
          id: 'manual_${DateTime.now().millisecondsSinceEpoch}',
          position: Offset(
            (_newHoldStart!.dx + _lastDragPosition!.dx) / 2,
            (_newHoldStart!.dy + _lastDragPosition!.dy) / 2,
          ),
          confidence: 1.0,
          width: width,
          height: height,
          isSelected: true,
          role: _currentSelectionMode,
        );
        setState(() => _detectedHolds.add(newHold));
      }
      setState(() {
        _newHoldStart = null;
        _lastDragPosition = null;
        _panIsEditGesture = false;
      });
    } else {
      setState(() {
        _lastDragPosition = null;
        _editingAction = null;
        _panIsEditGesture = false;
      });
    }
  }

  // ── Coordinate math ────────────────────────────────────────────────────────

  Offset? _screenToImageCoordinates(
      Offset viewportPosition, BoxConstraints constraints) {
    if (_imageSize == null) return null;

    final inverseMatrix =
        Matrix4.inverted(_transformationController.value);
    final contentPos =
        MatrixUtils.transformPoint(inverseMatrix, viewportPosition);

    final imageAspect = _imageSize!.width / _imageSize!.height;
    final containerAspect = constraints.maxWidth / constraints.maxHeight;

    double displayScale;
    double imageOffsetX = 0;
    double imageOffsetY = 0;

    if (imageAspect > containerAspect) {
      displayScale = constraints.maxWidth / _imageSize!.width;
      imageOffsetY =
          (constraints.maxHeight - _imageSize!.height * displayScale) / 2;
    } else {
      displayScale = constraints.maxHeight / _imageSize!.height;
      imageOffsetX =
          (constraints.maxWidth - _imageSize!.width * displayScale) / 2;
    }

    return Offset(
      ((contentPos.dx - imageOffsetX) / displayScale)
          .clamp(0.0, _imageSize!.width),
      ((contentPos.dy - imageOffsetY) / displayScale)
          .clamp(0.0, _imageSize!.height),
    );
  }

  bool _isPointInHold(Offset point, ClimbingHold hold) {
    return point.dx >= hold.position.dx - hold.width / 2 &&
        point.dx <= hold.position.dx + hold.width / 2 &&
        point.dy >= hold.position.dy - hold.height / 2 &&
        point.dy <= hold.position.dy + hold.height / 2;
  }

  // ── UI ─────────────────────────────────────────────────────────────────────

  /// Generic circle-avatar + label button used for Edit, Add, and zoom controls.
  /// Matches the visual style of [_buildRoleButtonColumn] so all toolbar
  /// buttons look consistent and never overflow their label.
  Widget _buildToolButton({
    required String label,
    required IconData icon,
    required Color color,
    required bool isActive,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            backgroundColor: isActive ? color : Colors.grey[300],
            radius: 18,
            child: Icon(icon, color: isActive ? Colors.white : color, size: 20),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
              color: isActive ? color : Colors.black87,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRoleButtonColumn(
      String label, HoldRole role, IconData icon, Color color) {
    final isActiveMode = _currentSelectionMode == role;

    Widget iconWidget;
    if (role == HoldRole.foot) {
      iconWidget = Image.asset('assets/icon/foot.png',
          width: 20, height: 20, color: isActiveMode ? Colors.white : color);
    } else {
      iconWidget =
          Icon(icon, color: isActiveMode ? Colors.white : color, size: 20);
    }

    return GestureDetector(
      onTap: () => setState(() => _currentSelectionMode = role),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            backgroundColor: isActiveMode ? color : Colors.grey[300],
            radius: 18,
            child: iconWidget,
          ),
          const SizedBox(height: 4),
          Text(label,
              style: TextStyle(
                fontSize: 12,
                fontWeight:
                    isActiveMode ? FontWeight.bold : FontWeight.normal,
                color: isActiveMode ? color : Colors.black87,
              )),
        ],
      ),
    );
  }

  Widget _buildBottomPanel() {
    final selectedCount =
        _detectedHolds.where((h) => h.isSelected).length;
    final startCount = _detectedHolds
        .where((h) => h.isSelected && h.role == HoldRole.start)
        .length;
    final finishCount = _detectedHolds
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
          // ── Toolbar: Edit / Add / Zoom — same circle+label style as role buttons
          Wrap(
            spacing: 8,
            runSpacing: 8,
            alignment: WrapAlignment.center,
            children: [
              // Edit / Done toggle
              _buildToolButton(
                label: _isEditingMode ? 'Done' : 'Edit',
                icon: _isEditingMode ? Icons.check : Icons.edit,
                color: Colors.orange,
                isActive: _isEditingMode,
                onTap: () => setState(() {
                  _isEditingMode = !_isEditingMode;
                  _isAddingHold = false;
                  _editingHold = null;
                  _newHoldStart = null;
                  _panIsEditGesture = false;
                }),
              ),
              // Add hold — only visible in edit mode
              if (_isEditingMode)
                _buildToolButton(
                  label: _isAddingHold ? 'Cancel' : 'Add',
                  icon: _isAddingHold ? Icons.close : Icons.add_box,
                  color: Colors.blue,
                  isActive: _isAddingHold,
                  onTap: () => setState(() {
                    _isAddingHold = !_isAddingHold;
                    _editingHold = null;
                    _newHoldStart = null;
                    _panIsEditGesture = false;
                  }),
                ),
              // Zoom in
              _buildToolButton(
                label: 'In',
                icon: Icons.zoom_in,
                color: Colors.grey[700]!,
                isActive: false,
                onTap: () => setState(() {
                  final s = _transformationController.value.getMaxScaleOnAxis();
                  final t = _transformationController.value.getTranslation();
                  _transformationController.value = Matrix4.identity()
                    ..translate(t.x, t.y)
                    ..scale((s * 1.3).clamp(0.5, 5.0));
                }),
              ),
              // Zoom out
              _buildToolButton(
                label: 'Out',
                icon: Icons.zoom_out,
                color: Colors.grey[700]!,
                isActive: false,
                onTap: () => setState(() {
                  final s = _transformationController.value.getMaxScaleOnAxis();
                  final t = _transformationController.value.getTranslation();
                  _transformationController.value = Matrix4.identity()
                    ..translate(t.x, t.y)
                    ..scale((s / 1.3).clamp(0.5, 5.0));
                }),
              ),
              // Reset zoom
              _buildToolButton(
                label: 'Reset',
                icon: Icons.crop_free,
                color: Colors.grey[700]!,
                isActive: false,
                onTap: () => setState(
                    () => _transformationController.value = Matrix4.identity()),
              ),
            ],
          ),

          if (_isAddingHold) ...[
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
                style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
                textAlign: TextAlign.center,
              ),
            ),
          ],

          if (!_isEditingMode) ...[
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: [
                _buildRoleButtonColumn('Start', HoldRole.start,
                    Icons.play_circle_filled, Colors.green),
                _buildRoleButtonColumn(
                    'Hand/Foot', HoldRole.middle, Icons.circle, Colors.blue),
                _buildRoleButtonColumn('Hand Only', HoldRole.hand,
                    Icons.back_hand, Colors.indigo),
                _buildRoleButtonColumn('Foot Only', HoldRole.foot,
                    Icons.directions_walk, Colors.purple),
                _buildRoleButtonColumn(
                    'Finish', HoldRole.finish, Icons.flag, Colors.red),
              ],
            ),
          ],

          if (_isEditingMode && _editingHold != null && !_isAddingHold) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.orange[50],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.orange),
              ),
              child: Column(
                children: [
                  const Text('Editing Hold',
                      style: TextStyle(
                          fontWeight: FontWeight.bold, fontSize: 12)),
                  const SizedBox(height: 4),
                  const Text('Drag center to move • Drag edges to resize',
                      style:
                          TextStyle(fontSize: 11, color: Colors.black87)),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      TextButton.icon(
                        onPressed: () => setState(() {
                          _detectedHolds.remove(_editingHold);
                          _editingHold = null;
                        }),
                        icon: const Icon(Icons.delete, size: 16),
                        label: const Text('Delete',
                            style: TextStyle(fontSize: 11)),
                        style: TextButton.styleFrom(
                            foregroundColor: Colors.red),
                      ),
                      TextButton.icon(
                        onPressed: () =>
                            setState(() => _editingHold = null),
                        icon: const Icon(Icons.close, size: 16),
                        label: const Text('Deselect',
                            style: TextStyle(fontSize: 11)),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],

          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Total: ${_detectedHolds.length}',
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
                    fontWeight: FontWeight.w500),
              ),
            ),
        ],
      ),
    );
  }
}

// ── Painter ────────────────────────────────────────────────────────────────────

class HoldMarkerPainter extends CustomPainter {
  final List<ClimbingHold> holds;
  final Size imageSize;
  final Size canvasSize;
  final ClimbingHold? editingHold;
  final bool isEditingMode;
  final bool isAddingHold;
  final Offset? newHoldStart;
  final Offset? newHoldEnd;
  final TransformationController transformationController;

  HoldMarkerPainter({
    required this.holds,
    required this.imageSize,
    required this.canvasSize,
    this.editingHold,
    this.isEditingMode = false,
    this.isAddingHold = false,
    this.newHoldStart,
    this.newHoldEnd,
    required this.transformationController,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width == 0 || imageSize.height == 0) return;

    final imageAspect = imageSize.width / imageSize.height;
    final canvasAspect = canvasSize.width / canvasSize.height;

    double scale;
    double offsetX = 0;
    double offsetY = 0;

    if (imageAspect > canvasAspect) {
      scale = canvasSize.width / imageSize.width;
      offsetY = (canvasSize.height - imageSize.height * scale) / 2;
    } else {
      scale = canvasSize.height / imageSize.height;
      offsetX = (canvasSize.width - imageSize.width * scale) / 2;
    }

    for (final hold in holds) {
      final centerX = hold.position.dx * scale + offsetX;
      final centerY = hold.position.dy * scale + offsetY;
      final boxWidth = hold.width * scale;
      final boxHeight = hold.height * scale;
      final left = centerX - boxWidth / 2;
      final top = centerY - boxHeight / 2;
      final rect = Rect.fromLTWH(left, top, boxWidth, boxHeight);
      final isBeingEdited = editingHold == hold;

      Color fillColor;
      Color borderColor;

      if (hold.isSelected) {
        switch (hold.role) {
          case HoldRole.start:
            fillColor = Colors.green.withOpacity(0.3);
            borderColor = Colors.green;
            break;
          case HoldRole.finish:
            fillColor = Colors.red.withOpacity(0.3);
            borderColor = Colors.red;
            break;
          case HoldRole.middle:
            fillColor = Colors.blue.withOpacity(0.3);
            borderColor = Colors.blue;
            break;
          case HoldRole.hand:
            fillColor =
                const Color.fromARGB(255, 33, 68, 243).withOpacity(0.3);
            borderColor = Colors.indigo;
            break;
          case HoldRole.foot:
            fillColor =
                const Color.fromARGB(255, 159, 33, 243).withOpacity(0.3);
            borderColor = Colors.purple;
            break;
        }
      } else {
        fillColor = Colors.grey.withOpacity(0.2);
        borderColor = Colors.grey;
      }

      if (isBeingEdited) {
        fillColor = Colors.orange.withOpacity(0.4);
        borderColor = Colors.orange;
      }

      canvas.drawRect(
          rect, Paint()..color = fillColor..style = PaintingStyle.fill);
      canvas.drawRect(
          rect,
          Paint()
            ..color = borderColor
            ..style = PaintingStyle.stroke
            ..strokeWidth = isBeingEdited ? 4 : (hold.isSelected ? 3 : 2));

      if (isBeingEdited && isEditingMode && !isAddingHold) {
        const hs = 12.0;
        final hp =
            Paint()..color = Colors.orange..style = PaintingStyle.fill;
        for (final pt in [
          Offset(left, top),
          Offset(left + boxWidth, top),
          Offset(left, top + boxHeight),
          Offset(left + boxWidth, top + boxHeight),
          Offset(centerX, top),
          Offset(centerX, top + boxHeight),
          Offset(left, centerY),
          Offset(left + boxWidth, centerY),
        ]) {
          canvas.drawCircle(pt, hs / 2, hp);
        }
      }

      if (hold.isSelected && !isBeingEdited) {
        const iconSize = 20.0;
        final iconX = centerX - iconSize / 2;
        final iconY = top - iconSize - 5;
        final iconPaint = Paint()
          ..color = borderColor
          ..style = PaintingStyle.fill;

        switch (hold.role) {
          case HoldRole.start:
            canvas.drawPath(
                Path()
                  ..moveTo(iconX, iconY)
                  ..lineTo(iconX, iconY + iconSize)
                  ..lineTo(iconX + iconSize, iconY + iconSize / 2)
                  ..close(),
                iconPaint);
            break;
          case HoldRole.finish:
            canvas.drawPath(
                Path()
                  ..moveTo(iconX, iconY + iconSize)
                  ..lineTo(iconX, iconY)
                  ..lineTo(iconX + iconSize * 0.7, iconY + iconSize * 0.3)
                  ..lineTo(iconX, iconY + iconSize * 0.6),
                iconPaint);
            break;
          case HoldRole.middle:
          case HoldRole.hand:
          case HoldRole.foot:
            canvas.drawCircle(
                Offset(iconX + iconSize / 2, iconY + iconSize / 2),
                iconSize / 3,
                iconPaint);
            break;
        }
      }

      final textPainter = TextPainter(
        text: TextSpan(
          text: '${(hold.confidence * 100).toInt()}%',
          style: TextStyle(
            color: hold.isSelected ? borderColor : Colors.grey[600],
            fontSize: 10,
            fontWeight: FontWeight.bold,
            shadows: const [Shadow(color: Colors.white, blurRadius: 3)],
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      textPainter.paint(canvas,
          Offset(centerX - textPainter.width / 2, top + boxHeight + 5));
    }

    if (isAddingHold && newHoldStart != null && newHoldEnd != null) {
      final startX = newHoldStart!.dx * scale + offsetX;
      final startY = newHoldStart!.dy * scale + offsetY;
      final endX = newHoldEnd!.dx * scale + offsetX;
      final endY = newHoldEnd!.dy * scale + offsetY;

      final previewRect = Rect.fromLTRB(
        startX < endX ? startX : endX,
        startY < endY ? startY : endY,
        startX < endX ? endX : startX,
        startY < endY ? endY : startY,
      );

      canvas.drawRect(
          previewRect,
          Paint()
            ..color = Colors.blue.withOpacity(0.3)
            ..style = PaintingStyle.fill);
      canvas.drawRect(
          previewRect,
          Paint()
            ..color = Colors.blue
            ..style = PaintingStyle.stroke
            ..strokeWidth = 3);
    }
  }

  @override
  bool shouldRepaint(covariant HoldMarkerPainter oldDelegate) =>
      oldDelegate.holds != holds ||
      oldDelegate.editingHold != editingHold ||
      oldDelegate.isEditingMode != isEditingMode ||
      oldDelegate.isAddingHold != isAddingHold ||
      oldDelegate.newHoldStart != newHoldStart ||
      oldDelegate.newHoldEnd != newHoldEnd;
}

extension DashedPath on Paint {
  set strokeDashArray(List<double> _) {}
}