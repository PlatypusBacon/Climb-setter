import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:typed_data';
import 'climbing_models.dart';
import 'hold_detection_service.dart';
import 'save_route_screen.dart';

class CreateRouteScreen extends StatefulWidget {
  final Function(ClimbingRoute) onRouteSaved;

  const CreateRouteScreen({super.key, required this.onRouteSaved});

  @override
  State<CreateRouteScreen> createState() => _CreateRouteScreenState();
}

class _CreateRouteScreenState extends State<CreateRouteScreen> {
  // Automatically use correct URL based on platform
  final HoldDetectionService _detectionService = HoldDetectionService(
    // modelAssetPath defaults to 'assets/model.tflite'
    // Override here if you named the file differently, e.g.:
    //   modelAssetPath: 'assets/centernet.tflite',
    confidenceThreshold: 0.5,
    inputSize: (width: 320, height: 320), // match your training --input-size
    numThreads: 2,
  );

  File? _selectedImage;
  Uint8List? _selectedImageBytes;  // For web platform
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
  
  final TransformationController _transformationController = TransformationController();
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

      // Read image bytes (works on all platforms)
      final bytes = await pickedFile.readAsBytes();
      
      setState(() {
        _selectedImageBytes = bytes;
        if (!kIsWeb) {
          _selectedImage = File(pickedFile.path);
        }
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

      // Read image bytes (works on all platforms)
      final bytes = await pickedFile.readAsBytes();
      
      setState(() {
        _selectedImageBytes = bytes;
        if (!kIsWeb) {
          _selectedImage = File(pickedFile.path);
        }
      });

      await _detectHolds();
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to take picture: $e';
        _isAnalyzing = false;
      });
    }
  }

  Future<void> _detectHolds() async {
    if (_selectedImageBytes == null) return;

    try {
      // Use bytes for detection (works on all platforms)
      final result = await _detectionService.detectHoldsFromBytes(_selectedImageBytes!);

      final holds = result.holds.map((detected) {
        return ClimbingHold(
          id: 'hold_${detected.center.x}_${detected.center.y}',
          position: Offset(detected.center.x, detected.center.y),
          confidence: detected.confidence,
          width: detected.bbox.width,
          height: detected.bbox.height,
        );
      }).toList();

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
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Detected ${holds.length} climbing holds!'),
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
            duration: Duration(seconds: 5),
          ),
        );
      }
    }
  }

  void _createRoute() {
    final selectedHolds = _detectedHolds.where((hold) => hold.isSelected).toList();

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
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.photo_library_outlined,
            size: 100,
            color: Colors.grey[400],
          ),
          const SizedBox(height: 24),
          Text(
            'Select a climbing wall image',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton.icon(
                onPressed: _selectImage,
                icon: const Icon(Icons.photo_library),
                label: const Text('Gallery'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                ),
              ),
              const SizedBox(width: 16),
              ElevatedButton.icon(
                onPressed: _takePicture,
                icon: const Icon(Icons.camera_alt),
                label: const Text('Camera'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                ),
              ),
            ],
          ),
        ],
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
                  child: Text(
                    _errorMessage!,
                    style: const TextStyle(color: Colors.red),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close, color: Colors.red),
                  onPressed: () {
                    setState(() {
                      _errorMessage = null;
                    });
                  },
                ),
              ],
            ),
          ),
        Expanded(
          child: InteractiveViewer(
            transformationController: _transformationController,
            minScale: 0.5,
            maxScale: 5.0,
            boundaryMargin: const EdgeInsets.all(100),
            panEnabled: !_isEditingMode && !_isAddingHold,
            scaleEnabled: !_isEditingMode && !_isAddingHold,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Image
                Center(
                  child: kIsWeb || _selectedImage == null
                      ? Image.memory(
                          _selectedImageBytes!,
                          fit: BoxFit.contain,
                        )
                      : Image.file(
                          _selectedImage!,
                          fit: BoxFit.contain,
                        ),
                ),
                // Hold markers overlay
                if (!_isAnalyzing && _detectedHolds.isNotEmpty)
                  LayoutBuilder(
                    builder: (context, constraints) {
                      return GestureDetector(
                        onTapDown: (details) => _handleTap(details.localPosition, constraints),
                        onPanStart: (_isEditingMode || _isAddingHold) 
                            ? (details) => _handlePanStart(details.localPosition, constraints) 
                            : null,
                        onPanUpdate: (_isEditingMode || _isAddingHold) 
                            ? (details) => _handlePanUpdate(details.localPosition, constraints) 
                            : null,
                        onPanEnd: (_isEditingMode || _isAddingHold) 
                            ? (details) => _handlePanEnd() 
                            : null,
                        child: CustomPaint(
                          size: Size(constraints.maxWidth, constraints.maxHeight),
                          painter: HoldMarkerPainter(
                            holds: _detectedHolds,
                            imageSize: _imageSize ?? Size.zero,
                            canvasSize: Size(constraints.maxWidth, constraints.maxHeight),
                            editingHold: _editingHold,
                            isEditingMode: _isEditingMode,
                            isAddingHold: _isAddingHold,
                            newHoldStart: _newHoldStart,
                            newHoldEnd: _lastDragPosition,
                            transformationController: _transformationController,
                          ),
                        ),
                      );
                    },
                  ),
                // Loading overlay
                if (_isAnalyzing)
                  Container(
                    color: Colors.black54,
                    child: const Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          CircularProgressIndicator(color: Colors.white),
                          SizedBox(height: 16),
                          Text(
                            'Analyzing holds...',
                            style: TextStyle(color: Colors.white, fontSize: 18),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
        _buildBottomPanel(),
      ],
    );
  }

  void _handleTap(Offset tapPosition, BoxConstraints constraints) {
    if (_imageSize == null || _detectedHolds.isEmpty) return;

    final imageCoords = _screenToImageCoordinates(tapPosition, constraints);
    if (imageCoords == null) return;

    // Find hold that contains this point (using bounding box)
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
          // In editing mode, select hold for editing
          _editingHold = tappedHold;
        } else {
          // In selection mode, toggle selection and role
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

  void _handlePanStart(Offset position, BoxConstraints constraints) {
    if (_imageSize == null) return;

    final imageCoords = _screenToImageCoordinates(position, constraints);
    if (imageCoords == null) return;

    if (_isAddingHold) {
      // Start drawing new hold
      setState(() {
        _newHoldStart = imageCoords;
        _lastDragPosition = imageCoords;
      });
    } else if (_isEditingMode) {
      // Existing edit logic
      for (final hold in _detectedHolds) {
        if (_isPointInHold(imageCoords, hold)) {
          setState(() {
            _editingHold = hold;
            _lastDragPosition = imageCoords;
            
            // Check if near edge for resize or center for move
            final distToLeft = (imageCoords.dx - (hold.position.dx - hold.width / 2)).abs();
            final distToRight = (imageCoords.dx - (hold.position.dx + hold.width / 2)).abs();
            final distToTop = (imageCoords.dy - (hold.position.dy - hold.height / 2)).abs();
            final distToBottom = (imageCoords.dy - (hold.position.dy + hold.height / 2)).abs();
            
            final edgeThreshold = 20.0;
            
            if (distToLeft < edgeThreshold || distToRight < edgeThreshold || 
                distToTop < edgeThreshold || distToBottom < edgeThreshold) {
              _editingAction = 'resize';
            } else {
              _editingAction = 'move';
            }
          });
          break;
        }
      }
    }
  }

  void _handlePanUpdate(Offset position, BoxConstraints constraints) {
    final imageCoords = _screenToImageCoordinates(position, constraints);
    if (imageCoords == null) return;

    if (_isAddingHold && _newHoldStart != null) {
      // Update new hold preview
      setState(() {
        _lastDragPosition = imageCoords;
      });
    } else if (_isEditingMode && _editingHold != null && _lastDragPosition != null) {
      // Existing edit logic
      final delta = imageCoords - _lastDragPosition!;

      setState(() {
        if (_editingAction == 'move') {
          _editingHold!.position = Offset(
            _editingHold!.position.dx + delta.dx,
            _editingHold!.position.dy + delta.dy,
          );
        } else if (_editingAction == 'resize') {
          final newWidth = (_editingHold!.width + delta.dx * 2).clamp(20.0, 200.0);
          final newHeight = (_editingHold!.height + delta.dy * 2).clamp(20.0, 200.0);
          
          _editingHold!.width = newWidth;
          _editingHold!.height = newHeight;
        }
        
        _lastDragPosition = imageCoords;
      });
    }
  }

  void _handlePanEnd() {
    if (_isAddingHold && _newHoldStart != null && _lastDragPosition != null) {
      // Create new hold
      final width = (_lastDragPosition!.dx - _newHoldStart!.dx).abs();
      final height = (_lastDragPosition!.dy - _newHoldStart!.dy).abs();
      
      if (width > 10 && height > 10) {
        final centerX = (_newHoldStart!.dx + _lastDragPosition!.dx) / 2;
        final centerY = (_newHoldStart!.dy + _lastDragPosition!.dy) / 2;
        
        final newHold = ClimbingHold(
          id: 'manual_${DateTime.now().millisecondsSinceEpoch}',
          position: Offset(centerX, centerY),
          confidence: 1.0, // Manually added holds have 100% confidence
          width: width,
          height: height,
          isSelected: true, // Auto-select
          role: _currentSelectionMode, // Use current role
        );
        
        setState(() {
          _detectedHolds.add(newHold);
        });
      }
      
      setState(() {
        _newHoldStart = null;
        _lastDragPosition = null;
      });
    } else {
      setState(() {
        _lastDragPosition = null;
        _editingAction = null;
      });
    }
  }

  Offset? _screenToImageCoordinates(Offset screenPosition, BoxConstraints constraints) {
    if (_imageSize == null) return null;

    // ── Step 1: Calculate where the image sits within the InteractiveViewer ──
    final imageAspect     = _imageSize!.width / _imageSize!.height;
    final containerAspect = constraints.maxWidth / constraints.maxHeight;
    
    double displayScale;
    double imageOffsetX = 0;
    double imageOffsetY = 0;
    double displayWidth;
    double displayHeight;
    
    if (imageAspect > containerAspect) {
      // Image is wider than container → letterbox top/bottom
      displayScale  = constraints.maxWidth / _imageSize!.width;
      displayWidth  = constraints.maxWidth;
      displayHeight = _imageSize!.height * displayScale;
      imageOffsetY  = (constraints.maxHeight - displayHeight) / 2;
    } else {
      // Image is taller than container → letterbox left/right
      displayScale  = constraints.maxHeight / _imageSize!.height;
      displayWidth  = _imageSize!.width * displayScale;
      displayHeight = constraints.maxHeight;
      imageOffsetX  = (constraints.maxWidth - displayWidth) / 2;
    }

    // ── Step 2: Convert screen tap to "image display space" ──
    // Remove the letterbox offset so (0,0) is now the image top-left
    final imageSpaceX = screenPosition.dx - imageOffsetX;
    final imageSpaceY = screenPosition.dy - imageOffsetY;

    // ── Step 3: Apply the inverse zoom/pan transformation ──
    // The transformation controller's matrix operates on the image's
    // coordinate space (after centering), so we invert it here.
    final matrix        = _transformationController.value;
    final inverseMatrix = Matrix4.inverted(matrix);
    final transformed   = MatrixUtils.transformPoint(
      inverseMatrix,
      Offset(imageSpaceX, imageSpaceY),
    );

    // ── Step 4: Scale from display pixels to original image pixels ──
    final imageX = transformed.dx / displayScale;
    final imageY = transformed.dy / displayScale;

    // ── Step 5: Clamp to image bounds (optional safety) ──
    final clampedX = imageX.clamp(0.0, _imageSize!.width);
    final clampedY = imageY.clamp(0.0, _imageSize!.height);

    return Offset(clampedX, clampedY);
  }

  bool _isPointInHold(Offset point, ClimbingHold hold) {
    final left = hold.position.dx - hold.width / 2;
    final right = hold.position.dx + hold.width / 2;
    final top = hold.position.dy - hold.height / 2;
    final bottom = hold.position.dy + hold.height / 2;
    
    return point.dx >= left && point.dx <= right && 
           point.dy >= top && point.dy <= bottom;
  }

  Widget _buildBottomPanel() {
    final selectedCount = _detectedHolds.where((hold) => hold.isSelected).length;
    final startCount = _detectedHolds.where((h) => h.isSelected && h.role == HoldRole.start).length;
    final finishCount = _detectedHolds.where((h) => h.isSelected && h.role == HoldRole.finish).length;

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
          // Mode toggle and zoom controls
          Row(
            children: [
              // Edit mode toggle
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () {
                    setState(() {
                      _isEditingMode = !_isEditingMode;
                      _isAddingHold = false;
                      _editingHold = null;
                      _newHoldStart = null;
                    });
                  },
                  icon: Icon(
                    _isEditingMode ? Icons.check : Icons.edit,
                    size: 18,
                  ),
                  label: Text(
                    _isEditingMode ? 'Done' : 'Edit',
                    style: const TextStyle(fontSize: 12),
                  ),
                  style: OutlinedButton.styleFrom(
                    backgroundColor: _isEditingMode ? Colors.orange[100] : Colors.white,
                    side: BorderSide(
                      color: _isEditingMode ? Colors.orange : Colors.grey,
                      width: 2,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              // Add hold button
              if (_isEditingMode)
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      setState(() {
                        _isAddingHold = !_isAddingHold;
                        _editingHold = null;
                        _newHoldStart = null;
                      });
                    },
                    icon: Icon(
                      _isAddingHold ? Icons.close : Icons.add_box,
                      size: 18,
                    ),
                    label: Text(
                      _isAddingHold ? 'Cancel' : 'Add',
                      style: const TextStyle(fontSize: 12),
                    ),
                    style: OutlinedButton.styleFrom(
                      backgroundColor: _isAddingHold ? Colors.blue[100] : Colors.white,
                      side: BorderSide(
                        color: _isAddingHold ? Colors.blue : Colors.grey,
                        width: 2,
                      ),
                    ),
                  ),
                ),
              if (_isEditingMode) const SizedBox(width: 8),
              // Zoom controls
              IconButton(
                onPressed: () {
                  setState(() {
                    final currentScale = _transformationController.value.getMaxScaleOnAxis();
                    final newScale = (currentScale * 1.3).clamp(0.5, 5.0);
                    final focalPoint = Offset(
                      _transformationController.value.getTranslation().x,
                      _transformationController.value.getTranslation().y,
                    );
                    _transformationController.value = Matrix4.identity()
                      ..translate(focalPoint.dx, focalPoint.dy)
                      ..scale(newScale);
                  });
                },
                icon: const Icon(Icons.zoom_in, size: 20),
                tooltip: 'Zoom In',
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
              ),
              IconButton(
                onPressed: () {
                  setState(() {
                    final currentScale = _transformationController.value.getMaxScaleOnAxis();
                    final newScale = (currentScale / 1.3).clamp(0.5, 5.0);
                    final focalPoint = Offset(
                      _transformationController.value.getTranslation().x,
                      _transformationController.value.getTranslation().y,
                    );
                    _transformationController.value = Matrix4.identity()
                      ..translate(focalPoint.dx, focalPoint.dy)
                      ..scale(newScale);
                  });
                },
                icon: const Icon(Icons.zoom_out, size: 20),
                tooltip: 'Zoom Out',
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
              ),
              IconButton(
                onPressed: () {
                  setState(() {
                    _transformationController.value = Matrix4.identity();
                  });
                },
                icon: const Icon(Icons.crop_free, size: 20),
                tooltip: 'Reset',
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
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
            // Hold role selector (only in selection mode)
            Row(
              children: [
                Expanded(
                  child: _buildRoleButton(
                    'Start',
                    HoldRole.start,
                    Icons.play_circle_filled,
                    Colors.green,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _buildRoleButton(
                    'Middle',
                    HoldRole.middle,
                    Icons.circle,
                    Colors.blue,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _buildRoleButton(
                    'Finish',
                    HoldRole.finish,
                    Icons.flag,
                    Colors.red,
                  ),
                ),
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
                  const Text(
                    'Editing Hold',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Drag center to move • Drag edges to resize',
                    style: TextStyle(fontSize: 11, color: Colors.black87),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      TextButton.icon(
                        onPressed: () {
                          setState(() {
                            _detectedHolds.remove(_editingHold);
                            _editingHold = null;
                          });
                        },
                        icon: const Icon(Icons.delete, size: 16),
                        label: const Text('Delete', style: TextStyle(fontSize: 11)),
                        style: TextButton.styleFrom(foregroundColor: Colors.red),
                      ),
                      TextButton.icon(
                        onPressed: () {
                          setState(() {
                            _editingHold = null;
                          });
                        },
                        icon: const Icon(Icons.close, size: 16),
                        label: const Text('Deselect', style: TextStyle(fontSize: 11)),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
          
          const SizedBox(height: 12),
          // Stats
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Total: ${_detectedHolds.length}',
                style: const TextStyle(fontSize: 12, color: Colors.grey),
              ),
              Text(
                'Start: $startCount | Middle: ${selectedCount - startCount - finishCount} | Finish: $finishCount',
                style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // Action buttons
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
                  onPressed: selectedCount > 0 && startCount > 0 && finishCount > 0 && !_isAnalyzing 
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

  Widget _buildRoleButton(String label, HoldRole role, IconData icon, Color color) {
    final isSelected = _currentSelectionMode == role;
    
    return OutlinedButton.icon(
      onPressed: () {
        setState(() {
          _currentSelectionMode = role;
        });
      },
      icon: Icon(
        icon,
        size: 18,
        color: isSelected ? Colors.white : color,
      ),
      label: Text(
        label,
        style: TextStyle(
          fontSize: 12,
          color: isSelected ? Colors.white : color,
        ),
      ),
      style: OutlinedButton.styleFrom(
        backgroundColor: isSelected ? color : Colors.white,
        side: BorderSide(color: color, width: 2),
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 12),
      ),
    );
  }
}

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

    // Calculate scale to fit image in canvas
    final imageAspect = imageSize.width / imageSize.height;
    final canvasAspect = canvasSize.width / canvasSize.height;
    
    double scale;
    double offsetX = 0;
    double offsetY = 0;
    
    if (imageAspect > canvasAspect) {
      scale = canvasSize.width / imageSize.width;
      final scaledHeight = imageSize.height * scale;
      offsetY = (canvasSize.height - scaledHeight) / 2;
    } else {
      scale = canvasSize.height / imageSize.height;
      final scaledWidth = imageSize.width * scale;
      offsetX = (canvasSize.width - scaledWidth) / 2;
    }

    // Draw existing holds
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
        }
      } else {
        fillColor = Colors.grey.withOpacity(0.2);
        borderColor = Colors.grey;
      }

      if (isBeingEdited) {
        fillColor = Colors.orange.withOpacity(0.4);
        borderColor = Colors.orange;
      }

      final fillPaint = Paint()
        ..color = fillColor
        ..style = PaintingStyle.fill;
      canvas.drawRect(rect, fillPaint);

      final borderPaint = Paint()
        ..color = borderColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = isBeingEdited ? 4 : (hold.isSelected ? 3 : 2);
      canvas.drawRect(rect, borderPaint);

      // Draw resize handles if being edited
      if (isBeingEdited && isEditingMode && !isAddingHold) {
        final handleSize = 12.0;
        final handlePaint = Paint()
          ..color = Colors.orange
          ..style = PaintingStyle.fill;
        
        canvas.drawCircle(Offset(left, top), handleSize / 2, handlePaint);
        canvas.drawCircle(Offset(left + boxWidth, top), handleSize / 2, handlePaint);
        canvas.drawCircle(Offset(left, top + boxHeight), handleSize / 2, handlePaint);
        canvas.drawCircle(Offset(left + boxWidth, top + boxHeight), handleSize / 2, handlePaint);
        canvas.drawCircle(Offset(centerX, top), handleSize / 2, handlePaint);
        canvas.drawCircle(Offset(centerX, top + boxHeight), handleSize / 2, handlePaint);
        canvas.drawCircle(Offset(left, centerY), handleSize / 2, handlePaint);
        canvas.drawCircle(Offset(left + boxWidth, centerY), handleSize / 2, handlePaint);
      }

      if (hold.isSelected && !isBeingEdited) {
        final iconSize = 20.0;
        final iconX = centerX - iconSize / 2;
        final iconY = top - iconSize - 5;
        
        final iconPaint = Paint()
          ..color = borderColor
          ..style = PaintingStyle.fill;
        
        switch (hold.role) {
          case HoldRole.start:
            final path = Path()
              ..moveTo(iconX, iconY)
              ..lineTo(iconX, iconY + iconSize)
              ..lineTo(iconX + iconSize, iconY + iconSize / 2)
              ..close();
            canvas.drawPath(path, iconPaint);
            break;
          case HoldRole.finish:
            final path = Path()
              ..moveTo(iconX, iconY + iconSize)
              ..lineTo(iconX, iconY)
              ..lineTo(iconX + iconSize * 0.7, iconY + iconSize * 0.3)
              ..lineTo(iconX, iconY + iconSize * 0.6);
            canvas.drawPath(path, iconPaint);
            break;
          case HoldRole.middle:
            canvas.drawCircle(
              Offset(iconX + iconSize / 2, iconY + iconSize / 2),
              iconSize / 3,
              iconPaint,
            );
            break;
        }
      }

      final confidenceText = '${(hold.confidence * 100).toInt()}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: confidenceText,
          style: TextStyle(
            color: hold.isSelected ? borderColor : Colors.grey[600],
            fontSize: 10,
            fontWeight: FontWeight.bold,
            shadows: const [
              Shadow(
                color: Colors.white,
                blurRadius: 3,
              ),
            ],
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(centerX - textPainter.width / 2, top + boxHeight + 5),
      );
    }

    // Draw new hold preview
    if (isAddingHold && newHoldStart != null && newHoldEnd != null) {
      final startX = newHoldStart!.dx * scale + offsetX;
      final startY = newHoldStart!.dy * scale + offsetY;
      final endX = newHoldEnd!.dx * scale + offsetX;
      final endY = newHoldEnd!.dy * scale + offsetY;
      
      final left = startX < endX ? startX : endX;
      final top = startY < endY ? startY : endY;
      final width = (startX - endX).abs();
      final height = (startY - endY).abs();
      
      final previewRect = Rect.fromLTWH(left, top, width, height);
      
      final previewFillPaint = Paint()
        ..color = Colors.blue.withOpacity(0.3)
        ..style = PaintingStyle.fill;
      canvas.drawRect(previewRect, previewFillPaint);
      
      final previewBorderPaint = Paint()
        ..color = Colors.blue
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3
        ..strokeDashArray = [5, 5]; // Dashed line
      canvas.drawRect(previewRect, previewBorderPaint);
    }
  }

  @override
  bool shouldRepaint(covariant HoldMarkerPainter oldDelegate) {
    return oldDelegate.holds != holds || 
           oldDelegate.editingHold != editingHold ||
           oldDelegate.isEditingMode != isEditingMode ||
           oldDelegate.isAddingHold != isAddingHold ||
           oldDelegate.newHoldStart != newHoldStart ||
           oldDelegate.newHoldEnd != newHoldEnd;
  }
}

// Extension to draw dashed lines
extension DashedPath on Paint {
  set strokeDashArray(List<double> dashArray) {
    // Note: Flutter doesn't natively support dashed lines in Paint
    // This is a placeholder - the actual implementation would need PathMetrics
  }
}