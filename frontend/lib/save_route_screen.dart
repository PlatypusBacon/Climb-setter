import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io';
import 'dart:typed_data';
import 'climbing_models.dart';

class SaveRouteScreen extends StatefulWidget {
  final String? imagePath;
  final Uint8List? imageBytes;
  final Size? imageSize;
  final List<ClimbingHold> selectedHolds;
  final Function(ClimbingRoute) onSave;

  const SaveRouteScreen({
    super.key,
    this.imagePath,
    this.imageBytes,
    this.imageSize,
    required this.selectedHolds,
    required this.onSave,
  });

  @override
  State<SaveRouteScreen> createState() => _SaveRouteScreenState();
}

class _SaveRouteScreenState extends State<SaveRouteScreen> {
  final _nameController = TextEditingController();
  String _selectedDifficulty = 'V0';

  final List<String> _difficulties = [
    'V0',
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6',
    'V7',
    'V8',
    'V9',
    'V10+'
  ];

  void _saveRoute() {
    if (_nameController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a route name')),
      );
      return;
    }

    final route = ClimbingRoute(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      name: _nameController.text,
      imagePath: widget.imagePath ?? 'web_${DateTime.now().millisecondsSinceEpoch}',
      imageBytes: widget.imageBytes,  // Store bytes for web
      holds: widget.selectedHolds,
      createdAt: DateTime.now(),
      difficulty: _selectedDifficulty,
    );

    widget.onSave(route);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Route saved successfully!')),
    );

    Navigator.popUntil(context, (route) => route.isFirst);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Save Route'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Preview of selected route
              Container(
                width: double.infinity,
                height: 200,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.grey[300]!),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Stack(
                    children: [
                      if (widget.imageBytes != null)
                        Image.memory(
                          widget.imageBytes!,
                          fit: BoxFit.cover,
                          width: double.infinity,
                        )
                      else if (widget.imagePath != null && !kIsWeb)
                        Image.file(
                          File(widget.imagePath!),
                          fit: BoxFit.cover,
                          width: double.infinity,
                        )
                      else
                        Container(
                          color: Colors.grey[300],
                          child: const Center(
                            child: Icon(Icons.image, size: 50),
                          ),
                        ),
                      // Overlay with selected holds
                      CustomPaint(
                        painter: _RoutePreviewPainter(
                          holds: widget.selectedHolds,
                          imageSize: widget.imageSize,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),
              TextField(
                controller: _nameController,
                decoration: const InputDecoration(
                  labelText: 'Route Name',
                  border: OutlineInputBorder(),
                  hintText: 'e.g., The Crimper',
                  prefixIcon: Icon(Icons.text_fields),
                ),
                textCapitalization: TextCapitalization.words,
              ),
              const SizedBox(height: 24),
              Text(
                'Difficulty',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              DropdownButtonFormField<String>(
                value: _selectedDifficulty,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.bar_chart),
                ),
                items: _difficulties.map((difficulty) {
                  return DropdownMenuItem(
                    value: difficulty,
                    child: Text(difficulty),
                  );
                }).toList(),
                onChanged: (value) {
                  setState(() {
                    _selectedDifficulty = value!;
                  });
                },
              ),
              const SizedBox(height: 24),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Route Summary',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 12),
                      _buildSummaryRow(
                        Icons.play_circle_filled,
                        'Start Holds',
                        '${widget.selectedHolds.where((h) => h.role == HoldRole.start).length}',
                      ),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                        Icons.sports_handball,
                        'Middle Holds',
                        '${widget.selectedHolds.where((h) => h.role == HoldRole.middle).length}',
                      ),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                        Icons.flag,
                        'Finish Holds',
                        '${widget.selectedHolds.where((h) => h.role == HoldRole.finish).length}',
                      ),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                        Icons.bar_chart,
                        'Difficulty',
                        _selectedDifficulty,
                      ),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                        Icons.percent,
                        'Avg Confidence',
                        _getAverageConfidence(),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 32),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: _saveRoute,
                  child: const Padding(
                    padding: EdgeInsets.all(16),
                    child: Text('Save to Library'),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSummaryRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 20, color: Colors.grey[600]),
        const SizedBox(width: 8),
        Text(
          '$label: ',
          style: const TextStyle(fontWeight: FontWeight.w500),
        ),
        Text(value),
      ],
    );
  }

  String _getAverageConfidence() {
    if (widget.selectedHolds.isEmpty) return '0%';
    
    final avg = widget.selectedHolds
        .map((h) => h.confidence)
        .reduce((a, b) => a + b) / widget.selectedHolds.length;
    
    return '${(avg * 100).toInt()}%';
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }
}

class _RoutePreviewPainter extends CustomPainter {
  final List<ClimbingHold> holds;
  final Size? imageSize;

  _RoutePreviewPainter({
    required this.holds,
    this.imageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize == null || imageSize!.width == 0 || imageSize!.height == 0) {
      return;
    }

    // Calculate scale
    final scaleX = size.width / imageSize!.width;
    final scaleY = size.height / imageSize!.height;
    final scale = scaleX < scaleY ? scaleX : scaleY;

    for (int i = 0; i < holds.length; i++) {
      final hold = holds[i];
      final x = hold.position.dx * scale;
      final y = hold.position.dy * scale;
      final width = hold.width * scale;
      final height = hold.height * scale;

      // Choose color based on role
      Color color;
      switch (hold.role) {
        case HoldRole.start:
          color = Colors.green;
          break;
        case HoldRole.finish:
          color = Colors.red;
          break;
        case HoldRole.middle:
          color = Colors.blue;
          break;
      }

      final paint = Paint()
        ..color = color.withOpacity(0.7)
        ..style = PaintingStyle.fill;

      final borderPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;

      // Draw rectangle
      final rect = Rect.fromCenter(
        center: Offset(x, y),
        width: width,
        height: height,
      );
      canvas.drawRect(rect, paint);
      canvas.drawRect(rect, borderPaint);

      // Draw number
      final textPainter = TextPainter(
        text: TextSpan(
          text: '${i + 1}',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 10,
            fontWeight: FontWeight.bold,
            shadows: [
              Shadow(
                color: Colors.black,
                blurRadius: 2,
              ),
            ],
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(x - textPainter.width / 2, y - textPainter.height / 2),
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}