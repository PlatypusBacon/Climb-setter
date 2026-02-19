import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io';
import 'dart:typed_data';
import 'climbing_models.dart';
import 'annotation_painter.dart';

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
  bool _isSequenceClimb = false; // new field

  final List<String> _difficulties = [
    'V0', 'V1', 'V2', 'V3', 'V4', 'V5',
    'V6', 'V7', 'V8', 'V9', 'V10+'
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
      imagePath: widget.imagePath ??
          'web_${DateTime.now().millisecondsSinceEpoch}',
      imageBytes: widget.imageBytes,
      imageSize: widget.imageSize, // pass through so detail screen can scale
      holds: widget.selectedHolds,
      createdAt: DateTime.now(),
      difficulty: _selectedDifficulty,
      isSequenceClimb: _isSequenceClimb,
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
              // Preview
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
                      CustomPaint(
                        painter: RouteAnnotationPainter(
                          holds: widget.selectedHolds,
                          imageSize: widget.imageSize,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // Route name
              TextField(
                controller: _nameController,
                decoration: const InputDecoration(
                  labelText: 'Route Name',
                  border: OutlineInputBorder(),
                  hintText: "e.g., Old Man's Sack",
                  prefixIcon: Icon(Icons.text_fields),
                ),
                textCapitalization: TextCapitalization.words,
              ),
              const SizedBox(height: 24),

              // Difficulty
              Text('Difficulty',
                  style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              DropdownButtonFormField<String>(
                value: _selectedDifficulty,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.bar_chart),
                ),
                items: _difficulties
                    .map((d) => DropdownMenuItem(value: d, child: Text(d)))
                    .toList(),
                onChanged: (value) =>
                    setState(() => _selectedDifficulty = value!),
              ),
              const SizedBox(height: 16),

              // ── Sequence climb checkbox ─────────────────────────────────
              Card(
                child: CheckboxListTile(
                  value: _isSequenceClimb,
                  onChanged: (v) =>
                      setState(() => _isSequenceClimb = v ?? false),
                  title: const Text(
                    'Sequence Climb',
                    style: TextStyle(fontWeight: FontWeight.w600),
                  ),
                  subtitle: const Text(
                    'Show numbered hold order when viewing this route',
                    style: TextStyle(fontSize: 12),
                  ),
                  secondary: const Icon(Icons.format_list_numbered),
                  controlAffinity: ListTileControlAffinity.leading,
                ),
              ),
              const SizedBox(height: 16),

              // Summary card
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Route Summary',
                          style: Theme.of(context).textTheme.titleMedium),
                      const SizedBox(height: 12),
                      _buildSummaryRow(
                        Icons.play_circle_filled,
                        'Start Holds',
                        '${widget.selectedHolds.where((h) => h.role == HoldRole.start).length}',
                      ),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                        Icons.sports_handball,
                        'Hand/Foot Holds',
                        '${widget.selectedHolds.where((h) => h.role == HoldRole.middle).length}',
                      ),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                        Icons.back_hand,
                        'Hand Only Holds',
                        '${widget.selectedHolds.where((h) => h.role == HoldRole.hand).length}',
                      ),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                        Icons.directions_walk,
                        'Foot Only Holds',
                        '${widget.selectedHolds.where((h) => h.role == HoldRole.foot).length}',
                      ),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                        Icons.flag,
                        'Finish Holds',
                        '${widget.selectedHolds.where((h) => h.role == HoldRole.finish).length}',
                      ),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                          Icons.bar_chart, 'Difficulty', _selectedDifficulty),
                      const SizedBox(height: 8),
                      _buildSummaryRow(
                          Icons.percent, 'Avg Confidence', _getAverageConfidence()),
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
        Text('$label: ',
            style: const TextStyle(fontWeight: FontWeight.w500)),
        Text(value),
      ],
    );
  }

  String _getAverageConfidence() {
    if (widget.selectedHolds.isEmpty) return '0%';
    final avg = widget.selectedHolds
            .map((h) => h.confidence)
            .reduce((a, b) => a + b) /
        widget.selectedHolds.length;
    return '${(avg * 100).toInt()}%';
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }
}