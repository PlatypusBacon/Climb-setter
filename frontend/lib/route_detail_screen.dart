import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io';
import 'climbing_models.dart';

class RouteDetailScreen extends StatelessWidget {
  final ClimbingRoute route;

  const RouteDetailScreen({super.key, required this.route});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(route.name),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Route image with holds
            Container(
              width: double.infinity,
              height: 400,
              child: Stack(
                children: [
                  route.imageBytes != null
                      ? Image.memory(
                          route.imageBytes!,
                          fit: BoxFit.contain,
                          width: double.infinity,
                          errorBuilder: (context, error, stackTrace) {
                            return Container(
                              color: Colors.grey[300],
                              child: Center(
                                child: Text(
                                  'Image not available',
                                  style: TextStyle(color: Colors.grey[600]),
                                ),
                              ),
                            );
                          },
                        )
                      : (!kIsWeb && route.imagePath.isNotEmpty && !route.imagePath.startsWith('web_')
                          ? Image.file(
                              File(route.imagePath),
                              fit: BoxFit.contain,
                              width: double.infinity,
                              errorBuilder: (context, error, stackTrace) {
                                return Container(
                                  color: Colors.grey[300],
                                  child: Center(
                                    child: Text(
                                      'Image not available',
                                      style: TextStyle(color: Colors.grey[600]),
                                    ),
                                  ),
                                );
                              },
                            )
                          : Container(
                              color: Colors.grey[300],
                              child: Center(
                                child: Text(
                                  'Image not available',
                                  style: TextStyle(color: Colors.grey[600]),
                                ),
                              ),
                            )),
                  // Hold markers
                  ...route.holds.asMap().entries.map((entry) {
                    final index = entry.key;
                    final hold = entry.value;
                    
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
                      case HoldRole.hand:
                        color = const Color.fromARGB(255, 33, 68, 243).withOpacity(0.3);
                        break;
                      case HoldRole.foot:
                        color = const Color.fromARGB(255, 159, 33, 243).withOpacity(0.3);
                        break;
                    }
                    
                    return Positioned(
                      left: hold.position.dx - hold.width / 2,
                      top: hold.position.dy - hold.height / 2,
                      child: Container(
                        width: hold.width,
                        height: hold.height,
                        decoration: BoxDecoration(
                          color: color.withOpacity(0.3),
                          border: Border.all(color: color, width: 2),
                        ),
                        child: Center(
                          child: Text(
                            '${index + 1}',
                            style: TextStyle(
                              color: color,
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                              shadows: const [
                                Shadow(
                                  color: Colors.white,
                                  blurRadius: 2,
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    );
                  }),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Tags
                  Row(
                    children: [
                      Chip(
                        label: Text(route.difficulty),
                        backgroundColor:
                            Theme.of(context).colorScheme.primaryContainer,
                      ),
                      const SizedBox(width: 8),
                      Chip(
                        label: Text('${route.holds.length} holds'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  
                  // Route Details
                  Text(
                    'Route Details',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 12),
                  _buildDetailRow(
                    Icons.calendar_today,
                    'Created',
                    _formatDate(route.createdAt),
                  ),
                  const SizedBox(height: 8),
                  _buildDetailRow(
                    Icons.sports_handball,
                    'Total Holds',
                    '${route.holds.length}',
                  ),
                  const SizedBox(height: 8),
                  _buildDetailRow(
                    Icons.bar_chart,
                    'Difficulty',
                    route.difficulty,
                  ),
                  const SizedBox(height: 8),
                  _buildDetailRow(
                    Icons.percent,
                    'Avg Confidence',
                    _getAverageConfidence(),
                  ),
                  const SizedBox(height: 24),
                  
                  // Hold Sequence
                  Text(
                    'Hold Sequence',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 12),
                  ...route.holds.asMap().entries.map((entry) {
                    final index = entry.key;
                    final hold = entry.value;
                    
                    // Choose color and icon based on role
                    Color color;
                    IconData icon;
                    String roleText;
                    
                    switch (hold.role) {
                      case HoldRole.start:
                        color = Colors.green;
                        icon = Icons.play_circle_filled;
                        roleText = 'Start';
                        break;
                      case HoldRole.finish:
                        color = Colors.red;
                        icon = Icons.flag;
                        roleText = 'Finish';
                        break;
                      case HoldRole.middle:
                        color = Colors.blue;
                        icon = Icons.circle;
                        roleText = 'Middle';
                        break;
                      case HoldRole.hand:
                        color = const Color.fromARGB(255, 33, 68, 243).withOpacity(0.3);
                        icon = Icons.circle;
                        roleText = 'Hand';
                        break;
                      case HoldRole.foot:
                        color = const Color.fromARGB(255, 159, 33, 243).withOpacity(0.3);
                        icon = Icons.circle;
                        roleText = 'Foot';
                        break;
                    }
                    
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Row(
                        children: [
                          Container(
                            width: 40,
                            height: 40,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: color,
                            ),
                            child: Center(
                              child: Text(
                                '${index + 1}',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Icon(icon, size: 16, color: color),
                                    const SizedBox(width: 4),
                                    Text(
                                      roleText,
                                      style: TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.bold,
                                        color: color,
                                      ),
                                    ),
                                  ],
                                ),
                                Text(
                                  'Position: (${hold.position.dx.toInt()}, ${hold.position.dy.toInt()})',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.grey[600],
                                  ),
                                ),
                              ],
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: _getConfidenceColor(hold.confidence),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              '${(hold.confidence * 100).toInt()}%',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ],
                      ),
                    );
                  }),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 20, color: Colors.grey[600]),
        const SizedBox(width: 12),
        Text(
          '$label: ',
          style: const TextStyle(fontWeight: FontWeight.w500),
        ),
        Text(value),
      ],
    );
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }

  String _getAverageConfidence() {
    if (route.holds.isEmpty) return '0%';
    
    final avg = route.holds
        .map((h) => h.confidence)
        .reduce((a, b) => a + b) / route.holds.length;
    
    return '${(avg * 100).toInt()}%';
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) {
      return Colors.green;
    } else if (confidence >= 0.6) {
      return Colors.orange;
    } else {
      return Colors.red;
    }
  }
}