import 'package:flutter/material.dart';
import '../data/climbing_models.dart';

/// Info + action panel shown at the bottom of the screen when a hold is
/// selected for editing (move / resize / delete).
///
/// All mutations are delegated upward — this widget is stateless.
class EditingHoldPanel extends StatelessWidget {
  final ClimbingHold hold;
  final VoidCallback onDelete;
  final VoidCallback onDeselect;

  const EditingHoldPanel({
    super.key,
    required this.hold,
    required this.onDelete,
    required this.onDeselect,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
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
                onPressed: onDelete,
                icon: const Icon(Icons.delete, size: 16),
                label: const Text('Delete', style: TextStyle(fontSize: 11)),
                style: TextButton.styleFrom(foregroundColor: Colors.red),
              ),
              TextButton.icon(
                onPressed: onDeselect,
                icon: const Icon(Icons.close, size: 16),
                label: const Text('Deselect', style: TextStyle(fontSize: 11)),
              ),
            ],
          ),
        ],
      ),
    );
  }
}