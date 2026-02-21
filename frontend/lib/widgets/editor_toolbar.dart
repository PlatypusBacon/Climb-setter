import 'package:flutter/material.dart';

/// The editing toolbar shown at the top of the bottom panel in [CreateRouteScreen].
///
/// Renders Edit/Done, Add, and zoom controls using a consistent
/// circle-avatar + label style. All state changes are delegated upward
/// via callbacks so this widget stays stateless and easy to test.
class EditorToolbar extends StatelessWidget {
  final bool isEditingMode;
  final bool isAddingHold;
  final VoidCallback onToggleEdit;
  final VoidCallback onToggleAdd;
  final VoidCallback onZoomIn;
  final VoidCallback onZoomOut;
  final VoidCallback onZoomReset;

  const EditorToolbar({
    super.key,
    required this.isEditingMode,
    required this.isAddingHold,
    required this.onToggleEdit,
    required this.onToggleAdd,
    required this.onZoomIn,
    required this.onZoomOut,
    required this.onZoomReset,
  });

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      alignment: WrapAlignment.center,
      children: [
        _ToolButton(
          label: isEditingMode ? 'Done' : 'Edit',
          icon: isEditingMode ? Icons.check : Icons.edit,
          color: Colors.orange,
          isActive: isEditingMode,
          onTap: onToggleEdit,
        ),
        if (isEditingMode)
          _ToolButton(
            label: isAddingHold ? 'Cancel' : 'Add',
            icon: isAddingHold ? Icons.close : Icons.add_box,
            color: Colors.blue,
            isActive: isAddingHold,
            onTap: onToggleAdd,
          ),
        _ToolButton(
          label: 'In',
          icon: Icons.zoom_in,
          color: Colors.grey[700]!,
          isActive: false,
          onTap: onZoomIn,
        ),
        _ToolButton(
          label: 'Out',
          icon: Icons.zoom_out,
          color: Colors.grey[700]!,
          isActive: false,
          onTap: onZoomOut,
        ),
        _ToolButton(
          label: 'Reset',
          icon: Icons.crop_free,
          color: Colors.grey[700]!,
          isActive: false,
          onTap: onZoomReset,
        ),
      ],
    );
  }
}

/// Internal circle-avatar + label button, reused by [EditorToolbar].
class _ToolButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final bool isActive;
  final VoidCallback onTap;

  const _ToolButton({
    required this.label,
    required this.icon,
    required this.color,
    required this.isActive,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            backgroundColor: isActive ? color : Colors.grey[300],
            radius: 18,
            child: Icon(
              icon,
              color: isActive ? Colors.white : color,
              size: 20,
            ),
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
}