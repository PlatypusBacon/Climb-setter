import 'package:flutter/material.dart';
import '../data/climbing_models.dart';
import '../utils/hold_role.dart';

/// The row of role-selector buttons shown in [CreateRouteScreen] when not in
/// editing mode. Tapping a button sets the active [HoldRole] so that
/// subsequent hold taps assign that role.
class HoldRoleSelector extends StatelessWidget {
  final HoldRole currentRole;
  final ValueChanged<HoldRole> onRoleChanged;

  const HoldRoleSelector({
    super.key,
    required this.currentRole,
    required this.onRoleChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      alignment: WrapAlignment.center,
      children: [
        _RoleButton(
          label: 'Start',
          role: HoldRole.start,
          icon: Icons.play_circle_filled,
          currentRole: currentRole,
          onTap: onRoleChanged,
        ),
        _RoleButton(
          label: 'Hand/Foot',
          role: HoldRole.middle,
          icon: Icons.circle,
          currentRole: currentRole,
          onTap: onRoleChanged,
        ),
        _RoleButton(
          label: 'Hand Only',
          role: HoldRole.hand,
          icon: Icons.back_hand,
          currentRole: currentRole,
          onTap: onRoleChanged,
        ),
        _RoleButton(
          label: 'Foot Only',
          role: HoldRole.foot,
          icon: Icons.directions_walk, // asset used inside _RoleButton
          currentRole: currentRole,
          onTap: onRoleChanged,
        ),
        _RoleButton(
          label: 'Finish',
          role: HoldRole.finish,
          icon: Icons.flag,
          currentRole: currentRole,
          onTap: onRoleChanged,
        ),
      ],
    );
  }
}

/// Single role button — circle avatar + label, consistent with [EditorToolbar].
class _RoleButton extends StatelessWidget {
  final String label;
  final HoldRole role;
  final IconData icon; // used only as fallback for non-foot roles
  final HoldRole currentRole;
  final ValueChanged<HoldRole> onTap;

  const _RoleButton({
    required this.label,
    required this.role,
    required this.icon,
    required this.currentRole,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final isActive = currentRole == role;
    final color = holdRoleColor(role);

    return GestureDetector(
      onTap: () => onTap(role),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            backgroundColor: isActive ? color : Colors.grey[300],
            radius: 18,
            child: holdRoleIcon(
              role,
              isActive ? Colors.white : color,
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