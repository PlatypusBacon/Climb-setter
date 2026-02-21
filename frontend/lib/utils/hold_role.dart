import 'package:flutter/material.dart';
import '../data/climbing_models.dart';

/// Returns the display colour associated with a [HoldRole].
/// Single source of truth — import this instead of duplicating switch blocks.
Color holdRoleColor(HoldRole role) {
  switch (role) {
    case HoldRole.start:
      return Colors.green;
    case HoldRole.finish:
      return Colors.red;
    case HoldRole.middle:
      return Colors.blue;
    case HoldRole.hand:
      return const Color.fromARGB(255, 33, 68, 243);
    case HoldRole.foot:
      return const Color.fromARGB(255, 159, 33, 243);
  }
}

/// Returns the human-readable label for a [HoldRole].
String holdRoleLabel(HoldRole role) {
  switch (role) {
    case HoldRole.start:
      return 'Start';
    case HoldRole.finish:
      return 'Finish';
    case HoldRole.middle:
      return 'Hand/Foot';
    case HoldRole.hand:
      return 'Hand Only';
    case HoldRole.foot:
      return 'Foot Only';
  }
}

/// Returns a widget (Image or Icon) representing the [HoldRole].
/// [color] is applied to both the image tint and icon colour.
Widget holdRoleIcon(HoldRole role, Color color, {double size = 20}) {
  if (role == HoldRole.foot) {
    return Image.asset(
      'assets/icon/foot.png',
      width: size,
      height: size,
      color: color,
    );
  }
  return Icon(_holdRoleIconData(role), size: size, color: color);
}

/// Returns the [IconData] for roles that use a Material icon.
/// Foot uses a custom asset, so it is not included here.
IconData _holdRoleIconData(HoldRole role) {
  switch (role) {
    case HoldRole.start:
      return Icons.play_circle_filled;
    case HoldRole.finish:
      return Icons.flag;
    case HoldRole.middle:
      return Icons.circle;
    case HoldRole.hand:
      return Icons.back_hand;
    case HoldRole.foot:
      return Icons.directions_walk; // fallback; prefer holdRoleIcon()
  }
}

/// Confidence badge colour: green ≥ 80 %, orange ≥ 60 %, red below.
Color confidenceBadgeColor(double confidence) {
  if (confidence >= 0.8) return Colors.green;
  if (confidence >= 0.6) return Colors.orange;
  return Colors.red;
}

/// Average confidence across a list of holds, formatted as a percentage string.
String averageConfidenceLabel(List holds) {
  if (holds.isEmpty) return '0%';
  final avg = holds.map((h) => h.confidence as double).reduce((a, b) => a + b) /
      holds.length;
  return '${(avg * 100).toInt()}%';
}