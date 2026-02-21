import 'package:flutter/material.dart';
import '../data/climbing_models.dart';
import '../utils/hold_role.dart';

/// Draws bounding-box overlays for all detected holds onto the canvas.
///
/// Extracted from [CreateRouteScreen] so the painter can be tested and
/// reused independently of the gesture/layout machinery.
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

  const HoldMarkerPainter({
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

  // ── Layout helpers ────────────────────────────────────────────────────────

  /// Returns the uniform scale factor and top-left offset that [Image.contain]
  /// would use to fit [imageSize] inside [canvasSize].
  ({double scale, double offsetX, double offsetY}) _fitContain() {
    final imageAspect = imageSize.width / imageSize.height;
    final canvasAspect = canvasSize.width / canvasSize.height;

    final double scale;
    double offsetX = 0;
    double offsetY = 0;

    if (imageAspect > canvasAspect) {
      scale = canvasSize.width / imageSize.width;
      offsetY = (canvasSize.height - imageSize.height * scale) / 2;
    } else {
      scale = canvasSize.height / imageSize.height;
      offsetX = (canvasSize.width - imageSize.width * scale) / 2;
    }

    return (scale: scale, offsetX: offsetX, offsetY: offsetY);
  }

  // ── Paint ─────────────────────────────────────────────────────────────────

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width == 0 || imageSize.height == 0) return;

    final (:scale, :offsetX, :offsetY) = _fitContain();

    for (final hold in holds) {
      _paintHold(canvas, hold, scale, offsetX, offsetY);
    }

    _paintNewHoldPreview(canvas, scale, offsetX, offsetY);
  }

  void _paintHold(Canvas canvas, ClimbingHold hold, double scale,
      double offsetX, double offsetY) {
    final centerX = hold.position.dx * scale + offsetX;
    final centerY = hold.position.dy * scale + offsetY;
    final boxWidth = hold.width * scale;
    final boxHeight = hold.height * scale;
    final left = centerX - boxWidth / 2;
    final top = centerY - boxHeight / 2;
    final rect = Rect.fromLTWH(left, top, boxWidth, boxHeight);
    final isBeingEdited = editingHold == hold;

    final (fillColor, borderColor) = _holdColors(hold, isBeingEdited);

    // Fill
    canvas.drawRect(
      rect,
      Paint()
        ..color = fillColor
        ..style = PaintingStyle.fill,
    );

    // Border
    canvas.drawRect(
      rect,
      Paint()
        ..color = borderColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = isBeingEdited ? 4 : (hold.isSelected ? 3 : 2),
    );

    // Resize handles when this hold is selected for editing
    if (isBeingEdited && isEditingMode && !isAddingHold) {
      _paintResizeHandles(
          canvas, left, top, boxWidth, boxHeight, centerX, centerY);
    }

    // Role icon above the box
    if (hold.isSelected && !isBeingEdited) {
      _paintRoleIcon(canvas, hold.role, borderColor, centerX, top);
    }

    // Confidence label below the box
    _paintConfidenceLabel(canvas, hold, borderColor, centerX, top, boxHeight);
  }

  (Color fill, Color border) _holdColors(
      ClimbingHold hold, bool isBeingEdited) {
    if (isBeingEdited) {
      return (Colors.orange.withOpacity(0.4), Colors.orange);
    }
    if (!hold.isSelected) {
      return (Colors.grey.withOpacity(0.2), Colors.grey);
    }
    final border = holdRoleColor(hold.role);
    return (border.withOpacity(0.3), border);
  }

  void _paintResizeHandles(Canvas canvas, double left, double top,
      double boxWidth, double boxHeight, double centerX, double centerY) {
    const hs = 12.0;
    final hp = Paint()
      ..color = Colors.orange
      ..style = PaintingStyle.fill;

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

  void _paintRoleIcon(Canvas canvas, HoldRole role, Color color,
      double centerX, double top) {
    const iconSize = 20.0;
    final iconX = centerX - iconSize / 2;
    final iconY = top - iconSize - 5;
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    switch (role) {
      case HoldRole.start:
        canvas.drawPath(
          Path()
            ..moveTo(iconX, iconY)
            ..lineTo(iconX, iconY + iconSize)
            ..lineTo(iconX + iconSize, iconY + iconSize / 2)
            ..close(),
          paint,
        );
      case HoldRole.finish:
        canvas.drawPath(
          Path()
            ..moveTo(iconX, iconY + iconSize)
            ..lineTo(iconX, iconY)
            ..lineTo(iconX + iconSize * 0.7, iconY + iconSize * 0.3)
            ..lineTo(iconX, iconY + iconSize * 0.6),
          paint,
        );
      case HoldRole.middle:
      case HoldRole.hand:
      case HoldRole.foot:
        canvas.drawCircle(
          Offset(iconX + iconSize / 2, iconY + iconSize / 2),
          iconSize / 3,
          paint,
        );
    }
  }

  void _paintConfidenceLabel(Canvas canvas, ClimbingHold hold, Color color,
      double centerX, double top, double boxHeight) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: '${(hold.confidence * 100).toInt()}%',
        style: TextStyle(
          color: hold.isSelected ? color : Colors.grey[600],
          fontSize: 10,
          fontWeight: FontWeight.bold,
          shadows: const [Shadow(color: Colors.white, blurRadius: 3)],
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    textPainter.paint(
      canvas,
      Offset(centerX - textPainter.width / 2, top + boxHeight + 5),
    );
  }

  void _paintNewHoldPreview(
      Canvas canvas, double scale, double offsetX, double offsetY) {
    if (!isAddingHold || newHoldStart == null || newHoldEnd == null) return;

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
        ..style = PaintingStyle.fill,
    );
    canvas.drawRect(
      previewRect,
      Paint()
        ..color = Colors.blue
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3,
    );
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