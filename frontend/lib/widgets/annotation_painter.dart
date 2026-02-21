import 'package:flutter/material.dart';
import '../data/climbing_models.dart';

class RouteAnnotationPainter extends CustomPainter {
  final List<ClimbingHold> holds;
  final Size? imageSize;

  RouteAnnotationPainter({
    required this.holds,
    this.imageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize == null || imageSize!.width == 0 || imageSize!.height == 0) {
      return;
    }

    final scaleX = size.width / imageSize!.width;
    final scaleY = size.height / imageSize!.height;
    final scale = scaleX < scaleY ? scaleX : scaleY;

    for (int i = 0; i < holds.length; i++) {
      final hold = holds[i];
      final x = hold.position.dx * scale;
      final y = hold.position.dy * scale;
      final width = hold.width * scale;
      final height = hold.height * scale;

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

      final paint = Paint()
        ..color = color.withOpacity(0.7)
        ..style = PaintingStyle.fill;

      final borderPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;

      final rect = Rect.fromCenter(
        center: Offset(x, y),
        width: width,
        height: height,
      );
      canvas.drawRect(rect, paint);
      canvas.drawRect(rect, borderPaint);

      final textPainter = TextPainter(
        text: TextSpan(
          text: '${i + 1}',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 10,
            fontWeight: FontWeight.bold,
            shadows: [Shadow(color: Colors.black, blurRadius: 2)],
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