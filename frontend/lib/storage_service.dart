import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/painting.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'climbing_models.dart';
import 'annotation_painter.dart'; // wherever RouteAnnotationPainter lives

class RouteStorageService {

  Future<String> saveWallImage(Uint8List imageBytes, String routeId) async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/routes/$routeId/wall.jpg');
    await file.parent.create(recursive: true);
    await file.writeAsBytes(imageBytes);
    return file.path;
  }

  Future<String> exportAnnotatedImage({
    required String routeId,
    required Uint8List imageBytes,
    required List<ClimbingHold> holds,
    required Size imageSize,
  }) async {
    final codec = await ui.instantiateImageCodec(imageBytes);
    final frame = await codec.getNextFrame();
    final wallImage = frame.image;

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    final size = Size(wallImage.width.toDouble(), wallImage.height.toDouble());

    canvas.drawImage(wallImage, Offset.zero, Paint());

    final painter = RouteAnnotationPainter(
      holds: holds,
      imageSize: imageSize,
    );
    painter.paint(canvas, size);

    final picture = recorder.endRecording();
    final img = await picture.toImage(wallImage.width, wallImage.height);
    final byteData = await img.toByteData(format: ui.ImageByteFormat.png);
    final pngBytes = byteData!.buffer.asUint8List();

    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/routes/$routeId/annotated.png');
    await file.parent.create(recursive: true);
    await file.writeAsBytes(pngBytes);

    return file.path;
  }

  Future<void> shareAnnotatedImage(String annotatedImagePath) async {
    await Share.shareXFiles(
      [XFile(annotatedImagePath)],
      text: 'Check out this climbing route!',
    );
  }
}