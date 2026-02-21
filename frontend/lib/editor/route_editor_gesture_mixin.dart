import 'package:flutter/material.dart';
import '../data/climbing_models.dart';
import 'editor_mode.dart';

/// Mixin that provides gesture handling and coordinate math for the route
/// editor canvas. Consumes and mutates state declared in [CreateRouteScreen].
///
/// Separated so [CreateRouteScreen] only needs to own lifecycle + UI wiring.
mixin RouteEditorGestureMixin<T extends StatefulWidget> on State<T> {
  // ── State that must be provided by the host widget ───────────────────────
  // These are declared abstract-style: the mixin reads/writes them as fields
  // on the concrete State class via the getters/setters below.
  // In Dart, mixins on State share the same object, so these fields live on
  // _CreateRouteScreenState and the mixin accesses them directly.

  Size? get imageSize;
  List<ClimbingHold> get detectedHolds;
  bool get isEditingMode;
  bool get isAddingHold;
  HoldRole get currentSelectionMode;
  TransformationController get transformationController;

  // Mutable gesture state — owned by the host, written by this mixin.
  EditorGestureMode gestureMode = EditorGestureMode.idle;
  Offset? lastDragPosition;
  Offset? newHoldStart;
  ClimbingHold? editingHold;
  String? editingAction; // 'move' | 'resize'

  // ── Tap ──────────────────────────────────────────────────────────────────

  void handleTap(Offset tapPosition, BoxConstraints constraints) {
    if (imageSize == null || detectedHolds.isEmpty) return;
    final imageCoords = screenToImageCoords(tapPosition, constraints);
    if (imageCoords == null) return;

    final tapped = holdAt(imageCoords);
    if (tapped == null) return;

    setState(() {
      if (isEditingMode) {
        editingHold = tapped;
      } else {
        if (!tapped.isSelected) {
          tapped.isSelected = true;
          tapped.role = currentSelectionMode;
        } else if (tapped.role == currentSelectionMode) {
          tapped.isSelected = false;
        } else {
          tapped.role = currentSelectionMode;
        }
      }
    });
  }

  // ── Pan (add-hold box drawing) ───────────────────────────────────────────

  void handlePanStart(Offset position, BoxConstraints constraints) {
    if (imageSize == null) return;
    final imageCoords = screenToImageCoords(position, constraints);
    if (imageCoords == null) return;
    setState(() {
      gestureMode = EditorGestureMode.addDraw;
      newHoldStart = imageCoords;
      lastDragPosition = imageCoords;
    });
  }

  void handlePanUpdate(Offset position, BoxConstraints constraints) {
    if (gestureMode != EditorGestureMode.addDraw) return;
    final imageCoords = screenToImageCoords(position, constraints);
    if (imageCoords == null) return;
    setState(() => lastDragPosition = imageCoords);
  }

  // ── Scale (edit/select; multi-finger passes to InteractiveViewer) ────────

  void handleScaleStart(
      ScaleStartDetails details, BoxConstraints constraints) {
    if (details.pointerCount > 1) {
      gestureMode = EditorGestureMode.idle;
      return;
    }

    if (imageSize == null) return;
    final imageCoords =
        screenToImageCoords(details.focalPoint, constraints);
    if (imageCoords == null) return;

    if (isEditingMode) {
      final hold = holdAt(imageCoords);
      if (hold != null) {
        setState(() {
          gestureMode = EditorGestureMode.editDrag;
          editingHold = hold;
          lastDragPosition = imageCoords;
          editingAction = isNearEdge(imageCoords, hold) ? 'resize' : 'move';
        });
        return;
      }
    }

    gestureMode = EditorGestureMode.idle;
  }

  void handleScaleUpdate(
      ScaleUpdateDetails details, BoxConstraints constraints) {
    if (gestureMode == EditorGestureMode.idle || details.pointerCount > 1) {
      return;
    }

    final imageCoords =
        screenToImageCoords(details.focalPoint, constraints);
    if (imageCoords == null) return;

    if (gestureMode == EditorGestureMode.addDraw && newHoldStart != null) {
      setState(() => lastDragPosition = imageCoords);
      return;
    }

    if (gestureMode == EditorGestureMode.editDrag &&
        editingHold != null &&
        lastDragPosition != null) {
      final delta = imageCoords - lastDragPosition!;
      setState(() {
        if (editingAction == 'move') {
          editingHold!.position = Offset(
            editingHold!.position.dx + delta.dx,
            editingHold!.position.dy + delta.dy,
          );
        } else if (editingAction == 'resize') {
          editingHold!.width =
              (editingHold!.width + delta.dx * 2).clamp(20.0, 200.0);
          editingHold!.height =
              (editingHold!.height + delta.dy * 2).clamp(20.0, 200.0);
        }
        lastDragPosition = imageCoords;
      });
    }
  }

  void handleGestureEnd(VoidCallback addHold) {
    if (gestureMode == EditorGestureMode.addDraw &&
        newHoldStart != null &&
        lastDragPosition != null) {
      final width = (lastDragPosition!.dx - newHoldStart!.dx).abs();
      final height = (lastDragPosition!.dy - newHoldStart!.dy).abs();
      if (width > 10 && height > 10) {
        addHold(); // screen supplies the setState + list.add call
      }
    }

    setState(() {
      gestureMode = EditorGestureMode.idle;
      newHoldStart = null;
      lastDragPosition = null;
      editingAction = null;
    });
  }

  // ── Coordinate helpers ───────────────────────────────────────────────────

  Offset? screenToImageCoords(
      Offset viewportPosition, BoxConstraints constraints) {
    if (imageSize == null) return null;

    final inverseMatrix =
        Matrix4.inverted(transformationController.value);
    final contentPos =
        MatrixUtils.transformPoint(inverseMatrix, viewportPosition);

    final imageAspect = imageSize!.width / imageSize!.height;
    final containerAspect = constraints.maxWidth / constraints.maxHeight;

    double displayScale;
    double imageOffsetX = 0;
    double imageOffsetY = 0;

    if (imageAspect > containerAspect) {
      displayScale = constraints.maxWidth / imageSize!.width;
      imageOffsetY =
          (constraints.maxHeight - imageSize!.height * displayScale) / 2;
    } else {
      displayScale = constraints.maxHeight / imageSize!.height;
      imageOffsetX =
          (constraints.maxWidth - imageSize!.width * displayScale) / 2;
    }

    return Offset(
      ((contentPos.dx - imageOffsetX) / displayScale)
          .clamp(0.0, imageSize!.width),
      ((contentPos.dy - imageOffsetY) / displayScale)
          .clamp(0.0, imageSize!.height),
    );
  }

  ClimbingHold? holdAt(Offset imagePoint) {
    for (final hold in detectedHolds) {
      if (imagePoint.dx >= hold.position.dx - hold.width / 2 &&
          imagePoint.dx <= hold.position.dx + hold.width / 2 &&
          imagePoint.dy >= hold.position.dy - hold.height / 2 &&
          imagePoint.dy <= hold.position.dy + hold.height / 2) {
        return hold;
      }
    }
    return null;
  }

  bool isNearEdge(Offset point, ClimbingHold hold) {
    const edgeThreshold = 20.0;
    return (point.dx - (hold.position.dx - hold.width / 2)).abs() <
            edgeThreshold ||
        (point.dx - (hold.position.dx + hold.width / 2)).abs() <
            edgeThreshold ||
        (point.dy - (hold.position.dy - hold.height / 2)).abs() <
            edgeThreshold ||
        (point.dy - (hold.position.dy + hold.height / 2)).abs() <
            edgeThreshold;
  }

  // ── Zoom ─────────────────────────────────────────────────────────────────

  void zoomBy(double factor) {
    final s = transformationController.value.getMaxScaleOnAxis();
    final t = transformationController.value.getTranslation();
    transformationController.value = Matrix4.identity()
      ..translate(t.x, t.y)
      ..scale((s * factor).clamp(0.5, 5.0));
  }
}