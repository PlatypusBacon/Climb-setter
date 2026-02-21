/// Describes what a single-finger touch gesture means in the route editor.
///
/// Keeping this as an enum (rather than scattered booleans) makes the gesture
/// routing in [CreateRouteScreen] exhaustive and easy to reason about.
enum EditorGestureMode {
  /// No gesture in progress — InteractiveViewer handles pan/zoom freely.
  idle,

  /// User is dragging an existing hold to move or resize it.
  editDrag,

  /// User is drawing a bounding box to add a new hold.
  addDraw,
}