import 'package:flutter/material.dart';
import 'dart:typed_data';

enum HoldRole {
  start,
  middle,
  finish,
}

class ClimbingHold {
  final String id;
  Offset position;  // Made mutable for editing
  final double confidence;
  double width;     // Made mutable for editing
  double height;    // Made mutable for editing
  bool isSelected;
  HoldRole role;

  ClimbingHold({
    required this.id,
    required this.position,
    required this.confidence,
    this.width = 40.0,
    this.height = 40.0,
    this.isSelected = false,
    this.role = HoldRole.middle,
  });
}

class ClimbingRoute {
  final String id;
  final String name;
  final String imagePath;
  final Uint8List? imageBytes;  // For web platform
  final List<ClimbingHold> holds;
  final DateTime createdAt;
  final String difficulty;

  ClimbingRoute({
    required this.id,
    required this.name,
    required this.imagePath,
    this.imageBytes,
    required this.holds,
    required this.createdAt,
    this.difficulty = 'V0',
  });
}