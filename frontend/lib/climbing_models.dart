import 'package:flutter/material.dart';
import 'dart:typed_data';

enum HoldRole { start, middle, hand, foot,finish }

class ClimbingHold {
  final String id;
  Offset position;
  final double confidence;
  double width;
  double height;
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

  Map<String, dynamic> toMap() => {
    'id': id,
    'position_dx': position.dx,
    'position_dy': position.dy,
    'confidence': confidence,
    'width': width,
    'height': height,
    'is_selected': isSelected ? 1 : 0,
    'role': role.name,
  };

  factory ClimbingHold.fromMap(Map<String, dynamic> map) => ClimbingHold(
    id: map['id'],
    position: Offset(map['position_dx'], map['position_dy']),
    confidence: map['confidence'],
    width: map['width'],
    height: map['height'],
    isSelected: map['is_selected'] == 1,
    role: HoldRole.values.byName(map['role']),
  );
}

class ClimbingRoute {
  final String id;
  final String name;
  final String imagePath;
  final Uint8List? imageBytes;
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