import 'dart:convert';
import 'climbing_models.dart';
import 'package:flutter/painting.dart';

class SavedRoute extends ClimbingRoute {
  final String? annotatedImagePath;

  SavedRoute({
    required super.id,
    required super.name,
    required super.imagePath,
    super.imageBytes,
    super.imageSize,
    required super.holds,
    required super.createdAt,
    super.difficulty,
    super.isSequenceClimb,
    this.annotatedImagePath,
  });

  Map<String, dynamic> toMap() => {
    'id': id,
    'name': name,
    'difficulty': difficulty,
    'holds': jsonEncode(holds.map((h) => h.toMap()).toList()),
    'image_path': imagePath,
    'annotated_image_path': annotatedImagePath,
    'created_at': createdAt.toIso8601String(),
    'image_width': imageSize?.width ?? 0,
    'image_height': imageSize?.height ?? 0,
    'is_sequence_climb': isSequenceClimb ? 1 : 0,
  };

  factory SavedRoute.fromMap(Map<String, dynamic> map) => SavedRoute(
    id: map['id'],
    name: map['name'],
    difficulty: map['difficulty'],
    holds: (jsonDecode(map['holds']) as List)
        .map((h) => ClimbingHold.fromMap(h as Map<String, dynamic>))
        .toList(),
    imagePath: map['image_path'],
    annotatedImagePath: map['annotated_image_path'],
    createdAt: DateTime.parse(map['created_at']),
    imageSize: Size(map['image_width'], map['image_height']),
    isSequenceClimb: map['is_sequence_climb'] == 1,
  );
}