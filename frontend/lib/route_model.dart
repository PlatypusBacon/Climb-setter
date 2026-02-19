import 'dart:convert';
import 'package:flutter/painting.dart';
import 'climbing_models.dart';

class SavedRoute {
  final String id;
  final String name;
  final String difficulty;
  final List<ClimbingHold> holds;
  final String imagePath;
  final String? annotatedImagePath;
  final DateTime createdAt;
  final Size imageSize;

  SavedRoute({
    required this.id,
    required this.name,
    required this.difficulty,
    required this.holds,
    required this.imagePath,
    this.annotatedImagePath,
    required this.createdAt,
    required this.imageSize,
  });

  Map<String, dynamic> toMap() => {
    'id': id,
    'name': name,
    'difficulty': difficulty,
    'holds': jsonEncode(holds.map((h) => h.toMap()).toList()),
    'image_path': imagePath,
    'annotated_image_path': annotatedImagePath,
    'created_at': createdAt.toIso8601String(),
    'image_width': imageSize.width,
    'image_height': imageSize.height,
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
  );
}