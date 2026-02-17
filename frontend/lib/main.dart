import 'package:flutter/material.dart';
import 'home_screen.dart';

void main() {
  runApp(const ClimbingRouteApp());
}

class ClimbingRouteApp extends StatelessWidget {
  const ClimbingRouteApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Climbing Route Creator',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepOrange,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}