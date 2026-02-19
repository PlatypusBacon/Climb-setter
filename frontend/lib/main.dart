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
          seedColor: const Color.fromARGB(255, 224, 86, 255),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}