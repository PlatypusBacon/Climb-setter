import 'package:flutter/material.dart';
import 'climbing_models.dart';
import 'create_route.dart';
import 'library_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;
  final List<ClimbingRoute> _savedRoutes = [];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  void _addRoute(ClimbingRoute route) {
    setState(() {
      _savedRoutes.add(route);
    });
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> screens = [
      CreateRouteScreen(onRouteSaved: _addRoute),
      LibraryScreen(),
    ];

    return Scaffold(
      body: screens[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: _onItemTapped,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.add_photo_alternate),
            label: 'Create Route',
          ),
          NavigationDestination(
            icon: Icon(Icons.library_books),
            label: 'Library',
          ),
        ],
      ),
    );
  }
}