import 'package:flutter/material.dart';
import 'dart:io';

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

// Models
class ClimbingHold {
  final String id;
  final Offset position;
  final HoldType type;
  bool isSelected;

  ClimbingHold({
    required this.id,
    required this.position,
    required this.type,
    this.isSelected = false,
  });
}

enum HoldType {
  jug,
  crimp,
  sloper,
  pinch,
  pocket,
  unknown,
}

class ClimbingRoute {
  final String id;
  final String name;
  final String imagePath;
  final List<ClimbingHold> holds;
  final DateTime createdAt;
  final String difficulty;

  ClimbingRoute({
    required this.id,
    required this.name,
    required this.imagePath,
    required this.holds,
    required this.createdAt,
    this.difficulty = 'V0',
  });
}

// Home Screen with Navigation
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
      LibraryScreen(routes: _savedRoutes),
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

// Create Route Screen
class CreateRouteScreen extends StatefulWidget {
  final Function(ClimbingRoute) onRouteSaved;

  const CreateRouteScreen({super.key, required this.onRouteSaved});

  @override
  State<CreateRouteScreen> createState() => _CreateRouteScreenState();
}

class _CreateRouteScreenState extends State<CreateRouteScreen> {
  String? _selectedImagePath;
  List<ClimbingHold> _detectedHolds = [];
  bool _isAnalyzing = false;

  void _selectImage() async {
    // In a real app, use image_picker package
    // For now, simulate image selection
    setState(() {
      _selectedImagePath = 'assets/climbing_wall.jpg'; // Placeholder
      _isAnalyzing = true;
    });

    // Simulate AI analysis delay
    await Future.delayed(const Duration(seconds: 2));

    setState(() {
      _detectedHolds = _generateMockHolds();
      _isAnalyzing = false;
    });
  }

  List<ClimbingHold> _generateMockHolds() {
    // Mock detected holds - in real app, this would be AI analysis
    return List.generate(
      15,
      (index) => ClimbingHold(
        id: 'hold_$index',
        position: Offset(
          50 + (index % 3) * 100.0,
          100 + (index ~/ 3) * 80.0,
        ),
        type: HoldType.values[index % HoldType.values.length],
      ),
    );
  }

  void _createRoute() {
    final selectedHolds =
        _detectedHolds.where((hold) => hold.isSelected).toList();

    if (selectedHolds.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select at least one hold')),
      );
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => SaveRouteScreen(
          imagePath: _selectedImagePath!,
          selectedHolds: selectedHolds,
          onSave: widget.onRouteSaved,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Create Route'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: _selectedImagePath == null
          ? _buildEmptyState()
          : _buildImageAnalysis(),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.photo_library_outlined,
            size: 100,
            color: Colors.grey[400],
          ),
          const SizedBox(height: 24),
          Text(
            'Select a climbing wall image',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: _selectImage,
            icon: const Icon(Icons.add_photo_alternate),
            label: const Text('Choose Image'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageAnalysis() {
    return Column(
      children: [
        Expanded(
          child: Stack(
            children: [
              Container(
                color: Colors.grey[300],
                child: Center(
                  child: Text(
                    'Climbing Wall Image\n(placeholder)',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey[600]),
                  ),
                ),
              ),
              if (_isAnalyzing)
                Container(
                  color: Colors.black54,
                  child: const Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(color: Colors.white),
                        SizedBox(height: 16),
                        Text(
                          'Analyzing holds...',
                          style: TextStyle(color: Colors.white, fontSize: 18),
                        ),
                      ],
                    ),
                  ),
                ),
              if (!_isAnalyzing)
                ..._detectedHolds.map((hold) => _buildHoldMarker(hold)),
            ],
          ),
        ),
        _buildBottomPanel(),
      ],
    );
  }

  Widget _buildHoldMarker(ClimbingHold hold) {
    return Positioned(
      left: hold.position.dx,
      top: hold.position.dy,
      child: GestureDetector(
        onTap: () {
          setState(() {
            hold.isSelected = !hold.isSelected;
          });
        },
        child: Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: hold.isSelected
                ? Colors.deepOrange.withOpacity(0.8)
                : Colors.blue.withOpacity(0.6),
            border: Border.all(
              color: Colors.white,
              width: 2,
            ),
          ),
          child: Center(
            child: Icon(
              hold.isSelected ? Icons.check : Icons.circle,
              color: Colors.white,
              size: 20,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBottomPanel() {
    final selectedCount =
        _detectedHolds.where((hold) => hold.isSelected).length;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 8,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Detected: ${_detectedHolds.length} holds',
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(
                'Selected: $selectedCount',
                style: TextStyle(
                  color: Theme.of(context).colorScheme.primary,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: _selectImage,
                  child: const Text('Change Image'),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: FilledButton(
                  onPressed: selectedCount > 0 ? _createRoute : null,
                  child: const Text('Create Route'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// Save Route Screen
class SaveRouteScreen extends StatefulWidget {
  final String imagePath;
  final List<ClimbingHold> selectedHolds;
  final Function(ClimbingRoute) onSave;

  const SaveRouteScreen({
    super.key,
    required this.imagePath,
    required this.selectedHolds,
    required this.onSave,
  });

  @override
  State<SaveRouteScreen> createState() => _SaveRouteScreenState();
}

class _SaveRouteScreenState extends State<SaveRouteScreen> {
  final _nameController = TextEditingController();
  String _selectedDifficulty = 'V0';

  final List<String> _difficulties = [
    'V0',
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6',
    'V7',
    'V8',
    'V9',
    'V10+'
  ];

  void _saveRoute() {
    if (_nameController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a route name')),
      );
      return;
    }

    final route = ClimbingRoute(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      name: _nameController.text,
      imagePath: widget.imagePath,
      holds: widget.selectedHolds,
      createdAt: DateTime.now(),
      difficulty: _selectedDifficulty,
    );

    widget.onSave(route);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Route saved successfully!')),
    );

    Navigator.popUntil(context, (route) => route.isFirst);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Save Route'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            TextField(
              controller: _nameController,
              decoration: const InputDecoration(
                labelText: 'Route Name',
                border: OutlineInputBorder(),
                hintText: 'e.g., The Crimper',
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Difficulty',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            DropdownButtonFormField<String>(
              value: _selectedDifficulty,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
              ),
              items: _difficulties.map((difficulty) {
                return DropdownMenuItem(
                  value: difficulty,
                  child: Text(difficulty),
                );
              }).toList(),
              onChanged: (value) {
                setState(() {
                  _selectedDifficulty = value!;
                });
              },
            ),
            const SizedBox(height: 24),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Route Summary',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    Text('Holds: ${widget.selectedHolds.length}'),
                    Text('Difficulty: $_selectedDifficulty'),
                  ],
                ),
              ),
            ),
            const Spacer(),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: _saveRoute,
                child: const Padding(
                  padding: EdgeInsets.all(16),
                  child: Text('Save to Library'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }
}

// Library Screen
class LibraryScreen extends StatelessWidget {
  final List<ClimbingRoute> routes;

  const LibraryScreen({super.key, required this.routes});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Route Library'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: routes.isEmpty
          ? _buildEmptyState(context)
          : _buildRouteList(context),
    );
  }

  Widget _buildEmptyState(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.collections_bookmark_outlined,
            size: 100,
            color: Colors.grey[400],
          ),
          const SizedBox(height: 24),
          Text(
            'No routes saved yet',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 8),
          Text(
            'Create your first climbing route!',
            style: TextStyle(color: Colors.grey[600]),
          ),
        ],
      ),
    );
  }

  Widget _buildRouteList(BuildContext context) {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: routes.length,
      itemBuilder: (context, index) {
        final route = routes[index];
        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          child: ListTile(
            leading: Container(
              width: 60,
              height: 60,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.image),
            ),
            title: Text(route.name),
            subtitle: Text(
              '${route.difficulty} • ${route.holds.length} holds\n${_formatDate(route.createdAt)}',
            ),
            isThreeLine: true,
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => RouteDetailScreen(route: route),
                ),
              );
            },
          ),
        );
      },
    );
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }
}

// Route Detail Screen
class RouteDetailScreen extends StatelessWidget {
  final ClimbingRoute route;

  const RouteDetailScreen({super.key, required this.route});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(route.name),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: double.infinity,
              height: 400,
              color: Colors.grey[300],
              child: Stack(
                children: [
                  Center(
                    child: Text(
                      'Route Image\n(placeholder)',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.grey[600]),
                    ),
                  ),
                  ...route.holds.map((hold) => Positioned(
                        left: hold.position.dx,
                        top: hold.position.dy,
                        child: Container(
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.deepOrange.withOpacity(0.8),
                            border: Border.all(color: Colors.white, width: 2),
                          ),
                          child: Center(
                            child: Text(
                              '${route.holds.indexOf(hold) + 1}',
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      )),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Chip(
                        label: Text(route.difficulty),
                        backgroundColor:
                            Theme.of(context).colorScheme.primaryContainer,
                      ),
                      const SizedBox(width: 8),
                      Chip(
                        label: Text('${route.holds.length} holds'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Route Details',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 8),
                  _buildDetailRow('Created', _formatDate(route.createdAt)),
                  _buildDetailRow('Total Holds', '${route.holds.length}'),
                  _buildDetailRow('Difficulty', route.difficulty),
                  const SizedBox(height: 16),
                  Text(
                    'Hold Sequence',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 8),
                  ...route.holds.asMap().entries.map((entry) {
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        children: [
                          Container(
                            width: 32,
                            height: 32,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: Theme.of(context).colorScheme.primary,
                            ),
                            child: Center(
                              child: Text(
                                '${entry.key + 1}',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Text(
                            _getHoldTypeName(entry.value.type),
                            style: Theme.of(context).textTheme.bodyLarge,
                          ),
                        ],
                      ),
                    );
                  }),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
          Text(value),
        ],
      ),
    );
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }

  String _getHoldTypeName(HoldType type) {
    return type.toString().split('.').last.toUpperCase();
  }
}