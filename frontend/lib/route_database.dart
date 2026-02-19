// route_database.dart
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'route_model.dart';


class RouteDatabase {
  static final RouteDatabase instance = RouteDatabase._();
  static Database? _db;
  RouteDatabase._();

  Future<Database> get database async {
    _db ??= await _initDb();
    return _db!;
  }

  Future<Database> _initDb() async {
    final dir = await getDatabasesPath();
    return openDatabase(
      join(dir, 'routes.db'),
      version: 1,
      onCreate: (db, version) => db.execute('''
        CREATE TABLE routes (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          difficulty TEXT NOT NULL,
          holds TEXT NOT NULL,
          image_path TEXT NOT NULL,
          annotated_image_path TEXT,
          created_at TEXT NOT NULL,
          image_width REAL NOT NULL,
          image_height REAL NOT NULL
        )
      '''),
    );
  }

  Future<void> insertRoute(SavedRoute route) async {
    final db = await database;
    await db.insert('routes', route.toMap(),
        conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<List<SavedRoute>> getAllRoutes() async {
    final db = await database;
    final maps = await db.query('routes', orderBy: 'created_at DESC');
    return maps.map(SavedRoute.fromMap).toList();
  }

  Future<void> deleteRoute(String id) async {
    final db = await database;
    await db.delete('routes', where: 'id = ?', whereArgs: [id]);
  }

  Future<void> updateAnnotatedPath(String id, String path) async {
    final db = await database;
    await db.update('routes',
      {'annotated_image_path': path},
      where: 'id = ?', whereArgs: [id],
    );
  }
}