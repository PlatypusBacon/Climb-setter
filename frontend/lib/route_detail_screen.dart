import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io';
import 'climbing_models.dart';

class RouteDetailScreen extends StatefulWidget {
  final ClimbingRoute route;

  const RouteDetailScreen({super.key, required this.route});

  @override
  State<RouteDetailScreen> createState() => _RouteDetailScreenState();
}

class _RouteDetailScreenState extends State<RouteDetailScreen> {
  final GlobalKey _stackKey = GlobalKey();
  Size? _displayedImageRect;
  Offset? _displayedImageOffset;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _computeImageLayout());
  }

  /// Works out where the image actually sits inside the 400-px-tall Stack,
  /// given that Image uses BoxFit.contain (letterboxed).
  void _computeImageLayout() {
    final imageSize = widget.route.imageSize;
    final ctx = _stackKey.currentContext;
    if (imageSize == null || ctx == null) return;

    final box = ctx.findRenderObject() as RenderBox;
    final containerSize = box.size;

    final imgW = imageSize.width;
    final imgH = imageSize.height;
    final cntW = containerSize.width;
    final cntH = containerSize.height;

    // BoxFit.contain: uniform scale so the whole image fits
    final scale = (cntW / imgW) < (cntH / imgH)
        ? (cntW / imgW)
        : (cntH / imgH);

    final displayedW = imgW * scale;
    final displayedH = imgH * scale;

    setState(() {
      _displayedImageRect = Size(displayedW, displayedH);
      _displayedImageOffset = Offset(
        (cntW - displayedW) / 2,
        (cntH - displayedH) / 2,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final route = widget.route;

    return Scaffold(
      appBar: AppBar(
        title: Text(route.name),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Image + hold overlay ──────────────────────────────────────
            SizedBox(
              width: double.infinity,
              height: 400,
              child: Stack(
                key: _stackKey,
                children: [
                  Positioned.fill(child: _buildImage(route)),
                  if (_displayedImageRect != null &&
                      _displayedImageOffset != null)
                    ..._buildHoldMarkers(route),
                ],
              ),
            ),

            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Tags
                  Wrap(
                    spacing: 8,
                    runSpacing: 4,
                    children: [
                      Chip(
                        label: Text(route.difficulty),
                        backgroundColor:
                            Theme.of(context).colorScheme.primaryContainer,
                      ),
                      Chip(label: Text('${route.holds.length} holds')),
                      if (route.isSequenceClimb)
                        Chip(
                          label: const Text('Sequence'),
                          backgroundColor: Colors.purple[100],
                          avatar: const Icon(Icons.format_list_numbered,
                              size: 16),
                        ),
                    ],
                  ),
                  const SizedBox(height: 24),

                  Text('Route Details',
                      style: Theme.of(context).textTheme.titleLarge),
                  const SizedBox(height: 12),
                  _buildDetailRow(Icons.calendar_today, 'Created',
                      _formatDate(route.createdAt)),
                  const SizedBox(height: 8),
                  _buildDetailRow(Icons.sports_handball, 'Total Holds',
                      '${route.holds.length}'),
                  const SizedBox(height: 8),
                  _buildDetailRow(
                      Icons.bar_chart, 'Difficulty', route.difficulty),
                  const SizedBox(height: 8),
                  _buildDetailRow(Icons.percent, 'Avg Confidence',
                      _getAverageConfidence()),
                  const SizedBox(height: 24),

                  // ── Hold Sequence — only shown for sequence climbs ───────
                  if (route.isSequenceClimb) ...[
                    Text('Hold Sequence',
                        style: Theme.of(context).textTheme.titleLarge),
                    const SizedBox(height: 12),
                    ...route.holds.asMap().entries.map(
                        (e) => _buildSequenceRow(e.key, e.value)),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Hold overlay ────────────────────────────────────────────────────────

  List<Widget> _buildHoldMarkers(ClimbingRoute route) {
    final imgSize = route.imageSize!;
    final dispSize = _displayedImageRect!;
    final offset = _displayedImageOffset!;

    // Pixels-per-original-pixel
    final scale = dispSize.width / imgSize.width;

    return route.holds.asMap().entries.map((entry) {
      final index = entry.key;
      final hold = entry.value;

      final dispCX = hold.position.dx * scale + offset.dx;
      final dispCY = hold.position.dy * scale + offset.dy;
      final dispW = hold.width * scale;
      final dispH = hold.height * scale;

      final color = _roleColor(hold.role);

      return Positioned(
        left: dispCX - dispW / 2,
        top: dispCY - dispH / 2,
        width: dispW,
        height: dispH,
        child: Container(
          decoration: BoxDecoration(
            color: color.withOpacity(0.25),
            border: Border.all(color: color, width: 2),
          ),
          child: Center(
            child: route.isSequenceClimb
                ? Text(
                    '${index + 1}',
                    style: TextStyle(
                      color: color,
                      fontWeight: FontWeight.bold,
                      fontSize: 11,
                      shadows: const [
                        Shadow(color: Colors.white, blurRadius: 2)
                      ],
                    ),
                  )
                : const SizedBox.shrink(),
          ),
        ),
      );
    }).toList();
  }

  // ── Sequence list ───────────────────────────────────────────────────────

  Widget _buildSequenceRow(int index, ClimbingHold hold) {
    final color = _roleColor(hold.role);
    final roleText = _roleText(hold.role);

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          // Number bubble
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(shape: BoxShape.circle, color: color),
            child: Center(
              child: Text(
                '${index + 1}',
                style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 16),
              ),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    _roleIconWidget(hold.role, color, size: 16),
                    const SizedBox(width: 4),
                    Text(roleText,
                        style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            color: color)),
                  ],
                ),
                Text(
                  'Position: (${hold.position.dx.toInt()}, ${hold.position.dy.toInt()})',
                  style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                ),
              ],
            ),
          ),
          // Confidence badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: _getConfidenceColor(hold.confidence),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              '${(hold.confidence * 100).toInt()}%',
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
    );
  }

  // ── Helpers ─────────────────────────────────────────────────────────────

  Widget _buildImage(ClimbingRoute route) {
    if (route.imageBytes != null) {
      return Image.memory(
        route.imageBytes!,
        fit: BoxFit.contain,
        width: double.infinity,
        height: double.infinity,
        errorBuilder: (_, __, ___) => _imagePlaceholder(),
      );
    }
    if (!kIsWeb &&
        route.imagePath.isNotEmpty &&
        !route.imagePath.startsWith('web_')) {
      return Image.file(
        File(route.imagePath),
        fit: BoxFit.contain,
        width: double.infinity,
        height: double.infinity,
        errorBuilder: (_, __, ___) => _imagePlaceholder(),
      );
    }
    return _imagePlaceholder();
  }

  Widget _imagePlaceholder() => Container(
        color: Colors.grey[300],
        child: Center(
            child: Text('Image not available',
                style: TextStyle(color: Colors.grey[600]))),
      );

  Color _roleColor(HoldRole role) {
    switch (role) {
      case HoldRole.start:
        return Colors.green;
      case HoldRole.finish:
        return Colors.red;
      case HoldRole.middle:
        return Colors.blue;
      case HoldRole.hand:
        return const Color.fromARGB(255, 33, 68, 243);
      case HoldRole.foot:
        return const Color.fromARGB(255, 159, 33, 243);
    }
  }

  String _roleText(HoldRole role) {
    switch (role) {
      case HoldRole.start:
        return 'Start';
      case HoldRole.finish:
        return 'Finish';
      case HoldRole.middle:
        return 'Hand/Foot';
      case HoldRole.hand:
        return 'Hand Only';
      case HoldRole.foot:
        return 'Foot Only';
    }
  }

  /// Foot role uses assets/foot.png; all others use Material icons.
  Widget _roleIconWidget(HoldRole role, Color color, {double size = 20}) {
    if (role == HoldRole.foot) {
      return Image.asset(
        'assets/foot.png',
        width: size,
        height: size,
        color: color,
      );
    }
    final IconData icon;
    switch (role) {
      case HoldRole.start:
        icon = Icons.play_circle_filled;
        break;
      case HoldRole.finish:
        icon = Icons.flag;
        break;
      case HoldRole.middle:
        icon = Icons.circle;
        break;
      case HoldRole.hand:
        icon = Icons.back_hand;
        break;
      default:
        icon = Icons.circle;
    }
    return Icon(icon, size: size, color: color);
  }

  Widget _buildDetailRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 20, color: Colors.grey[600]),
        const SizedBox(width: 12),
        Text('$label: ',
            style: const TextStyle(fontWeight: FontWeight.w500)),
        Text(value),
      ],
    );
  }

  String _formatDate(DateTime date) =>
      '${date.day}/${date.month}/${date.year}';

  String _getAverageConfidence() {
    if (widget.route.holds.isEmpty) return '0%';
    final avg = widget.route.holds
            .map((h) => h.confidence)
            .reduce((a, b) => a + b) /
        widget.route.holds.length;
    return '${(avg * 100).toInt()}%';
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.6) return Colors.orange;
    return Colors.red;
  }
}