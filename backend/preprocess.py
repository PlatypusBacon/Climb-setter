import json
import os
from pathlib import Path

def convert_via_to_training(via_json_path, output_dir):
    with open(via_json_path, 'r') as f:
        via_data = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_id, file_data in via_data.items():
        filename = file_data['filename']
        regions = file_data.get('regions', {})

        holds = []

        for region in regions.values():
            shape = region['shape_attributes']
            name = shape['name']

            if name == 'ellipse':
                cx = shape['cx']
                cy = shape['cy']
                rx = shape['rx']
                ry = shape['ry']

                holds.append({
                    "x": cx,
                    "y": cy,
                    "width": rx * 2,
                    "height": ry * 2
                })

            elif name == 'polygon':
                xs = shape['all_points_x']
                ys = shape['all_points_y']

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min

                holds.append({
                    "x": cx,
                    "y": cy,
                    "width": w,
                    "height": h
                })

        output_json = {
            "holds": holds
        }

        out_path = output_dir / (Path(filename).stem + ".json")

        with open(out_path, 'w') as f:
            json.dump(output_json, f, indent=2)

        print(f"Converted {filename}")
if __name__ == "__main__":
    via_json_path = "data/label/3.json"
    output_dir = "data/label/test"
    convert_via_to_training(via_json_path, output_dir)