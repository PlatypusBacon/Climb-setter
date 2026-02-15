pip install -r requirements.txt
python prepare_data.py --sample
python prepare_data.py --annotate --images ./raw_images --output ./dataset
python train_model.py --dataset ./dataset --output ./models --mobile --convert

mkdir -p frontend_flutter/assets
   cp models/model.tflite frontend_flutter/assets/

flutter pub get
flutter run