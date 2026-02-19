import argparse
import numpy as np
import tensorflow as tf

model_path = "models/centernet/best_model.keras"
output_path = 'models/centernet/model.tflite'
print(f"\nLoading model from {model_path} ...")
model = tf.keras.models.load_model(model_path)
model.summary(line_length=100)

# ── Convert WITHOUT quantisation ───────────────────────────────────────
# Dynamic-range quantisation makes Conv2DTranspose weights INT8 but keeps
# activations FLOAT32.  The TRANSPOSE_CONV TFLite kernel does not support
# that mixed mode and raises:
#   weights->type != input->type (INT8 != FLOAT32)
# So we export a pure FLOAT32 model.  The size penalty is ~3-4x vs INT8
# but inference is correct.  For a smaller model retrain with
# UpSampling2D→Conv2DTranspose replacement and full INT8 quant.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# No converter.optimizations — keep everything FLOAT32

print("Converting to TFLite (FLOAT32, no quantisation) ...")
tflite_model = converter.convert()

with open(output_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved → {output_path}  ({len(tflite_model)/1024:.1f} KB)")

# ── Print exact output tensor order ────────────────────────────────────
# TFLite does NOT guarantee output order matches the Keras dict order.
# Print this so you can confirm which index is heatmap / wh / offset.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n── Input tensors ────────────────────────────────────────────────")
for d in input_details:
    print(f"  [{d['index']}] '{d['name']}'  "
            f"shape={d['shape'].tolist()}  dtype={d['dtype'].__name__}")

print("\n── Output tensors ───────────────────────────────────────────────")
for d in output_details:
    print(f"  [{d['index']}] '{d['name']}'  "
            f"shape={d['shape'].tolist()}  dtype={d['dtype'].__name__}")

# ── Sanity-check: run a random inference and print value ranges ─────────
print("\n── Sanity-check inference (random input) ────────────────────────")
inp = input_details[0]
dummy = np.random.rand(*inp["shape"]).astype(np.float32)
interpreter.set_tensor(inp["index"], dummy)
interpreter.invoke()

for d in output_details:
    out = interpreter.get_tensor(d["index"])
    print(f"  '{d['name']}'  "
            f"shape={out.shape}  "
            f"min={out.min():.4f}  max={out.max():.4f}  "
            f"mean={out.mean():.4f}")

print("\n✓ Done.  Copy the output tensor order into HoldDetectionService "
        "if the Dart auto-detection needs overriding.\n")