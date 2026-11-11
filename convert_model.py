"""Run this script in your Anaconda environment to convert the Keras model to ONNX."""
import subprocess
import sys
import os

# Install tf2onnx if not present
subprocess.check_call([sys.executable, "-m", "pip", "install", "tf2onnx", "-q"])

import tensorflow as tf

model = tf.keras.models.load_model("digit_model.keras")

# Save as SavedModel first (avoids Keras tensor issues)
saved_model_dir = "temp_saved_model"
model.export(saved_model_dir)

# Convert SavedModel to ONNX using command line
subprocess.check_call([
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", saved_model_dir,
    "--output", "digit_model.onnx"
])

# Clean up temp folder
import shutil
shutil.rmtree(saved_model_dir, ignore_errors=True)

print("Converted! digit_model.onnx created successfully")
