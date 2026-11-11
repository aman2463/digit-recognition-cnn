"""Run this script in your Anaconda environment to convert the Keras model to TFLite."""
import tensorflow as tf

model = tf.keras.models.load_model("digit_model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("digit_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Converted! digit_model.tflite ({len(tflite_model)} bytes)")
