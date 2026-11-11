import base64
import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("digit_model.keras")


def preprocess(data_url):
    """Turn the canvas drawing (white on black) into a 28x28 MNIST-style input."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("L")

    # Crop tightly around the digit
    bbox = img.point(lambda p: 255 if p > 40 else 0).getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)

    # Fit into 20x20 without stretching, place on a 28x28 canvas
    scale = 20.0 / max(img.size)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))

    # Centre by centre of mass, like MNIST digits
    arr = np.array(canvas, dtype=np.float32)
    total = arr.sum()
    ys, xs = np.indices(arr.shape)
    cy = (ys * arr).sum() / total
    cx = (xs * arr).sum() / total
    arr = np.roll(arr, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))

    return (arr / 255.0).reshape(1, 28, 28, 1)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image = payload.get("image")
    arr = preprocess(image) if image else None
    if arr is None:
        return jsonify(error="Draw a digit first."), 400

    digit = int(np.argmax(model.predict(arr, verbose=0)[0]))
    return jsonify(digit=digit)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
