from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("cifar10_model.h5")

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image).astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        img = Image.open(request.files["image"])
        processed = preprocess_image(img)
        result = model.predict(processed)
        prediction = classes[np.argmax(result)]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
