

import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf 

from PIL import Image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
app = Flask(__name__)


MODEL_PATH = 'models/trained_model.h5'


model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')

def predict_label(img_path):
	img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
	img = tf.keras.preprocessing.image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	preds = model.predict(img)
	if preds==1:
		preds ="Pneumonia"
	else:
		preds="Normal"

	return preds


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)
if __name__ == '__main__':
        app.run(host="0.0.0.0",port=8000)
    
