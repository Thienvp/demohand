import streamlit as st
import cv2
import numpy as np
import math
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from cvzone.HandTrackingModule import HandDetector
import joblib

# Constants
offset = 20
img_size = 200

# Load pre-trained models
class VGGFeatureExtractor:
    def __init__(self):
        IMG_SHAPE = (img_size, img_size, 3)
        vgg16_weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        vgg16_model = VGG16(weights=vgg16_weight_path, include_top=False, input_shape=IMG_SHAPE)
        self.vgg_feature_extractor = Sequential()
        self.vgg_feature_extractor.add(vgg16_model)
        self.vgg_feature_extractor.add(Flatten())
        self.vgg_feature_extractor.layers[0].trainable = False

    def transform(self, X):
        features = self.vgg_feature_extractor.predict(X)
        return np.array(features)

vgg = VGGFeatureExtractor()
svm = joblib.load("svm.pkl")

# Streamlit interface
st.title("Hand Gesture Recognition")
run = st.checkbox('Run')
frame_window = st.image([])
predicted_label = st.empty()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def postprocessing(label):
    if label == 0: return "left"
    elif label == 1: return "right"
    elif label == 2: return "up"
    return "None"

if run:
    while True:
        success, img = cap.read()
        if not success:
            st.write("Error: cannot get image!!")
            break
        img = cv2.flip(img, 1)
        hand, img = detector.findHands(img, draw=False)
        if hand:
            hand = hand[0]
            x, y, w, h = hand['bbox']
            ratio = h / w
            if ratio > 1:
                padding = (h - w) / 2
                imgCrop = img[y - offset: y + h + offset, x - math.ceil(padding) - offset: x + w + math.floor(padding) + offset]
            else:
                padding = (w - h) / 2
                imgCrop = img[y - math.ceil(padding) - offset: y + h + math.floor(padding) + offset, x - offset: x + w + offset]
            imgCrop = cv2.flip(imgCrop, 1)

            if imgCrop is not None and imgCrop.shape[0] * imgCrop.shape[1] != 0:
                imgCrop = cv2.resize(imgCrop, (img_size, img_size))
                frame_window.image(imgCrop, channels="RGB")
                imgProcessed = preprocess_image(imgCrop)
                features = vgg.transform(imgProcessed)
                prediction = svm.predict(features)
                predicted_label.text(f"Prediction: {postprocessing(prediction[0])}")

        # frame_window.image(img, channels="BGR")
        if not run:
            break

cap.release()
cv2.destroyAllWindows()
