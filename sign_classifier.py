import keras
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd

# Function to format the image in the way the model expects it
def preprocessing(img):
  img = cv2.cvtColor(np.array(img),cv2.COLOR_BGR2GRAY)
  img = cv2.equalizeHist(img)
  img = img/255
  img = cv2.resize(img,(32,32))  
  img = img.reshape(1,32,32,1)
  return img
  
def sign_classifier(img, weights_file):
    # Load the model
    model = tf.keras.models.load_model(weights_file)
    signnames = pd.read_csv('german-traffic-signs/signnames.csv')


    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 32, 32, 1), dtype=np.float32)
    image = preprocessing(img)
    

    # Load the image into the array
    data[0] = image

    # run the inference
    prediction = model.predict_classes(data)
    return signnames.SignName[prediction[0]] 
    