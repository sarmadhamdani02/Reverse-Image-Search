
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors # to calculate the distance between neighbors
import numpy as np
from numpy.linalg import norm

import streamlit as sl
import os
from PIL import Image

import pickle

sl.set_page_config(page_title="Reverse Image Search 🌟")
sl.title("Reverse Image Search")

# 1- Upload File
# 2- Load File``
# 3- Save it -> extract features
# 4- Comparisons
# 5- Swow Recommendations

featureList = np.array(pickle.load(open("embedding.pkl", 'rb')))
fileNames = np.array(pickle.load(open("fileNames.pkl", 'rb')))

def saveImage(uploadedFile): # function to save image in uploads folder
    try:
        with open(os.path.join("Uploads", uploadedFile.name), "wb") as f:
            f.write(uploadedFile.getbuffer())
            return 1
    except:
        return 0
    
# Creating a function which will extract the features from the images and convert them into (2048) form and then we will normalize them.
def extract_features(imagePath, model):
    img = image.load_img(imagePath, target_size= (224, 224)) #importing the image
    imgArray = image.img_to_array(img) #converting image to array
    expandImgArray = np.expand_dims(imgArray, axis= 0) # converting image array to 4D-array, having batches of images
    preprocessedImg = preprocess_input(expandImgArray) # Converting image array to different format which will be accepted by ResNet model. i.e converting rgb to bgr etc.
    result = model.predict(preprocessedImg).flatten() # predicting result and flattening it i.e convering to 1D array
    normalizedResult = result/norm(result) #normalizing result

    return normalizedResult

def showRecommendations(features, featureList): #comparision
    # Calculating the nearest neighbors from the test image
    neighbours = NearestNeighbors(n_neighbors= 5, algorithm= 'brute', metric="euclidean") # will find 5 nearest neighbours, compared with all the images and will find euclidean distances
    neighbours.fit(featureList) #adding input data

    distances, indices =  neighbours.kneighbors([features])

    return indices

model  = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False # it is false because we are not training

model = tensorflow.keras.Sequential([ # we are adding our own last layer i.e GlobalMaxPooling2D
    model,
    GlobalMaxPooling2D()
])

uploadedFile = sl.file_uploader("Upload an Image") #creating a file uploader

if uploadedFile is not None:
    if (saveImage(uploadedFile)):

        # show the image
        uploadedImage = Image.open(uploadedFile)
        sl.image(uploadedImage)

        # feature extraction
        features = extract_features(os.path.join("Uploads",uploadedFile.name) ,model)
        # sl.text(features)
        
        # showing Recommendations
        indices = showRecommendations(features, featureList)

        col1, col2, col3, col4, col5 = sl.columns(5)

        with col1:
            sl.image(fileNames[indices[0][0]])

        with col2:
            sl.image(fileNames[indices[0][1]])

        with col3:
            sl.image(fileNames[indices[0][2]])

        with col4:
            sl.image(fileNames[indices[0][3]])

        with col5:
            sl.image(fileNames[indices[0][4]])


        
    else:
        sl.header("Something went wrong while uploading the image")