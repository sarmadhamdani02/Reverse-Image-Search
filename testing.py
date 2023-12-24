
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors # to calculate the distance between neighbors

import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm
from PIL import Image

# import cv2

model  = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False # it is false because we are not training

model = tensorflow.keras.Sequential([ # we are adding our own last layer i.e GlobalMaxPooling2D
    model,
    GlobalMaxPooling2D()
])

def extract_features(imagePath, model):
    img = image.load_img(imagePath, target_size= (224, 224)) #importing the image
    imgArray = image.img_to_array(img) #converting image to array
    expandImgArray = np.expand_dims(imgArray, axis= 0) # converting image array to 4D-array, having batches of images
    preprocessedImg = preprocess_input(expandImgArray) # Converting image array to different format which will be accepted by ResNet model. i.e converting rgb to bgr etc.
    result = model.predict(preprocessedImg).flatten() # predicting result and flattening it i.e convering to 1D array
    normalizedResult = result/norm(result) #normalizing result

    return normalizedResult

featureList = np.array(pickle.load(open("embedding.pkl", 'rb')))
fileNames = np.array(pickle.load(open("fileNames.pkl", 'rb')))

# print((featureList).shape)

# extracting features from the test images
imagePath = './TestImages/watch_testImage.jpeg'


# Calculating the nearest neighbors from the test image
neighbours = NearestNeighbors(n_neighbors= 5, algorithm= 'brute', metric="euclidean") # will find 6 nearest neighbours, compared with all the images and will find euclidean distances
neighbours.fit(featureList) #adding input data

distances, indices =  neighbours.kneighbors([extract_features(imagePath, model)])

# for files in tqdm(indices[0]):
#     tempImage = cv2.imread(fileNames[files])
#     cv2.imwrite(f'output_{files}.jpg', tempImage)


# cv2.waitKey(0)


for files in tqdm(indices[0]):
    tempImage = Image.open(fileNames[files])
    tempImage.show()
