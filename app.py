
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm #tells progress of the for loop
import pickle

model  = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False # it is false because we are not training

model = tensorflow.keras.Sequential([ # we are adding our own last layer i.e GlobalMaxPooling2D
    model,
    GlobalMaxPooling2D()
])

# print(model.summary())

# Creating a function which will extract the features from the images and convert them into (2048) form and then we will normalize them.
def extract_features(imagePath, model):
    img = image.load_img(imagePath, target_size= (224, 224)) #importing the image
    imgArray = image.img_to_array(img) #converting image to array
    expandImgArray = np.expand_dims(imgArray, axis= 0) # converting image array to 4D-array, having batches of images
    preprocessedImg = preprocess_input(expandImgArray) # Converting image array to different format which will be accepted by ResNet model. i.e converting rgb to bgr etc.
    result = model.predict(preprocessedImg).flatten() # predicting result and flattening it i.e convering to 1D array
    normalizedResult = result/norm(result) #normalizing result

    return normalizedResult

# adding file paths to a python list
fileNames = []

for files in os.listdir('images'):
    fileNames.append(os.path.join('images' ,files))

# Now we are creating a 2D-List of all the features of the images, will contain 2048 features for each image
featureList = []
for files in tqdm(fileNames):
    featureList.append(extract_features(files, model))

print(np.array(featureList).shape)

# saving the featureList and fileName into respective files so they can be used later
pickle.dump(featureList, open('embedding.pkl', 'wb'))   
pickle.dump(fileNames, open('fileNames.pkl', 'wb'))

