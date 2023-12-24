
# Reverse Image Search

So, in Reverse Image Search what happens is that user uploads an image and our system finds similar mages from the dataset/database and shows those images. Similar feature is there in Google and Google lens where you can search images by uploading an image instead of writing a text input.

This project is inspired by [This](https://www.youtube.com/watch?v=xanJe6e8Xuw&ab_channel=CampusX) youtube video, so if u need a detailed explanation how things are working here, you can definitely check it out. 😊

### Basic Working of project
The working of this project is quite simple. We have a dataset from kaggle named as [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small). This dataset cotains 44,441 fashion related images. So, there is a `app.py` file in the project, this is where we are extracting the features from the dataset images. In `main.py` we are creating a GUI using `streamlit` library. Here, user can upload an image and five similar images from the dataset will be shown.

### How to get this project?
If you want to use this project or want to run it on your device, firstly clone this project in your device using `git clone` command.

In this git repo, the dataset is **NOT** included and **in order to run the code, you have to add image dataset in `image`**. You can either add the same data set as [above](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small), then you can directyly run the `main.py` file. But if you want to add your own dataset, you have to delete `embedding.pkl` and `fileNames.pkl` files and run the `app.py` to extract features of the images from the dataset.

### How to run?

To run the code, make sure to have dataset images in image folder in the project file and you should have features and file names of the dataset images in `embedding.pkl` and `fileNames.pkl` files, respectively.

Then, terminal, open the project file and enter,

`streamlit run main.py`

This wll run the `main.py` on localhost. 

