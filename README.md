
# Reverse Image Search

So, in Reverse Image Search what happens is that user uploads an image and our system finds similar mages from the dataset/database and shows those images. Similar feature is there in Google and Google lens where you can search images by uploading an image instead of writing a text input.

This project is inspired by [This](https://www.youtube.com/watch?v=xanJe6e8Xuw&ab_channel=CampusX) youtube video, so if u need a detailed explanation how things are working here, you can definitely check it out. 😊

### Basic Working of project
The working of this project is quite simple. We have a dataset (you can download any dataset you want, make sure to download all those images and add them into images folder in the project file). So, there is a `app.py` file in the project, this is where we are extracting the features from the dataset images. In `main.py` we are creating a GUI using `streamlit` library. Here, user can upload an image, image's features will be extracted, compared with the features of dataset images and five most similar images will be shown.

### How to get this project?
If you want to use this project or want to run it on your device, firstly clone this project in your device using `git clone` command.

To run this projet you of course need a dataset. Download any dataset and add all those images in `images` folder in the code files. If you do not have any idea which dataset to download, I will recommend you to download [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from kaggle.

Then you have to run `app.py` to extract features. This will take some time to run depending upon your device's specs and the number of images from the dataset (in my case it took 4 about 4 hours 😬). This is because it is extracting features from all those images from the dataset.

Once, the `app.py` run sucessfully, this will add `embedding.py` (containing all the features) and `fileNames.py` (containing names of all the images0 in the project file. Now, project is ready to run.

### How to run?

Now, you should have an `images` folder with all the images along with `embedding.pkl` and `fileNames.pkl` files.

Now, simply open the project files in the terminal and enter

`streamlit run main.py`

This wll run the `main.py` on localhost. 

