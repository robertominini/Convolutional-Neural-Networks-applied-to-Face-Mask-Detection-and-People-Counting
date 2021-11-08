# Convolutional Neural Networks applied to Face Mask Detection and People Counting
In this repository, you can find three different CNNs which can carry out live detection of faces maks and people in a frame. This repository represents the implemented code for the Bachelor Thesis of Roberto Minini 3076167, Bocconi University.


## Steps to run the code

Downloading the code won't be enought to make the program work. Follow the following steps in order to be able to run the code.

STEP 1: The YOLO model weights file is too heavy to upload on GitHub. Please go to https://pjreddie.com/darknet/yolo/ and head to the following section of the site:

![image](https://user-images.githubusercontent.com/60971557/120808211-2345e600-c549-11eb-8106-43a211937d60.png)

Download the weights for the YOLOv3 tiny version. To do this, just click on the orange text.

After doing this put the file in the yolov3 folder. The result should look like this:
![image](https://user-images.githubusercontent.com/60971557/120808781-b121d100-c549-11eb-8927-b040c6405e4f.png)

STEP 2: if you want to train the algorithm using train_mask_detector.py and the datasets used in the thesis, you will need to insert the datasets mnaully, since they were too heavy to upload on GitHub. If you just want to check the already trained algorithm, you don't need to do this step.

Small dataset: https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset
Expanded dataset: use both the pictures from the small dataset and the ones from the Face Mask Detection ~12K Images Dataset( https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)

After downloading, create two folders with the names "small_dataset" and "expanded_dataset" and be sure to put them in the main directory. Your directory should look like:

![image](https://user-images.githubusercontent.com/60971557/120811059-e29b9c00-c54b-11eb-8c57-932916e2653e.png)

In each dataset folder there should be 2 sub-folders named "with_mask" and "without_mask". In these 2 sub-folders you should put ALL the pictures corresponding to that category (the separation between train, test and validation is done later by the code, and it must NOT be done here).

It should look like this:

![image](https://user-images.githubusercontent.com/60971557/120809336-3b6a3500-c54a-11eb-9f1f-2914ac30344a.png)

## How make live predictions with a webcam

Just open detect_webcam, connect a webcam to your computer and run the code.

If you have more cameras, select the right one by changing the video_source integer in this fucntion:

![image](https://user-images.githubusercontent.com/60971557/120810067-06121700-c54b-11eb-8cce-d1531d29c2a2.png)

## How to train the model

Insert a database of your choice. Make sure to format the dataset folder in the way described in STEP 2 of the previous section. 
After doing this, open train_mask_detector and set the dataset_name variable with the exact name of your dataset folder.

![image](https://user-images.githubusercontent.com/60971557/120810695-8df82100-c54b-11eb-8f46-231e5b25ae5f.png)

After doing this, you are ready to run the code.



