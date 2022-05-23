# Advanced_LaneLinesDetection
## Repo Link  [Here](https://github.com/Hager-Waleed/Advanced_LaneLinesDetection)
## 1. Dependencies

- Python 3.x
- NumPy
- Matplotlib (for charting and visualising images)
- OpenCV 3.x
- Pickle (for storing the camera calibration matrix and distortion coefficients)
- MoviePy (to process video files)
- Skiimag
- Scipy
- Pickle

## 2. Pipeline
The various steps invovled in the pipeline are as follows, each of these has also been discussed in more detail in the sub sections below:

-Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
- Finding cars using sliding-window technique and use your trained classifier to search for vehicles in images.
- Apply a perspective transform to rectify image ("birds-eye view").
- Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
- Detect lane pixels and fit to find the lane boundary.
- Estimate a bounding box for vehicles detected

## Useage Example 

- Clone the repo
- Open the Git Bach window in the folder that you cloned the repo 
- Write this command  ./run.sh  'input video path''output video path' --debug  0/1
  - if the debug mode = 1, then the output video will contain all pipline stages
  
![image](https://user-images.githubusercontent.com/74071911/169896623-d8797327-090c-4de6-827e-acbc4b26c2a8.png)



