# Advanced_LaneLinesDetection
## Repo Link  
## 1. Dependencies
Python 3.x
NumPy
Matplotlib (for charting and visualising images)
OpenCV 3.x
Pickle (for storing the camera calibration matrix and distortion coefficients)
MoviePy (to process video files)
## 2. Pipeline
The various steps invovled in the pipeline are as follows, each of these has also been discussed in more detail in the sub sections below:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Apply a perspective transform to rectify image ("birds-eye view").
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Useage Example 
- Clone the repo
- Open the Git Bach window in the folder that you cloned the repo 
- Write this command  ./run.sh  'input video path''output video path' --debug  0/1
  - if the debug mode = 1, then the output video will contain all pipline stages
  
  ![image](https://user-images.githubusercontent.com/74071911/164306518-e362ba65-30f2-4976-87b0-ffdb0db29114.png)


  - if the debug mode = 0, then the output video will contain lane lines detection only 
  ![image](https://user-images.githubusercontent.com/74071911/164306786-9454f5bc-ae8f-4a64-b089-f97030bfd9eb.png)

  
