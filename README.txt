# Visual-Odometry-Perception-for-Autonomous-Robots
This project will estimate the 3D motion of the camera and the result will be a plot of the trajectory of the camera. This project involves the implementation of concepts of Visual Odometry.

File name: Visual_Odometry.py (Final code to get the output)
	   convertToColor.py (To undistort and convert to color images)

Instructions to run the program:
    1) Copy the folder named "provided_data" to this same location as "Visual_Odometry.py".
        i.e. "Visual_Odometry.py" and "provided_data" should be located in the same directory.
    2) Copy ReadCameraModel.py and UndistortImage.py into the "provided_data" directory (if not already present).
    3) Copy the "model" folder into the "provided_data" directory (if not already present).
    4) If you already have undistorted images:
        a)   Create a folder named "undistorted_input_images"
        b)   Copy the undistorted images into the directory "undistorted_input_images"
        c)   Run the command:

            python Visual_Odometry.py

      Otherwise, you can tell the program to undistort the input images by doing the following steps:
        a)   Create a folder called "input_images" in the same directory as the project file.
        b)   Copy the grayscale source images into "input_images"
        c)   Run the command (notice the undistort flag):

            python Visual_Odometry5.py --undistort

The structure should be as follows:
    ----------------------------------------------
    Project_directory
    +---- Visual_Odometry.py
    +---- provided_data
          +---- model
          +---- ReadCameraModel.py
          +---- UndistortImage.py
    +---- undistorted_input_images
    +---- input_images (if images are not already undistorted)
    ----------------------------------------------


We imported the following libraries:
* numpy
* cv2
* sys
* os
* argparse
* matplotlib

The trajectory output of our program can be found in the report and also in the zip file
