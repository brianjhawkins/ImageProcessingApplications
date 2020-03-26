# ImageProcessingApplications
Used OpenCV, Tensorflow, and Qt Libraries along with Google Colab to implement these programs. Not each project used all of the libraries, and more specific information is listed below for each program.

# Binarization
Libraries: OpenCV and Qt

Simply takes an input image and applies a binarization to it. The threshold for binarization can be set by a trackbar.

# Histogram Equalization
Libraries: OpenCV and Qt

Takes an input image and provides four trackbars which allow for the definition of levels for which the image will be equalized. After equalization, two histogram graphs will be output for comparison of the initial image's histogram and the equalized image's histogram.

# LoG and Hough Transform
Libraries: OpenCV and Qt

Takes an input image and provides a trackbar which allows for the modification of the operator size for the LoG edge detection. There are two buttons that can be pressed 'LoG' and 'Hough'. The 'LoG' button performs Laplacian of Gaussian edge detection on the image and outputs the initial image and the edge detection image for comparison. The 'Hough' button performs a Hough Transform on the edge detected image and outputs a modified input image with edge detected lines added to it and the edge detected image for comparison. It also outputs the accumulator used during the transform.

The Hough Transform logic was a slightly modified version of the tutorial by Bruno Keymolen at: http://www.keymolen.com/2013/05/hough-transformation-c-implementation.html

# Project
Libraries: OpenCV, Qt, and Tensorflow using Google Colab environment

Provides the files used in my Bear Classification project. Provides the code for the TrainingImageGenerator which was used to generate the dataset for training the Tensorflow neural network as well as the Google Colab file which contains the logic for the neural network itself.

The neural network file is a modified version of the Advanced Image Classification tutorial from Tensorflow.org at: https://www.tensorflow.org/tutorials/images/classification
