// BinarizationApp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

// Image settings
const int IMAGE_WIDTH = 300;
const int IMAGE_HEIGHT = IMAGE_WIDTH;

// Window settings
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const string WINDOW_NAME = "Binarization App";

// Trackbar settings
const string TRACKBAR_NAME = "Threshold";
const int threshold_slider_max = 255;
int threshold_slider;
int thresholdValue;

// Button settings
const string BUTTON_NAME = "Binarize";

// Image Matrices
Mat initialImage;
Mat binaryImage;
Mat combinedImage;

// UI Element Callbacks
static void on_trackbar(int, void*);
void on_button_press(int, void*);

// Creates the UI for the binarization application
int main()
{
	// sets initial threshold value
	threshold_slider = threshold_slider_max / 2;

	// relative path within project to an image
	string imageLocation = "crackingacoldone.jpg";

	// load initial image
	initialImage = imread(samples::findFile(imageLocation), IMREAD_GRAYSCALE);

	// if image not found, return error
	if (initialImage.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	// create a copy of the intial image to a new binary image
	initialImage.copyTo(binaryImage);

	// horizontally concatenate images together to display in one window
	hconcat(initialImage, binaryImage, combinedImage);

	// Create window and UI elements
	namedWindow(WINDOW_NAME, WINDOW_NORMAL);
	resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
	createTrackbar(TRACKBAR_NAME, WINDOW_NAME, &threshold_slider, threshold_slider_max, on_trackbar);
	createButton(BUTTON_NAME, on_button_press);
	imshow(WINDOW_NAME, combinedImage);

	// Initialize callbacks for the elements
	on_trackbar(threshold_slider, 0);
	on_button_press(thresholdValue, 0);

	waitKey(0);
}

// Stores the threshold slider value in new variable
static void on_trackbar(int, void*) {
	thresholdValue = threshold_slider;
}

// Binarizes image
void on_button_press(int, void*) {
	// color value at a given location in initial image
	int pixelValue;

	// check every location in initial image
	for (int x = 0; x < initialImage.rows; x++) {
		for (int y = 0; y < initialImage.cols; y++) {
			// grab color value at each location in intial image
			pixelValue = initialImage.at<uchar>(x, y);
			// Set binary image value based on pixelValue when compared to threshold
			if (pixelValue < thresholdValue) {
				binaryImage.at<uchar>(x, y) = 0;
			}
			else {
				binaryImage.at<uchar>(x, y) = 255;
			}
		}
	}

	// horizontally concatenate images together to display them in same window
	hconcat(initialImage, binaryImage, combinedImage);
	// show combined image
	imshow(WINDOW_NAME, combinedImage);
}