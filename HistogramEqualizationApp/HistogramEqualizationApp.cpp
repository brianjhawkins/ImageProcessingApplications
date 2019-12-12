// HistogramEqualizationApp.cpp : This file contains the 'main' function. Program execution begins and ends there.
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
const int GRAY_LEVELS = 256;
int histogram[GRAY_LEVELS];
int equalizedHistogram[GRAY_LEVELS];
int sigmaOfHistograms[GRAY_LEVELS];
int q[GRAY_LEVELS];

// Window settings
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const string WINDOW_NAME = "Histogram Equalization App";

// Trackbar settings
const string A_TRACKBAR_NAME = "A";
const string B_TRACKBAR_NAME = "B";
const string C_TRACKBAR_NAME = "C";
const string D_TRACKBAR_NAME = "D";
const int trackbar_slider_max = 10000;
int a_trackbar_slider;
int b_trackbar_slider;
int c_trackbar_slider;
int d_trackbar_slider;
int aValue;
int bValue;
int cValue;
int dValue;

// Button settings
const string BUTTON_NAME = "Equalize Image";

// Image Matrices
Mat initialImage;
Mat equalizedImage;
Mat combinedImage;
Mat initialHistImage;
Mat equalizedHistImage;

int binWidth = 2;
int histWidth = GRAY_LEVELS * binWidth;
int histHeight = 400;
int intialHistMax;
int equalizedHistMax;

// UI element callbacks
static void on_a_trackbar(int, void*);
static void on_b_trackbar(int, void*);
static void on_c_trackbar(int, void*);
static void on_d_trackbar(int, void*);
void on_button_press(int, void*);

// Equalizes a histogram between given intervals [a, b] to [c, d]
void HistogramEqualizationWithLevels(int a, int b, int c, int d);

// Creates UI for histogram equalization application
int main()
{
	// Intial trackbar value settings
	a_trackbar_slider = 0;
	b_trackbar_slider = 0;
	c_trackbar_slider = trackbar_slider_max;
	d_trackbar_slider = trackbar_slider_max;	

	initialHistImage = Mat(histHeight, histWidth, IMREAD_GRAYSCALE);
	equalizedHistImage = Mat(histHeight, histWidth, IMREAD_GRAYSCALE);

	// relative path to image location within project
	//string imageLocation = "crackingacoldone.jpg";
	string imageLocation = "car.jpg";

	// load in initial image
	initialImage = imread(samples::findFile(imageLocation), IMREAD_GRAYSCALE);

	// it initial image is not found, return an error
	if (initialImage.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	// copy initial image values to equalized image
	initialImage.copyTo(equalizedImage);

	// horizontally concatenate initial image and equalized image to display both on screen
	hconcat(initialImage, equalizedImage, combinedImage);

	// Create window and UI elements
	namedWindow(WINDOW_NAME, WINDOW_NORMAL);
	resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
	createTrackbar(A_TRACKBAR_NAME, WINDOW_NAME, &a_trackbar_slider, trackbar_slider_max, on_a_trackbar);
	createTrackbar(B_TRACKBAR_NAME, WINDOW_NAME, &b_trackbar_slider, trackbar_slider_max, on_b_trackbar);
	createTrackbar(C_TRACKBAR_NAME, WINDOW_NAME, &c_trackbar_slider, trackbar_slider_max, on_c_trackbar);
	createTrackbar(D_TRACKBAR_NAME, WINDOW_NAME, &d_trackbar_slider, trackbar_slider_max, on_d_trackbar);
	createButton(BUTTON_NAME, on_button_press);
	imshow(WINDOW_NAME, combinedImage);

	// Call initial callbacks
	on_a_trackbar(a_trackbar_slider, 0);
	on_b_trackbar(b_trackbar_slider, 0);
	on_c_trackbar(c_trackbar_slider, 0);
	on_d_trackbar(d_trackbar_slider, 0);
	on_button_press(0, 0);

	waitKey(0);
}

// callback for a trackbar which sets aValue for interval [a,b]
static void on_a_trackbar(int, void*) {
	aValue = ((float)a_trackbar_slider / trackbar_slider_max) * (GRAY_LEVELS - 1);
}

// callback for b trackbar which sets bValue for interval [a,b]
static void on_b_trackbar(int, void*) {
	bValue = ((float)b_trackbar_slider / trackbar_slider_max) * (GRAY_LEVELS - 1);
}

// callback for c trackbar which sets cValue for interval [c,d]
static void on_c_trackbar(int, void*) {
	cValue = ((float)c_trackbar_slider / trackbar_slider_max) * (GRAY_LEVELS - 1);
}

// callback for d trackbar which sets dValue for interval[c, d]
static void on_d_trackbar(int, void*) {
	dValue = ((float)d_trackbar_slider / trackbar_slider_max) * (GRAY_LEVELS - 1);
}

// callback for when button is pressed, performs histogram equalization
void on_button_press(int, void*) {	
	initialHistImage = 0;
	equalizedHistImage = 0;

	// Clear values from arrays
	for (int i = 0; i < GRAY_LEVELS; i++) {
		histogram[i] = 0;
		equalizedHistogram[i] = 0;
		q[i] = 0;
	}

	intialHistMax = 0;
	equalizedHistMax = 0;

	// if a > 0, equalize histogram from [0,a] to [0,c]
	if (aValue > 0) {
		HistogramEqualizationWithLevels(0, aValue, 0, cValue);
	}
	// equalize histogram from [a,b] to [c,d]
	HistogramEqualizationWithLevels(aValue, bValue, cValue, dValue);
	// if b < 255, equalize histogram from [b,M] to [d,M]
	if (bValue < GRAY_LEVELS - 1) {
		HistogramEqualizationWithLevels(bValue, GRAY_LEVELS - 1, dValue, GRAY_LEVELS - 1);
	}

	// For each location in the initial image determine how the value
	// at that location is mapped to the equalized image using q[]
	for (int x = 0; x < initialImage.rows; x++) {
		for (int y = 0; y < initialImage.cols; y++) {
			equalizedImage.at<uchar>(x, y) = q[initialImage.at<uchar>(x, y)];
		}
	}

	for (int x = 0; x < initialImage.rows; x++) {
		for (int y = 0; y < initialImage.cols; y++) {
			if (equalizedHistogram[equalizedImage.at<uchar>(x, y)] > equalizedHistMax) {
				equalizedHistMax = equalizedHistogram[equalizedImage.at<uchar>(x, y)];
			}
		}
	}

	float max = (intialHistMax > equalizedHistMax) ? intialHistMax : equalizedHistMax;

	for (int i = 1; i < GRAY_LEVELS; i++) {
		line(initialHistImage,
			Point((binWidth * (i - 1)), histHeight - (histogram[i - 1] / max) * histHeight),
			Point((binWidth * i), histHeight - (histogram[i] / max) * histHeight),
			Scalar(255, 255, 255),
			2,
			8,
			0);
		line(equalizedHistImage,
			Point((binWidth * (i - 1)), histHeight - (equalizedHistogram[i - 1] / max) * histHeight),
			Point((binWidth * i), histHeight - (equalizedHistogram[i] / max) * histHeight),
			Scalar(255, 255, 255),
			2,
			8,
			0);
	}

	// horizontally concatenate initial image and equalized image to display both on screen
	hconcat(initialImage, equalizedImage, combinedImage);
	// show combined image on screen
	imshow(WINDOW_NAME, combinedImage);
	imshow("Initial Image Histogram", initialHistImage);
	imshow("Equalized Image Histogram", equalizedHistImage);
}

// Equalizes histogram of image segment based on input levels (a, b) to (c, d)
void HistogramEqualizationWithLevels(int a, int b, int c, int d) {
	float tempQ;

	// Generate initial histogram of image segment
	for (int x = 0; x < initialImage.rows; x++) {
		for (int y = 0; y < initialImage.cols; y++) {
			histogram[initialImage.at<uchar>(x, y)] += 1;
			if (histogram[initialImage.at<uchar>(x, y)] > intialHistMax) {
				intialHistMax = histogram[initialImage.at<uchar>(x, y)];
			}
		}
	}

	// Calculate all values for sigmaOfHistograms between (a, b)
	sigmaOfHistograms[a] = histogram[a];
	for (int i = a + 1; i <= b; i++) {
		sigmaOfHistograms[i] = sigmaOfHistograms[i - 1] + histogram[i];
	}

	// Set float value of constant used to calculate lookup table values
	tempQ = (float)(d - c) / sigmaOfHistograms[b];
	// Initialize lookup table and equalized histogram based on first intensity
	// level to be considered in (a, b) i.e. a
	q[a] = tempQ * sigmaOfHistograms[a] + c;
	equalizedHistogram[q[a]] = histogram[a];
	// Go through all remaining intensity levels in (a, b) and set remaining values of
	// the lookup table and the equalized histogram
	for (int p = a + 1; p <= b; p++) {
		q[p] = tempQ * sigmaOfHistograms[p] + c;
		equalizedHistogram[q[p]] += histogram[p];
	}
}