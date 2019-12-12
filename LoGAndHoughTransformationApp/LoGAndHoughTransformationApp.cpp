// LoGAndHoughTransformationApp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
// Hough Transform was following the implementation of Bruno Keymolen at: http://www.keymolen.com/2013/05/hough-transformation-c-implementation.html
// Implementation was modified to accomodate UI and data structure of this program

#include "pch.h"
#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

// Window settings
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const string WINDOW_NAME = "LoG And Hough Transformation App";

// Trackbar settings
const string TRACKBAR_NAME = "Sigma";
const int trackbar_slider_max = 50;
int trackbar_slider;
float sigmaValue;
int operatorSize;

// Button settings
const string LOG_BUTTON_NAME = "LoG";
const string HOUGH_BUTTON_NAME = "Hough";

// Image Matrices
Mat initialImage;
Mat houghInitialImage;
Mat modifiedImage;
Mat finalImage;
Mat houghAccumulator;
Mat combinedImage;

Mat filter;

// Hough Button settings
bool edgeDetectedImage = false;
const float pi = 3.1415926535897;
const float DEG2RAD = pi / 180.0f;
// Threshold for hough line drawing
const int THRESHOLD = 190;

static void on_trackbar(int, void*);
void on_log_button_press(int, void*);
void on_hough_button_press(int, void*);
int LinearInterpolation(float x, float x0, float y0, float x1, float y1);
float GetValue(Mat* m, int x, int y);
bool ZeroCrossing(Mat* m, int x, int y);

// Creates a UI for performing LoG and Hough Transforms
int main()
{
	// Initial trackbar value
	trackbar_slider = trackbar_slider_max / 2;

	// Defines where the image file is located
	//string imageLocation = "crackingacoldone.jpg";
	string imageLocation = "fence.jpg";

	// Reads in the initial image
	initialImage = imread(samples::findFile(imageLocation), IMREAD_GRAYSCALE);

	if (initialImage.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	// Initializes the final image as a copy of the initial image
	initialImage.copyTo(finalImage);

	// Concatenates the initial and final images together
	hconcat(initialImage, finalImage, combinedImage);

	// Initializes the UI elements and shows the concatenated image
	namedWindow(WINDOW_NAME, WINDOW_NORMAL);
	resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
	createTrackbar(TRACKBAR_NAME, WINDOW_NAME, &trackbar_slider, trackbar_slider_max, on_trackbar);
	createButton(LOG_BUTTON_NAME, on_log_button_press);
	createButton(HOUGH_BUTTON_NAME, on_hough_button_press);
	imshow(WINDOW_NAME, combinedImage);

	on_trackbar(trackbar_slider, 0);

	waitKey(0);
}

// Takes the trackbar value and uses it to calculate the operator size of the filter
static void on_trackbar(int, void*) {
	sigmaValue = (float)trackbar_slider / 10;
	sigmaValue = max(0.1f, sigmaValue);

	// Linearly interpolates the operator size based on the sigma value
	if (sigmaValue > 4.0f) {
		operatorSize = LinearInterpolation(sigmaValue, 4.0f, 25, 5.0f, 31);
	}
	else if (sigmaValue > 3.0f) {
		operatorSize = LinearInterpolation(sigmaValue, 3.0f, 21, 4.0f, 25);
	}
	else if (sigmaValue > 2.0f) {
		operatorSize = LinearInterpolation(sigmaValue, 2.0f, 15, 3.0f, 21);
	}
	else if (sigmaValue > 1.0f) {
		operatorSize = LinearInterpolation(sigmaValue, 1.0f, 7, 2.0f, 15);
	}
	else {
		operatorSize = LinearInterpolation(sigmaValue, 0.1f, 3, 1.0f, 7);
	}
}

// Performs LoG calculations on the initial image and outputs the edge detected image
void on_log_button_press(int, void*) {
	int pixelValue;

	destroyWindow("Hough Accumulator");

	filter = Mat(2 * operatorSize + 1, 2 * operatorSize + 1, DataType<float>::type);
	modifiedImage = Mat(initialImage.rows, initialImage.cols, DataType<float>::type);

	float sigmaSquared = sigmaValue * sigmaValue;
	float sigmaFour = sigmaSquared * sigmaSquared;
	float e = 2.71828;

	// Create LoG operator
	for (int x = -operatorSize; x <= operatorSize; x++) {
		for (int y = -operatorSize; y <= operatorSize; y++) {
			float rSquared = x * x + y * y;
			float eValue = pow(e, -((rSquared) / (2 * sigmaSquared)));

			float value = ((rSquared - 2 * sigmaSquared) / sigmaFour) * eValue;

			filter.at<float>(y + operatorSize, x + operatorSize) = value;
		}
	}

	// Convolution with operator
	for (int x = 0; x < initialImage.cols; x++) {
		for (int y = 0; y < initialImage.rows; y++) {
			float result = 0;

			// Cycle through filter values and sum the result with the value in the image
			for (int i = 0; i < filter.cols; i++) {
				for (int j = 0; j < filter.rows; j++) {
					int a = x - operatorSize + i;
					int b = y - operatorSize + j;
					result += GetValue(&filter, i, j) * GetValue(&initialImage, a, b) / 255.0f;
				}
			}

			//Set result in modified image
			modifiedImage.at<float>(y, x) = result;
		}
	}

	// Zero-Crossing detection
	for (int x = 0; x < modifiedImage.cols; x++) {
		for (int y = 0; y < modifiedImage.rows; y++) {
			int finalValue = 0;
			float pixelValue = GetValue(&modifiedImage, x, y);
			
			// if pixel value approximately 0
			if (pixelValue > -0.5 && pixelValue < 0.5) {
				if (ZeroCrossing(&modifiedImage, x, y)) {
					finalValue = 255;
				}
			}
			finalImage.at<uchar>(y, x) = finalValue;
		}
	}

	edgeDetectedImage = true;

	// Concatenate and show combined image
	hconcat(initialImage, finalImage, combinedImage);
	imshow(WINDOW_NAME, combinedImage);
}

// Performs Hough Transformation on an edge detected image
void on_hough_button_press(int, void*) {
	// Only works if LoG was pressed before it
	if (edgeDetectedImage) {
		// Copy initial image to hough initial image
		initialImage.copyTo(houghInitialImage);
		// Set accumulator values
		// Accumulator is the image of the p and theta space
		int accumulatorHeight;
		// represents -90 degrees to 90 degrees
		int accumulatorWidth = 180;
		int imageHeight = finalImage.rows;
		int imageWidth = finalImage.cols;
		float houghHeight;

		if (imageHeight > imageWidth) {
			houghHeight = (sqrt(2.0) * imageHeight) / 2.0;
		}
		else {
			houghHeight = (sqrt(2.0) * imageWidth) / 2.0;
		}

		accumulatorHeight = (houghHeight) * 2.0;

		// Initialize accumulator and set all values to 0
		houghAccumulator = Mat(accumulatorHeight, accumulatorWidth, DataType<float>::type);
		houghAccumulator = 0;

		// Center of accumulator
		float centerX = imageWidth / 2;
		float centerY = imageHeight / 2;

		// Scan through edge detected image to fill accumulator
		for (int y = 0; y < imageHeight; y++) {
			for (int x = 0; x < imageWidth; x++) {
				float imageValue = GetValue(&finalImage, x, y);
				// If edge detected pixel value is white
				// Plot p and theta space line in accumulator
				if (imageValue > 250) {
					for (int d = 0; d < accumulatorWidth; d++) {
						float r = (((float)x - centerX) * cos((float)d * DEG2RAD)) + (((float)y - centerY) * sin((float)d * DEG2RAD));
						houghAccumulator.at<float>((int)(r + houghHeight), d) += 1;
					}
				}
			}
		}

		vector<int> lines;

		// Scan through accumulator to find points of many lines intersecting
		for (int r = 0; r < accumulatorHeight; r++) {
			for (int d = 0; d < accumulatorWidth; d++) {
				// If the pixel value is greater than threshold
				// i.e. if more lines than the threshold intersect at a point
				if (GetValue(&houghAccumulator, d, r) >= THRESHOLD) {
					// Check if point is a local maxima within a 9x9 space around the pixel
					int localMaxima = GetValue(&houghAccumulator, d, r);
					for (int y = -4; y <= 4; y++) {
						for (int x = -4; x <= 4; x++) {
							if (GetValue(&houghAccumulator, d + x, r + y) > localMaxima) {
								localMaxima = GetValue(&houghAccumulator, d + x, r + y);
								x = 5;
								y = 5;
							}
						}
					}

					// If the pixel is not a local maxima, skip plotting a line for it
					if (localMaxima > GetValue(&houghAccumulator, d, r)) {
						continue;
					}
					
					// Coordinate values for the line (x1, y1) to (x2, y2)
					int x1 = 0;
					int x2 = 0;
					int y1 = 0;
					int y2 = 0;

					// If theta is between 45 and 135 degrees
					if (d >= 45 && d <= 135) {
						// y = (r - x*cos(d)) / sin(d)
						x1 = 0;
						y1 = ((float)(r - (accumulatorHeight / 2)) - ((x1 - (imageWidth / 2)) * cos(d * DEG2RAD))) / sin(d * DEG2RAD) + (imageHeight / 2);
						x2 = imageWidth;
						y2 = ((float)(r - (accumulatorHeight / 2)) - ((x2 - (imageWidth / 2)) * cos(d * DEG2RAD))) / sin(d * DEG2RAD) + (imageHeight / 2);
					}
					else {
						// x = (r - y*sin(d)) / cos(d)
						y1 = 0;
						x1 = ((float)(r - (accumulatorHeight / 2)) - ((y1 - (imageWidth / 2)) * cos(d * DEG2RAD))) / sin(d * DEG2RAD) + (imageHeight / 2);
						y2 = imageHeight;
						x2 = ((float)(r - (accumulatorHeight / 2)) - ((y2 - (imageWidth / 2)) * cos(d * DEG2RAD))) / sin(d * DEG2RAD) + (imageHeight / 2);
					}

					// Push coordinates into the lines list
					lines.push_back(x1);
					lines.push_back(y1);
					lines.push_back(x2);
					lines.push_back(y2);
				}
			}
		}

		// Plot lines on the hough initial image
		for (int i = 0; i < lines.size(); i += 4) {
			line(houghInitialImage, Point(lines[i], lines[i + 1]), Point(lines[i + 2], lines[i + 3]), Scalar(255, 255, 255), 2, 8);
		}

		// Concatenate hough initial image and final image
		hconcat(houghInitialImage, finalImage, combinedImage);

		// Divide all values in hough accumulator by 255 in order to normalize it
		// Create more visually interesting results rather than pure white
		houghAccumulator /= 255.0f;

		// Show hough accumulator and the combined image
		imshow("Hough Accumulator", houghAccumulator);
		imshow(WINDOW_NAME, combinedImage);
	}
	else {
		cout << "No edge detected image. Use LoG on image first." << endl;
	}
}

// Performs performs linear interpolation between two points with values attributed to the points
int LinearInterpolation(float x, float x0, float y0, float x1, float y1) {
	int y;

	y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0));

	return y;
}

// Returns a value from an input matrix
float GetValue(Mat* m, int x, int y) {
	float value;

	// Return 0 if the coordinate is outside of the matrix
	if (x < 0 || x >= m->cols) {
		value = 0;
	}
	else if (y < 0 || y >= m->rows) {
		value = 0;
	}
	// Return a value if the coordinate is inside the matrix
	else {
		// Grayscale matrix
		if (m->type() == CV_8U) {
			value = m->at<uchar>(y, x);
		}
		// Float matrix
		else {
			value = m->at<float>(y, x);
		}
	}

	return value;
}

// Performs zero crossing detection for a specific point in an input matrix
bool ZeroCrossing(Mat* m, int x, int y) {
	bool b = false;

	// Gets all of the adjacent values to the input point
	float left = GetValue(m, x - 1, y);
	float right = GetValue(m, x + 1, y);
	float top = GetValue(m, x, y - 1);
	float bottom = GetValue(m, x, y + 1);
	float topLeft = GetValue(m, x - 1, y - 1);
	float topRight = GetValue(m, x + 1, y - 1);
	float bottomLeft = GetValue(m, x - 1, y + 1);
	float bottomRight = GetValue(m, x + 1, y + 1);

	// Left-right check
	if (((left > 0) && (right < 0)) || ((left < 0) && (right > 0))) {
		b = true;
	}
	// Top-bottom check
	else if (((top > 0) && (bottom < 0)) || ((top < 0) && (bottom > 0))) {
		b = true;
	}
	// Top Left - Bottom Right check
	else if (((topLeft > 0) && (bottomRight < 0)) || ((topLeft < 0) && (bottomRight > 0))) {
		b = true;
	}
	// Bottom Left - Top Right check
	else if (((bottomLeft > 0) && (topRight < 0)) || ((bottomLeft < 0) && (topRight > 0))) {
		b = true;
	}

	return b;
}