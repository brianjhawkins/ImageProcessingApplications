// LoGAndHoughTransformationApp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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

bool edgeDetectedImage = false;
const float pi = 3.1415926535897;
const float DEG2RAD = pi / 180.0f;
const int THRESHOLD = 190;

static void on_trackbar(int, void*);
void on_log_button_press(int, void*);
void on_hough_button_press(int, void*);
int LinearInterpolation(float x, float x0, float y0, float x1, float y1);
float GetValue(Mat* m, int x, int y);
bool ZeroCrossing(Mat* m, int x, int y);

int main()
{
	trackbar_slider = trackbar_slider_max / 2;
	//string imageLocation = "crackingacoldone.jpg";
	string imageLocation = "fence.jpg";

	initialImage = imread(samples::findFile(imageLocation), IMREAD_GRAYSCALE);
	cout << initialImage.size << endl;

	if (initialImage.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	initialImage.copyTo(finalImage);
	hconcat(initialImage, finalImage, combinedImage);

	namedWindow(WINDOW_NAME, WINDOW_NORMAL);
	resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
	createTrackbar(TRACKBAR_NAME, WINDOW_NAME, &trackbar_slider, trackbar_slider_max, on_trackbar);
	createButton(LOG_BUTTON_NAME, on_log_button_press);
	createButton(HOUGH_BUTTON_NAME, on_hough_button_press);
	imshow(WINDOW_NAME, combinedImage);

	on_trackbar(trackbar_slider, 0);

	waitKey(0);
}

static void on_trackbar(int, void*) {
	sigmaValue = (float)trackbar_slider / 10;
	sigmaValue = max(0.1f, sigmaValue);

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

			for (int i = 0; i < filter.cols; i++) {
				for (int j = 0; j < filter.rows; j++) {
					int a = x - operatorSize + i;
					int b = y - operatorSize + j;
					result += GetValue(&filter, i, j) * GetValue(&initialImage, a, b) / 255.0f;
				}
			}

			//cout << x << ", " << y << endl;
			modifiedImage.at<float>(y, x) = result;
		}
	}

	// Zero-Crossing detection
	for (int x = 0; x < modifiedImage.cols; x++) {
		for (int y = 0; y < modifiedImage.rows; y++) {
			int finalValue = 0;
			float pixelValue = GetValue(&modifiedImage, x, y);
			
			// if pixel value approximately 0
			if (pixelValue > -1 && pixelValue < 1) {
				if (ZeroCrossing(&modifiedImage, x, y)) {
					finalValue = 255;
				}
			}
			finalImage.at<uchar>(y, x) = finalValue;
		}
	}

	edgeDetectedImage = true;

	hconcat(initialImage, finalImage, combinedImage);
	imshow(WINDOW_NAME, combinedImage);
}

void on_hough_button_press(int, void*) {
	if (edgeDetectedImage) {
		initialImage.copyTo(houghInitialImage);
		int accumulatorHeight;
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

		houghAccumulator = Mat(accumulatorHeight, accumulatorWidth, DataType<float>::type);
		houghAccumulator = 0;

		float centerX = imageWidth / 2;
		float centerY = imageHeight / 2;

		for (int y = 0; y < imageHeight; y++) {
			for (int x = 0; x < imageWidth; x++) {
				float imageValue = GetValue(&finalImage, x, y);
				if (imageValue > 250) {
					for (int d = 0; d < accumulatorWidth; d++) {
						float r = (((float)x - centerX) * cos((float)d * DEG2RAD)) + (((float)y - centerY) * sin((float)d * DEG2RAD));
						houghAccumulator.at<float>((int)(r + houghHeight), d) += 1;
					}
				}
			}
		}

		vector<int> lines;

		for (int r = 0; r < accumulatorHeight; r++) {
			for (int d = 0; d < accumulatorWidth; d++) {
				if (GetValue(&houghAccumulator, d, r) >= THRESHOLD) {
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

					if (localMaxima > GetValue(&houghAccumulator, d, r)) {
						continue;
					}
					
					int x1 = 0;
					int x2 = 0;
					int y1 = 0;
					int y2 = 0;

					if (d >= 45 && d <= 135) {
						x1 = 0;
						y1 = ((float)(r - (accumulatorHeight / 2)) - ((x1 - (imageWidth / 2)) * cos(d * DEG2RAD))) / sin(d * DEG2RAD) + (imageHeight / 2);
						x2 = imageWidth;
						y2 = ((float)(r - (accumulatorHeight / 2)) - ((x2 - (imageWidth / 2)) * cos(d * DEG2RAD))) / sin(d * DEG2RAD) + (imageHeight / 2);
					}
					else {
						y1 = 0;
						x1 = ((float)(r - (accumulatorHeight / 2)) - ((y1 - (imageWidth / 2)) * cos(d * DEG2RAD))) / sin(d * DEG2RAD) + (imageHeight / 2);
						y2 = imageHeight;
						x2 = ((float)(r - (accumulatorHeight / 2)) - ((y2 - (imageWidth / 2)) * cos(d * DEG2RAD))) / sin(d * DEG2RAD) + (imageHeight / 2);
					}

					lines.push_back(x1);
					lines.push_back(y1);
					lines.push_back(x2);
					lines.push_back(y2);
				}
			}
		}

		for (int i = 0; i < lines.size(); i += 4) {
			line(houghInitialImage, Point(lines[i], lines[i + 1]), Point(lines[i + 2], lines[i + 3]), Scalar(0, 0, 255), 2, 8);
		}

		hconcat(houghInitialImage, finalImage, combinedImage);

		houghAccumulator /= 255.0f;

		imshow("Hough Accumulator", houghAccumulator);
		imshow(WINDOW_NAME, combinedImage);
	}
	else {
		cout << "No edge detected image. Use LoG on image first." << endl;
	}
}

int LinearInterpolation(float x, float x0, float y0, float x1, float y1) {
	int y;

	y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0));

	return y;
}

float GetValue(Mat* m, int x, int y) {
	float value;

	if (x < 0 || x >= m->cols) {
		value = 0;
	}
	else if (y < 0 || y >= m->rows) {
		value = 0;
	}
	else {
		if (m->type() == CV_8U) {
			value = m->at<uchar>(y, x);
		}
		else {
			value = m->at<float>(y, x);
		}
	}

	return value;
}

bool ZeroCrossing(Mat* m, int x, int y) {
	bool b = false;

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