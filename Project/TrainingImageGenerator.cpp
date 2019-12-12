// TrainingImageGenerator.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <math.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// Window settings
const string WINDOW_NAME = "Training Image Generator";
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

// Image settings
const int BACKGROUND_HEIGHT = 300;
const int BACKGROUND_WIDTH = 300;

const int ANIMAL_HEIGHT = 100;
const int ANIMAL_WIDTH = 100;

// Image counts
const int NUM_BEAR_IMAGES = 23;
const int NUM_BACK_IMAGES = 13;

const int NUM_ELK_IMAGES = 11;

const int NUM_RACCOON_IMAGES = 11;

const int NUM_TEST_BEAR_IMAGES = 40;
const int NUM_TEST_NOT_BEAR_IMAGES = 40;

// Current image numbers
int currentBearImageNumber = 1;
int currentBackgroundImageNumber = 1;
int currentTrainingImageNumber = 1;
int currentTestingImageNumber = 1;

Mat animalImage;
Mat backgroundImage;

// Directory locations for the various images
const string bearImageFolderLocation = "BearClassificationImages/Bears/";
string bearImageName;

const string elkImageFolderLocation = "BearClassificationImages/Elks/";
string elkImageName;

const string raccoonImageFolderLocation = "BearClassificationImages/Raccoons/";
string raccoonImageName;

const string backgroundImageFolderLocation = "BearClassificationImages/Environments/";
string backgroundImageName;
string backgroundImageLocation;

const string trainingImageFolderLocation = "BearClassificationImages/TrainingImages/";
string trainingImageName;
string trainingImageLocation;

const string bearTestingImageFolderLocation = "BearClassificationImages/ColoredTestingImages/Bear/";
string bearTestingImageName;

const string notBearTestingImageFolderLocation = "BearClassificationImages/ColoredTestingImages/NotBear/";
string notBearTestingImageName;

const string grayTestingImageFolderLocation = "BearClassificationImages/GrayTestingImages/";
string grayTestingImageName;
string grayTestingImageLocation;

// Generates the training and testing images for the dataset
// Provides a brief view of a random image at the end
int main()
{	
	Mat tempBackgroundImage;

	// Defines space where animal images may be moved around on the background
	Vec2i topLeftAnimalCorner;
	Vec2i backgroundCenter = Vec2i(BACKGROUND_HEIGHT / 2, BACKGROUND_WIDTH / 2);
	Vec2i centeredAnimalTopLeftCorner = Vec2i(backgroundCenter[0] - ANIMAL_HEIGHT / 2, backgroundCenter[1] - ANIMAL_WIDTH / 2);
	
	int NUM_OF_IMAGES;
	string animalImageLocation;

	// Generate images for bears, elks, and raccoons
	for (int i = 0; i < 3; i++) {
		
		// Set NUM_OF_IMAGES for each iteration
		if (i == 0) {
			NUM_OF_IMAGES = NUM_BEAR_IMAGES;
		}
		else if (i == 1) {
			NUM_OF_IMAGES = NUM_ELK_IMAGES;
		}
		else if (i == 2) {
			NUM_OF_IMAGES = NUM_RACCOON_IMAGES;
		}

		// Go through all background images
		for (int x = 1; x <= NUM_BACK_IMAGES; x++) {
			backgroundImageName = "env";
			backgroundImageName += to_string(x);
			backgroundImageName += ".jpg";

			backgroundImageLocation = backgroundImageFolderLocation + backgroundImageName;
			backgroundImage = imread(samples::findFile(backgroundImageLocation), IMREAD_UNCHANGED);

			resize(backgroundImage, backgroundImage, Size(BACKGROUND_HEIGHT, BACKGROUND_WIDTH));

			// Go through all animal images for the iteration
			for (int y = 1; y <= NUM_OF_IMAGES; y++) {
				backgroundImage.copyTo(tempBackgroundImage);

				// Define animal image location for given iteration
				if (i == 0) {
					bearImageName = "bear";
					bearImageName += to_string(y);
					bearImageName += ".png";

					animalImageLocation = bearImageFolderLocation + bearImageName;
				}
				else if (i == 1) {
					elkImageName = "elk";
					elkImageName += to_string(y);
					elkImageName += ".png";

					animalImageLocation = elkImageFolderLocation + elkImageName;
				}
				else if (i == 2) {
					raccoonImageName = "raccoon";
					raccoonImageName += to_string(y);
					raccoonImageName += ".png";

					animalImageLocation = raccoonImageFolderLocation + raccoonImageName;
				}

				animalImage = imread(samples::findFile(animalImageLocation), IMREAD_UNCHANGED);

				resize(animalImage, animalImage, Size(ANIMAL_HEIGHT, ANIMAL_WIDTH));

				resize(animalImage, animalImage, Size(), (100 + (rand() % 101 - 50)) / 100.0f, (100 + (rand() % 101 - 50)) / 100.0f);

				topLeftAnimalCorner = Vec2i(centeredAnimalTopLeftCorner[0] + (rand() % 101) - 50, centeredAnimalTopLeftCorner[1] + (rand() % 101) - 50);

				// Overlay animal image on top of the background
				// ignores transparent pixels
				for (int r = 0; r < animalImage.rows; r++) {
					for (int c = 0; c < animalImage.cols; c++) {
						Vec4b animalVector = animalImage.at<Vec4b>(r, c);
						if (animalVector[3] > 0) {
							tempBackgroundImage.at<Vec3b>(r + topLeftAnimalCorner[0], c + topLeftAnimalCorner[1]) = Vec3b(animalVector[0], animalVector[1], animalVector[2]);
						}
					}
				}

				// Convert final image to grayscale
				cvtColor(tempBackgroundImage, tempBackgroundImage, COLOR_BGR2GRAY);

				trainingImageName = "trainingImage";
				trainingImageName += to_string(currentTrainingImageNumber);
				trainingImageName += ".jpg";

				trainingImageLocation = trainingImageFolderLocation + trainingImageName;

				// Save training image to training directory
				imwrite(trainingImageLocation, tempBackgroundImage);

				currentTrainingImageNumber++;
			}
		}
	}

	// Grays out the bear test images
	for (int i = 1; i <= NUM_TEST_BEAR_IMAGES; i++) {
		bearTestingImageName = "bear";
		bearTestingImageName += to_string(i);
		bearTestingImageName += ".jpg";

		animalImageLocation = bearTestingImageFolderLocation + bearTestingImageName;

		animalImage = imread(samples::findFile(animalImageLocation), IMREAD_GRAYSCALE);

		grayTestingImageName = "testingImage";
		grayTestingImageName += to_string(currentTestingImageNumber);
		grayTestingImageName += ".jpg";

		grayTestingImageLocation = grayTestingImageFolderLocation + grayTestingImageName;

		imwrite(grayTestingImageLocation, animalImage);
		
		currentTestingImageNumber++;
	}

	// Grays out elk and raccoon testing images
	for (int i = 1; i <= NUM_TEST_NOT_BEAR_IMAGES; i++) {
		notBearTestingImageName = "notbear";
		notBearTestingImageName += to_string(i);
		notBearTestingImageName += ".jpg";

		animalImageLocation = notBearTestingImageFolderLocation + notBearTestingImageName;

		animalImage = imread(samples::findFile(animalImageLocation), IMREAD_GRAYSCALE);

		grayTestingImageName = "testingImage";
		grayTestingImageName += to_string(currentTestingImageNumber);
		grayTestingImageName += ".jpg";

		grayTestingImageLocation = grayTestingImageFolderLocation + grayTestingImageName;

		imwrite(grayTestingImageLocation, animalImage);

		currentTestingImageNumber++;
	}

	namedWindow(WINDOW_NAME);
	cv::resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);

	cv::imshow(WINDOW_NAME, tempBackgroundImage);

	cv::waitKey(0);
}