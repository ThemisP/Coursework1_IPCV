/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <list>
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);
void sobel(Mat &input, Mat &magnitudeThreshold, Mat &directionOut);
void convolute(Mat &input, Mat &output, Mat &kernel);
void GaussianBlur2(cv::Mat &input, int size, cv::Mat &blurredOutput);
void magnitudeImages(Mat &x, Mat &y, Mat &output);
void directionAtan(Mat &x, Mat &y, Mat &output);
void normaliseImage(Mat &image, Mat &output);
void houghTransform(Mat &magnitudeThresh, Mat &direction,int *houghCircles);
void plotHoughSpace(int *houghCircles, int rows, int columns);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main(int argc, const char** argv)
{
	// 1. Read Input Image
	Mat frame = imread("dart15.jpg", CV_LOAD_IMAGE_COLOR);
	Mat magnitudeThreshold, direction;
	
	sobel(frame, magnitudeThreshold, direction);
	int *houghCircles = NULL;
	houghCircles = new int[magnitudeThreshold.rows * magnitudeThreshold.cols * (90)]();
	houghTransform(magnitudeThreshold, direction, houghCircles);
	plotHoughSpace(houghCircles, magnitudeThreshold.rows, magnitudeThreshold.cols);
	

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay(frame);

	// 4. Save Result Image
	imwrite("detected.jpg", frame);

	namedWindow("Display window", CV_WINDOW_AUTOSIZE);

	imshow("Display window", frame);
	waitKey(0);
	
	frame.release();
	
	

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

	// 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

	// 4. Draw box around faces found
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2);
	}

}

void plotHoughSpace(int *houghCircles, int rows, int columns) {
	if (houghCircles == NULL) return;
	Mat image;
	image.create(Size(rows, columns), CV_32F);
	int radius = 90;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			int sum = 0;
			for (int r = 0; r < radius; r++)
			{
				sum += houghCircles[i + rows * (j + columns * r)];
			}
			image.at<float>(i, j) = sum;
		}
	}
	normaliseImage(image, image);
	namedWindow("Hough Space", CV_WINDOW_AUTOSIZE);

	imshow("Hough Space", image);
}

void houghTransform(Mat &magnitudeThresh, Mat &direction, int *houghCircles) {
	int rows = magnitudeThresh.rows;
	int cols = magnitudeThresh.cols;
	int radiuslow = 10, radiushigh = 100;
	
	for (int x = 0; x<rows; x++)
		for (int y = 0; y<cols; y++)
			if ((int)magnitudeThresh.at<uchar>(x, y) != 0)
				for (int r = radiuslow; r < radiushigh; r++) {
					int x0 = round(x + r * sin(direction.at<float>(x, y)));
					int y0 = round(y + r * cos(direction.at<float>(x, y)));
					if (x0 >= 0 && y0>=0 && x0<magnitudeThresh.rows && y0 <magnitudeThresh.cols)
						houghCircles[x0 + rows * (y0 + cols * (r - radiuslow))] += 1;
					x0 = round(x - r * sin(direction.at<float>(x, y)));
					y0 = round(y - r * cos(direction.at<float>(x, y)));
					if (x0 >= 0 && y0>=0 && x0<magnitudeThresh.rows && y0 <magnitudeThresh.cols)
						houghCircles[x0 + rows * (y0 + cols * (r - radiuslow))] += 1;
				}
}

void sobel(Mat &input, Mat &magnitudeThreshold, Mat &directionOut) {
	Mat inputGray;
	cvtColor(input, inputGray, CV_BGR2GRAY);
	//Gaussian Blur Image
	Mat blurredIm;
	GaussianBlur2(inputGray, 11, blurredIm);

	//Derivative X
	float xdata[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	Mat kernelX(3, 3, CV_32F, xdata);
	Mat derX, derXNormalised;
	convolute(blurredIm, derX, kernelX);
	derXNormalised.create(derX.size(), derX.type());
	normaliseImage(derX, derXNormalised);

	//Derivative Y
	float ydata[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	Mat kernelY(3, 3, CV_32F, ydata);
	Mat derY, derYNormalised;
	convolute(blurredIm, derY, kernelY);
	derYNormalised.create(derY.size(), derY.type());
	normaliseImage(derY, derYNormalised);

	//Magnitude (arctan of derx and dery)
	Mat magnitude;
	magnitudeImages(derX, derY, magnitude);
	normaliseImage(magnitude, magnitude);
	threshold(magnitude, magnitude, 70, 255, CV_THRESH_BINARY);
	namedWindow("Magnitude", CV_WINDOW_AUTOSIZE);
	imshow("Magnitude", magnitude);
	magnitudeThreshold.create(magnitude.size(), magnitude.type());
	magnitudeThreshold = magnitude.clone();



	Mat direction, dirShow;
	directionAtan(derX, derY, direction);
	directionOut.create(direction.size(), direction.type());
	directionOut = direction.clone();
	normaliseImage(direction, dirShow);
	namedWindow("Direction", CV_WINDOW_AUTOSIZE);
	imshow("Direction", dirShow);

}

void normaliseImage(Mat &image, Mat &output) {
	float max = 0, min = 1000000;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			if (image.at<float>(i, j) > max)
				max = image.at<float>(i, j);
			if (image.at<float>(i, j) < min)
				min = image.at<float>(i, j);
		}
	}
	output = (image - min) / (max - min) * 255;
	output.convertTo(output, CV_8UC1);
}

void magnitudeImages(Mat &x, Mat &y, Mat &output) {
	output.create(x.size(), CV_32F);
	if (x.size() == y.size()) {
		for (int i = 0; i < x.rows; i++)
		{
			for (int j = 0; j < x.cols; j++)
			{
				output.at<float>(i, j) = sqrt(pow(x.at<float>(i, j), 2) + pow(y.at<float>(i, j), 2));
			}
		}
	}
}


void directionAtan(Mat &x, Mat &y, Mat &output) {
	output.create(x.size(), CV_32F);
	if (x.size() == y.size()) {
		for (int i = 0; i < x.rows; i++)
		{
			for (int j = 0; j < x.cols; j++)
			{
				output.at<float>(i, j) = atan2f(y.at<float>(i, j), x.at<float>(i, j));
			}
		}
	}
	output = output;
}

void convolute(Mat &input, Mat &output, Mat &kernel) {
	output.create(input.size(), CV_32F);
	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE);

	// now we can do the convoltion
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			float sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
			{
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
					float kernalval = kernel.at<float>(kernelx, kernely);

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			output.at<float>(i, j) = sum;
		}
	}
}

void GaussianBlur2(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE);

	// now we can do the convoltion
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			double sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
			{
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
					double kernalval = kernel.at<double>(kernelx, kernely);

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = sum;
		}
	}
}