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

#define M_PI 3.1415926535897

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);
void sobel(Mat &input, Mat &magnitudeThreshold, Mat &directionOut, int rank);
void convolute(Mat &input, Mat &output, Mat &kernel);
void GaussianBlur2(cv::Mat &input, int size, cv::Mat &blurredOutput);
void magnitudeImages(Mat &x, Mat &y, Mat &output);
void directionAtan(Mat &x, Mat &y, Mat &output);
void normaliseImage(Mat &image, Mat &output);
void houghCircleDetect(Mat &magnitudeThresh, Mat &direction,int ***houghCircles);
void houghLineDetect(Mat &magnitudeThresh, Mat &direction,int **houghLines);
void plotHoughSpaceCircles(int ***houghCircles, int rows, int columns, int rank);
int plotHoughSpaceLines(Mat &frame, int **houghLines,int diagonal, int length,int threshold, int rank, Point startPos);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

int **malloc2dArray(int dim1, int dim2)
{
	int i, j;

	int **array = (int **)malloc(dim1 * sizeof(int *));

	for (i = 0; i < dim1; i++) {

		array[i] = (int *)malloc(dim2 * sizeof(int ));
		for (int j = 0; j < dim2; j++) {
			array[i][j] = 0;
		}

	}
	return array;

}

int ***malloc3dArray(int dim1, int dim2, int dim3)
{
	int i, j, k;

	int ***array = (int ***)malloc(dim1 * sizeof(int **));

	for (i = 0; i < dim1; i++) {

		array[i] = (int **)malloc(dim2 * sizeof(int *));

		for (j = 0; j < dim2; j++) {

			array[i][j] = (int *)malloc(dim3 * sizeof(int));
			for (k = 0; k < dim3; k++)
				array[i][j][k] = 0;
		}

	}
	return array;

}


/** @function main */
int main(int argc, const char** argv)
{
	string name = "dart15";
	// 1. Read Input Image
	Mat frame = imread(name+".jpg", CV_LOAD_IMAGE_COLOR);
	

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay(frame);

	// 4. Save Result Image
	imwrite(name+"_detected.jpg", frame);

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
		int x = faces[i].x, y = faces[i].y, width = faces[i].width, height = faces[i].height;
		
		
		std::cout << "Rect " << i << "has size: " << faces[i].width << "x" << faces[i].height  << std::endl;

		Mat magnitudeThreshold, direction;
		int xCheck = x - 0.25*width, yCheck = y - 0.25*height, endXCheck = x+width + 0.25*width, endYCheck = y + height + 0.25*height;
		if (xCheck < 0) xCheck = 0;
		if (endXCheck >= frame.cols) endXCheck = frame.cols-1;
		if (yCheck < 0) yCheck = 0;
		if (endYCheck >= frame.rows) endYCheck = frame.rows-1;

		std::cout << "Rect CHECK " << i << "has size: " << endXCheck -xCheck << "x" << endYCheck-yCheck << std::endl;
		std::cout << "Rect CHECK " << i << "," << endXCheck << ","<< xCheck << "," << endYCheck << "," << yCheck << std::endl;
		Rect roi(x, y, width, height);
		Mat cropped(frame, roi);

		sobel(cropped, magnitudeThreshold, direction, i);
		/*int ***houghCircles = NULL;
		houghCircles = malloc3dArray(magnitudeThreshold.rows, magnitudeThreshold.cols, 90);
		houghCircleDetect(magnitudeThreshold, direction, houghCircles);
		plotHoughSpaceCircles(houghCircles, magnitudeThreshold.rows, magnitudeThreshold.cols, i);*/

		int **houghLines = NULL;
		int diagonal = ceil(sqrt(pow(magnitudeThreshold.rows,2) + pow(magnitudeThreshold.cols,2)));
		houghLines = malloc2dArray(diagonal,360);
		houghLineDetect(magnitudeThreshold, direction, houghLines);
		int count = plotHoughSpaceLines(frame, houghLines, diagonal, magnitudeThreshold.cols, 70, i, Point(x, y));
		std::cout << "Number of lines detected in image " << i << " is: " << count << std::endl;

		if(count>10)
			rectangle(frame, Point(x, y), Point(x + width, y + height), Scalar(0, 255, 0), 2);
	}

}

void plotHoughSpaceCircles(int ***houghCircles, int rows, int columns, int rank) {
	if (houghCircles == NULL) return;
	Mat image;
	image.create(Size(columns, rows), CV_32F);
	int radius = 90;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			float sum = 0;
			for (int r = 0; r < radius; r++)
			{
				sum += houghCircles[i][j][r];
			}
			image.at<float>(i, j) = sum;
		}
	}
	normaliseImage(image, image);
	namedWindow("Hough Space Circles" + std::to_string(rank), CV_WINDOW_AUTOSIZE);

	imshow("Hough Space Circles"+std::to_string(rank), image);
}

int plotHoughSpaceLines(Mat &frame, int **houghLines, int diagonal, int length, int threshold,  int rank, Point startPos) {
	if (houghLines == NULL) return -1;
	Mat image;
	image.create(Size(360, diagonal), CV_32F);
	int count = 0;
	for (int i = 0; i < diagonal; i++)
	{
		for (int theta = 0; theta < 360; theta+=45)
		{
			for (int error = -5; error < 5; error++) {
				if ((theta + error) >= 0 && (theta + error) < 360) {
					int point = houghLines[i][(theta + error)];
					int xStart, yStart, xEnd, yEnd;
					image.at<float>(i, (theta + error)) = point;
					if (point > threshold) {
						//image.at<float>(theta, i) = point;
						//if ((theta + error) % 180 == 0) {
						//	yStart = 0;
						//	xStart = round(i / cos((theta + error) * M_PI / 180));
						//	yEnd = length;
						//	xEnd = round(i / cos((theta + error) * M_PI / 180));

						//}
						//else if ((theta + error) % 90 == 0) {
						//	xStart = 0;
						//	yStart = round(i / sin((theta + error) * M_PI / 180));;
						//	xEnd = length;
						//	yEnd = round(i / sin((theta + error) * M_PI / 180));;
						//}
						//else {
						//	xStart = 0;
						//	yStart = round(i / sin((theta + error) * M_PI / 180));
						//	xEnd = length;
						//	yEnd = round((i - xEnd * cos((theta + error) * M_PI / 180)) / (sin((theta + error) * M_PI / 180)));
						//}
						////cout << "start: " << xStart << "," << yStart << "  end: " << xEnd << "," << yEnd << endl;
						//arrowedLine(frame, Point(xStart + startPos.x, yStart + startPos.y), Point(xEnd + startPos.x, yEnd + startPos.y), Scalar(0, 0, 255), 1, 4, 0, 0.02);
						count++;
					}
				}
			}
		}
	}
	/*normaliseImage(image, image);
	namedWindow("Hough Space Lines" + std::to_string(rank), CV_WINDOW_AUTOSIZE);

	imshow("Hough Space Lines" + std::to_string(rank), image);*/
	return count;
}

void houghCircleDetect(Mat &magnitudeThresh, Mat &direction, int ***houghCircles) {
	int rows = magnitudeThresh.rows;
	int cols = magnitudeThresh.cols;
	int radiuslow = 10, radiushigh = 100;
	
	for (int x = 0; x < rows; x++)
		for (int y = 0; y < cols; y++) {
			if ((int)magnitudeThresh.at<uchar>(x, y) != 0)
				for (int r = radiuslow; r < radiushigh; r++) {

					int x0 = round(x + r * sin(direction.at<float>(x, y)));
					int y0 = round(y + r * cos(direction.at<float>(x, y)));
					if (x0 >= 0 && y0 >= 0 && x0 < rows && y0 < cols)
						houghCircles[x0][y0][r - radiuslow] += 1;

					x0 = round(x - r * sin(direction.at<float>(x, y)));
					y0 = round(y - r * cos(direction.at<float>(x, y)));
					if (x0 >= 0 && y0 >= 0 && x0 < rows && y0 < cols)
						houghCircles[x0][y0][r - radiuslow] += 1;
				}
		}
}

void houghLineDetect(Mat &magnitudeThresh, Mat &direction, int **houghLines) {
	int rows = magnitudeThresh.rows;
	int cols = magnitudeThresh.cols;
	int diagonal = ceil(sqrt(pow(rows, 2) * pow(cols, 2)));

	for (int x = 0; x < rows; x++)
		for (int y = 0; y < cols; y++)
			if ((int)magnitudeThresh.at<uchar>(x, y) != 0) {
				for (int theta = 0; theta < 360; theta++) {
					int ro = round(x * cos(theta*(M_PI/180)) + y * sin(theta*(M_PI/180)));
					if (ro >= 0 && ro < diagonal) {
						houghLines[ro][theta] += 1;
					}
				}
			}
}

void sobel(Mat &input, Mat &magnitudeThreshold, Mat &directionOut, int rank) {
	Mat inputGray;
	cvtColor(input, inputGray, CV_BGR2GRAY);
	//Gaussian Blur Image
	Mat blurredIm;
	GaussianBlur2(inputGray, 17, blurredIm);

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
	threshold(magnitude, magnitude, 120, 255, CV_THRESH_BINARY);
	//namedWindow("Magnitude"+ std::to_string(rank), CV_WINDOW_AUTOSIZE);
	//imshow("Magnitude"+std::to_string(rank), magnitude);
	magnitudeThreshold.create(magnitude.size(), magnitude.type());
	magnitudeThreshold = magnitude.clone();



	Mat direction, dirShow;
	directionAtan(derX, derY, direction);
	directionOut.create(direction.size(), direction.type());
	directionOut = direction.clone();
	normaliseImage(direction, dirShow);
	//namedWindow("Direction"+std::to_string(rank), CV_WINDOW_AUTOSIZE);
	//imshow("Direction"+std::to_string(rank), dirShow);

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
	output = output - M_PI/2;
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
					float imageval = (float)paddedInput.at<uchar>(imagex, imagey);
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