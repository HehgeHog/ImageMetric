#include"Functions.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>

int main()
{
	cv::VideoCapture cap("video/test1.mp4");
	if (!cap.isOpened())
	{
		std::cout << "Error opening video stream" << std::endl;
		return -1;
	}

	bool flag = 0;
	int step1 = 0;
	int step2 = 0;
	int step3 = 0;
	int step4 = -20;

	cv::namedWindow("Original", cv::WINDOW_NORMAL);
	cv::namedWindow("Modified", cv::WINDOW_NORMAL);
	cv::namedWindow("Trackbar", cv::WINDOW_NORMAL);

	cv::createTrackbar("Sharpening:", "Trackbar", &step1, 50);
	cv::createTrackbar("Contrast:", "Trackbar", &step2, 50);
	cv::createTrackbar("Saturation:", "Trackbar", &step3, 50);
	cv::createTrackbar("BrightnessChange:", "Trackbar", &step4, 20);

	while (flag == 0)
	{
		cv::Mat img;
		if (!cap.read(img))
		{
			std::cout << "No Frame available" << std::endl;
			return 0;
		}

		cv::imshow("Original", img);

		double t0 = (double)cv::getTickCount();
		Functions::CalcMetrics(img);

		//cv::Mat res = Functions::SimpleDeNoise(img, 3);

		cv::Mat res;

		res = Functions::ImageSharpening(img, step1); // повышение резкости
		res = Functions::ContrastEnhancement(res, step2); // повышение контраста
		res = Functions::Saturation(res, step3); // повышение насыщенности
		res = Functions::BrightnessChange(res, step4); // изменение яркости
		
		Functions::CalcMetrics(res);

		cv::imshow("Modified", res);

		std::cout << "Time to calculate: " << ((double)cv::getTickCount() - t0) / cv::getTickFrequency() << " seconds" << std::endl << std::endl;
		
		if (cv::waitKey(1) == 27) flag = 1;
	}

	return 0;
}