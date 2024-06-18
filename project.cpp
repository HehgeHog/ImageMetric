#include"Functions.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>

int main()
{
	cv::VideoCapture cap("video/cam_1_14.mp4");
	if (!cap.isOpened())
	{
		std::cout << "Error opening video stream" << std::endl;
		return -1;
	}

	bool flag = 0;
	int step = 0;

	cv::namedWindow("Original", cv::WINDOW_NORMAL);
	cv::namedWindow("Modified", cv::WINDOW_NORMAL);
	cv::namedWindow("Trackbar", 0);

	cv::createTrackbar("name:", "Trackbar", &step, 15);

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

		std::cout << "step = " << step << std::endl;

		//res = Functions::ImageSharpening(img, step); // повышение резкости
		res = Functions::ContrastEnhancement(res, 0); // повышение контраста
		//res = Functions::Saturation(res, 0); // повышение насыщенности
		//res = Functions::BrightnessChange(img, 0); // изменение яркости
		
		Functions::CalcMetrics(res);

		cv::imshow("Modified", img);

		std::cout << "Time to calculate: " << ((double)cv::getTickCount() - t0) / cv::getTickFrequency() << " seconds" << std::endl << std::endl;
		
		if (cv::waitKey(1) == 27) flag = 1;
	}

	return 0;
}