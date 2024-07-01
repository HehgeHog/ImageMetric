#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "metricks.h"

int main()
{
	std::vector<int> functions;
	std::vector<float> odds_first(5); // в скобках количество метрик
	std::vector<float> odds_second(5);

	SelectingFunctions(functions);

	cv::VideoCapture cap("video/cam_1_14.mp4");
	if (!cap.isOpened())
	{
		std::cout << "Error opening video stream" << std::endl;
		return -1;
	}

	cv::namedWindow("original", cv::WINDOW_NORMAL);
	cv::namedWindow("result", cv::WINDOW_NORMAL);
	cv::namedWindow("trackbar", cv::WINDOW_NORMAL);

	bool flag = 0;
	float saturation_fstep = 1.0;
	int saturation_step = 0;
	int sharpening_step = 0;
	int inversion_step = 0;
	int denoise_step = 0;
	int step_light = 0, step_black = 0;
	
	cv::createTrackbar("DeNoise:", "trackbar", &denoise_step, 1);
	cv::createTrackbar("Inversion:", "trackbar", &inversion_step, 1);
	cv::createTrackbar("Sharpening:", "trackbar", &sharpening_step, 1);
	cv::createTrackbar("Saturation:", "trackbar", &saturation_step, 40);
	cv::createTrackbar("PosBright:", "trackbar", &step_light, 20);
	cv::createTrackbar("NegBright:", "trackbar", &step_black, 20);

	while (flag == 0)
	{
		cv::Mat img, res;
		if (!cap.read(img))
		{
			std::cout << "No Frame available" << std::endl;
			return 0;
		}

		cv::imshow("original", img);

		double t0 = (double)cv::getTickCount();

		saturation_fstep += saturation_step / 10;
		if (step_black != 0) // если значение затемнени€ != 0 то теперь значение €ркости это затемнение
		{
			step_light = -step_black;
		}

		//функции работы с изображени€ми

		CalcMetrics(functions, img, odds_first);

		res = SimpleDeNoise_CUDA(img, denoise_step);

		res = Image_Inversion_CUDA(res, inversion_step);
		res = ImageSharpening_CUDA(res, sharpening_step);
		res = BrightnessChange_CUDA(res, step_light);
		res = Saturation_CUDA(res, saturation_fstep);

		cv::imshow("result", res);
		
		CalcMetrics(functions, res, odds_second);

		Changes(odds_first, odds_second);

		//----------------------------------------

		std::cout << "Time to calculate: " << ((double)cv::getTickCount() - t0) / cv::getTickFrequency() << " seconds" << std::endl << std::endl;

		saturation_fstep = 1.0;

		if (cv::waitKey(1) == 27) flag = 1;
	}

	return 0;
}