#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "metricks.h"

int main()
{
	std::vector<int> functions;
	float coffACMO_CUDA_first = 0.0; float coffACMO_CUDA_second = 0.0;
	float coffHELM_CUDA_first = 0.0; float coffHELM_CUDA_second = 0.0;
	float coffGLVM_CUDA_first = 0.0; float coffGLVM_CUDA_second = 0.0;
	float coffACMO_first = 0.0; float coffACMO_second = 0.0;

	SelectingFunctions(functions);

	cv::VideoCapture cap("D:/CUDA projects/CudaRuntime1/CudaRuntime1/images/cam_1_14.mp4");
	if (!cap.isOpened())
	{
		std::cout << "Error opening video stream" << std::endl;
		return -1;
	}

	cv::namedWindow("original", cv::WINDOW_NORMAL);
	cv::namedWindow("result", cv::WINDOW_NORMAL);
	cv::namedWindow("trackbar", cv::WINDOW_NORMAL);

	bool flag = 0;

	float fstep = 1.0;
	int istep = 0;

	int step1 = 0;
	int step2 = 0;
	int step_light = 0, step_black = 0;

	cv::createTrackbar("Sharpening:", "trackbar", &step1, 1);
	cv::createTrackbar("Inversion:", "trackbar", &step2, 1);
	cv::createTrackbar("Saturation:", "trackbar", &istep, 40);
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

		fstep += istep / 10;
		if (step_black != 0) // если значение затемнения != 0 то теперь значение яркости это затемнение
		{
			step_light = -step_black;
		}

		//функции работы с изображениями

		CalcMetrics(functions, img, coffACMO_CUDA_first, coffHELM_CUDA_first, coffGLVM_CUDA_first, coffACMO_first);

		//res = SimpleDeNoise_CUDA(img);

		res = Image_Inversion_CUDA(img, step2);
		res = ImageSharpening_CUDA(res, step1);
		
		res = BrightnessChange_CUDA(res, step_light);
		res = Saturation_CUDA(res, fstep);

		cv::imshow("result", res);
		
		CalcMetrics(functions, res, coffACMO_CUDA_second, coffHELM_CUDA_second, coffGLVM_CUDA_second, coffACMO_second);

		Changes(coffACMO_CUDA_first, coffHELM_CUDA_first, coffGLVM_CUDA_first, coffACMO_first,
			coffACMO_CUDA_second, coffHELM_CUDA_second, coffGLVM_CUDA_second, coffACMO_second);

		//----------------------------------------

		std::cout << "Time to calculate: " << ((double)cv::getTickCount() - t0) / cv::getTickFrequency() << " seconds" << std::endl << std::endl;

		fstep = 1.0;

		if (cv::waitKey(1) == 27) flag = 1;
	}

	return 0;
}