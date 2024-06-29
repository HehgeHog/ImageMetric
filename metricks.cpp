#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "metricks.h"

using namespace std;
using namespace cv;

int main()
{
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
		cv::Mat img;
		if (!cap.read(img))
		{
			std::cout << "No Frame available" << std::endl;
			return 0;
		}

		cv::imshow("original", img);

		double t0 = (double)cv::getTickCount();

		cv::Mat res;

		fstep += istep / 10;
		if (step_black != 0) // если значение затемнения != 0 то теперь значение яркости это затемнение
		{
			step_light = -step_black;
		}

		//float coffHELM_first = HELM_CUDA(img);

		//res = SimpleDeNoise_CUDA(img);

		res = Image_Inversion_CUDA(img, step2);
		//res = ImageSharpening_CUDA(res, step1);
		
		//res = BrightnessChange_CUDA(res, step_light);
		//res = Saturation_CUDA(res, fstep);

		//float coffHELM_second = HELM_CUDA(res);
		//float coffACMO = ACMO_CUDA(img);
		float coffGLVM = GLVM_CUDA(img);

		//std::cout << "HELM (sharpness): " << (-1)*(coffHELM_second - coffHELM_first) << std::endl;
		//std::cout << "ACMO (sharpness): " << coffACMO << std::endl;
		std::cout << "GLVM (sharpness): " << coffGLVM << std::endl;

		cv::imshow("result", res);

		std::cout << "Time to calculate: " << ((double)cv::getTickCount() - t0) / cv::getTickFrequency() << " seconds" << std::endl << std::endl;

		fstep = 1.0;

		if (cv::waitKey(1) == 27) flag = 1;
	}

	return 0;
}