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

	while (flag == 0)
	{
		cv::Mat img;
		if (!cap.read(img))
		{
			std::cout << "No Frame available" << std::endl;
			return 0;
		}

		cv::namedWindow("original", cv::WINDOW_NORMAL);
		cv::imshow("original", img);

		double t0 = (double)cv::getTickCount();
		double coffACMO = Functions::ACMO(img);
		double coffHISE = Functions::HISE(img);
		double coffBREN = Functions::BREN(img);
		
		//double coffCONT = Functions::CONT(img);
		//double coffHELM = Functions::HELM(img); // долгий расчет
		//double coffGLVM = Functions::GLVM(img); // долгий расчет
		//double coffGLVA = Functions::GLVA(img); // долгий расчет

		std::cout << "ACMO (sharpness and contrast): " << coffACMO << std::endl;
		std::cout << "HISE (sharpness): " << coffHISE << std::endl;
		std::cout << "BREN (sharpness): " << coffBREN << std::endl;
		//std::cout << "CONT (contrast): " << coffCONT << std::endl;
		//std::cout << "HELM (sharpness): " << coffHELM << std::endl;
		//std::cout << "GLVM (sharpness): " << coffGLVM << std::endl;
		//std::cout << "GLVA (sharpness): " << coffGLVA << std::endl;

		//cv::Mat res = Functions::SimpleDeNoise(img, 3);

		cv::Mat res;

		//res = Functions::ImageSharpening(img, 2); // повышение резкости
		res = Functions::ContrastEnhancement(img, 1); // повышение контраста
		//res = Functions::Saturation(res, 1); // повышение насыщенности
		//res = Functions::BrightnessChange(img, -5); // изменение яркости

		coffACMO = Functions::ACMO(res);
		coffHISE = Functions::HISE(res);
		coffBREN = Functions::BREN(res);

		std::cout << "ACMO2 (sharpness and contrast): " << coffACMO << std::endl;
		std::cout << "HISE2 (sharpness): " << coffHISE << std::endl;
		std::cout << "BREN2 (sharpness): " << coffBREN << std::endl << std::endl;

		cv::namedWindow("modified", cv::WINDOW_NORMAL);
		cv::imshow("modified", res);

		std::cout << "Time to calculate: " << ((double)cv::getTickCount() - t0) / cv::getTickFrequency() << " seconds" << std::endl << std::endl;
		if (cv::waitKey(1) == 27) flag = 1;
	}

	return 0;
}