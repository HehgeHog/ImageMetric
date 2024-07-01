#include"Functions.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>

int main()
{
	std::vector<int> functions;
	std::vector<float> odds_first(7); // в скобках количество метрик
	std::vector<float> odds_second(7);

	Functions::SelectingFunctions(functions);
	
	cv::VideoCapture cap("video/cam_1_14.mp4");
	if (!cap.isOpened())
	{
		std::cout << "Error opening video stream" << std::endl;
		return -1;
	}

	bool flag = 0;
	int denoise_step = 0;
	int sharpening_step = 0;
	int contrast_step = 0;
	int saturation_step = 0;
	int step_light = 0, step_black = 0;
	int expo_step = 0;
	int hue_step = 0;
	int temperature_step = 0;

	cv::namedWindow("original", cv::WINDOW_NORMAL);
	cv::namedWindow("result", cv::WINDOW_NORMAL);
	cv::namedWindow("trackbar", cv::WINDOW_NORMAL);

	cv::createTrackbar("Denoise:", "trackbar", &denoise_step, 1);
	cv::createTrackbar("Sharpening:", "trackbar", &sharpening_step, 40);
	cv::createTrackbar("Contrast:", "trackbar", &contrast_step, 40);
	cv::createTrackbar("Saturation:", "trackbar", &saturation_step, 40);
	cv::createTrackbar("PosBright:", "trackbar", &step_light, 20);
	cv::createTrackbar("NegBright:", "trackbar", &step_black, 20);
	cv::createTrackbar("Expo:", "trackbar", &expo_step, 40);
	cv::createTrackbar("Hue:", "trackbar", &hue_step, 40);
	cv::createTrackbar("Temperature :", "trackbar", &temperature_step, 40);

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
		if (step_black != 0) // если значение затемнения != 0 то теперь значение яркости это затемнение
		{
			step_light = -step_black;
		}

		//функции работы с изображениями
		
		Functions::CalcMetrics(functions, img, odds_first);
		
		res = Functions::SimpleDeNoise(img, 3, denoise_step); // медианный фильтр изображения

		res = Functions::ImageSharpening(res, sharpening_step); // повышение резкости
		res = Functions::ContrastEnhancement(res, contrast_step); // повышение контраста
		res = Functions::Saturation(res, saturation_step); // повышение насыщенности
		res = Functions::BrightnessChange(res, step_light); // изменение яркости
		res = Functions::Expo(res, expo_step); // изменения экспозиции
		res = Functions::Hue(res, hue_step); // изменения оттенка
		res = Functions::Temperature(res, temperature_step); // изменения цветовой температуры

		cv::imshow("result", res);

		Functions::CalcMetrics(functions, res, odds_second);
		Functions::Changes(odds_first, odds_second);

		//----------------------------------------

		std::cout << "Time to calculate: " << ((double)cv::getTickCount() - t0) / cv::getTickFrequency() << " seconds" << std::endl << std::endl;
		
		if (cv::waitKey(1) == 27) flag = 1;
	}

	return 0;
}