﻿#include"Functions.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>


int main()
{
	cv::Mat img = cv::imread("images/1-A.bmp");

	cv::imshow("original", img);

	if (cv::waitKey(0) == 27) return 0;
	return 0;
}