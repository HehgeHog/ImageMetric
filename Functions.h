#pragma once
#include<string>
#include<vector>
#include<iostream>
#include<opencv2/opencv.hpp>

class Functions
{
public:
	// оценочная функция резкости и контраста на основе абсолютного центрального момента
	static double ACMO(cv::Mat& img); //Absolute Central Moment Operator
};