#pragma once
#include<string>
#include<vector>
#include<iostream>
#include<opencv2/opencv.hpp>

class Functions
{
public:
	// ��������� ������� �������� � ��������� �� ������ ����������� ������������ �������
	static double ACMO(cv::Mat& img); //Absolute Central Moment Operator
};