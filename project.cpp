#include"Functions.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>


int main()
{
	cv::Mat img = cv::imread("images/1-A.bmp", 0);

	if(img.empty())
	{
		std::cout << "Image reading error" << std::endl;
		return -1;
	}

	cv::imshow("original", img);
	
	double coffACMO = Functions::ACMO(img);
	
	std::cout << "ACMO: " << coffACMO << std::endl;

	if (cv::waitKey(0) == 27) return 0;
	return 0;
}