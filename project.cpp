#include"Functions.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>


int main()
{
	cv::Mat img = cv::imread("images/1-A.bmp", 0);
	int* hist;
	hist = Functions::Hist8(img);

	for (int i = 0; i < 256; i++)
	{
		std::cout << i << ": " << hist[i] << std::endl;
	}

	cv::imshow("original", img);
	if (cv::waitKey(0) == 27) return 0;
	return 0;
}
