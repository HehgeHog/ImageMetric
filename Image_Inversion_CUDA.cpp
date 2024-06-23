#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Inversion_CUDA.h"

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
	
	bool flag = 0;

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

		/*Image_Inversion_CUDA(img.data, img.cols, img.rows, img.channels());*/
		cv::Mat res = ImageSharpening(img, 3);

		cv::imshow("result", img);

		std::cout << "Time to calculate: " << ((double)cv::getTickCount() - t0) / cv::getTickFrequency() << " seconds" << std::endl << std::endl;

		if (cv::waitKey(1) == 27) flag = 1;
	}

	return 0;
}