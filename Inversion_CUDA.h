#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#ifndef _Inversion_CUDA_
#define _Inversion_CUDA_
	void Image_Inversion_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels);
	cv::Mat ImageSharpening(cv::Mat& img, int step);
#endif