#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#ifndef _Inversion_CUDA_
#define _Inversion_CUDA_
	cv::Mat Image_Inversion_CUDA(cv::Mat& img, int step);
	cv::Mat ImageSharpening_CUDA(cv::Mat& img, int step);
	cv::Mat BrightnessChange_CUDA(cv::Mat& img, int step);
	cv::Mat Saturation_CUDA(cv::Mat& img, float step);

	cv::Mat SimpleDeNoise_CUDA(cv::Mat& img);

	cv::Mat HELM_CUDA(cv::Mat& img);
#endif