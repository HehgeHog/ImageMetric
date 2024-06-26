#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#ifndef metricks
#define metricks
	cv::Mat Image_Inversion_CUDA(cv::Mat& img, int step);
	cv::Mat ImageSharpening_CUDA(cv::Mat& img, int step);
	cv::Mat BrightnessChange_CUDA(cv::Mat& img, int step);
	cv::Mat Saturation_CUDA(cv::Mat& img, float step);

	cv::Mat SimpleDeNoise_CUDA(cv::Mat& img);

	float HELM_CUDA(cv::Mat& img);
	float ACMO_CUDA(cv::Mat& img);
#endif