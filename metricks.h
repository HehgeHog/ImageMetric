#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#ifndef metricks
#define metricks
	cv::Mat Image_Inversion_CUDA(cv::Mat& img, int step);
	cv::Mat ImageSharpening_CUDA(cv::Mat& img, int step);
	cv::Mat BrightnessChange_CUDA(cv::Mat& img, int step);
	cv::Mat Saturation_CUDA(cv::Mat& img, float step);

	cv::Mat SimpleDeNoise_CUDA(cv::Mat& img, int step);

	float HELM_CUDA(cv::Mat& img); //Helmli's measure
	float ACMO_CUDA(cv::Mat& img); //Absolute Central Moment Operator
	float GLVM_CUDA(cv::Mat& img); //gray-level variance modified

	float ACMO(cv::Mat& img);
	float HISE(cv::Mat& img);

	void CalcMetrics(std::vector<int> src, cv::Mat& img, std::vector<float>& odds);
	void SelectingFunctions(std::vector<int>& dst);
	void Changes(std::vector<float> odds_first, std::vector<float> odds_second);
#endif