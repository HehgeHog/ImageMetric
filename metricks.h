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
	float GLVM_CUDA(cv::Mat& img);

	float ACMO(cv::Mat& img);

	void CalcMetrics(std::vector<int> src, cv::Mat& img, float& cACMO_CUDA, float& cHELM_CUDA, float& cGLVM_CUDA, float& cACMO);
	void SelectingFunctions(std::vector<int>& dst);
	void Changes(float& cACMO_CUDA1, float& cHELM_CUDA1, float& cGLVM_CUDA1, float& cACMO1,
		float& cACMO_CUDA2, float& cHELM_CUDA2, float& cGLVM_CUDA2, float& cACMO2);
#endif