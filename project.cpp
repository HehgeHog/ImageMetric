#include"Functions.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>


int main()
{
	cv::Mat img = cv::imread("images/frame.png", 1);

	if(img.empty())
	{
		std::cout << "Image reading error" << std::endl;
		return -1;
	}

	cv::imshow("original", img);
	
	double coffACMO = Functions::ACMO(img);
	double coffHISE = Functions::HISE(img);
	double coffBREN = Functions::BREN(img);
	double coffCONT = Functions::CONT(img);
	//double coffHELM = Functions::HELM(img); // долгий расчет
	//double coffGLVM = Functions::GLVM(img); // долгий расчет
	//double coffGLVA = Functions::GLVA(img); // долгий расчет

	std::cout << "ACMO (sharpness and contrast): " << coffACMO << std::endl;
	std::cout << "HISE (sharpness): " << coffHISE << std::endl;
	std::cout << "BREN (sharpness): " << coffBREN << std::endl;
	std::cout << "CONT (contrast): " << coffCONT << std::endl;
	//std::cout << "HELM (sharpness): " << coffHELM << std::endl;
	//std::cout << "GLVM (sharpness): " << coffGLVM << std::endl;
	//std::cout << "GLVA (sharpness): " << coffGLVA << std::endl;

	if (cv::waitKey(0) == 27) return 0;
	return 0;
}