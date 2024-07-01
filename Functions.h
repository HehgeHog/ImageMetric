#pragma once
#include<string>
#include<vector>
#include<iostream>
#include<opencv2/opencv.hpp>

class Functions
{
public:
	static void CalcMetrics_old(cv::Mat& img);
	static void show(double coff, std::string name);

	static void CalcMetrics(std::vector<int> list, cv::Mat& img, std::vector<float>& odds);
	static void SelectingFunctions(std::vector<int>& dst);
	static void Changes(std::vector<float> odds_first, std::vector<float> odds_second);

	static double ACMO(cv::Mat& img); //Absolute Central Moment Operator
	static double HISE(cv::Mat& img); //histogram entropy
	static double BREN(cv::Mat& img); //Brenner's focus measure 
	static double CONT(cv::Mat& img); //image contrast
	static double HELM(cv::Mat& img); //Helmli's measure
	static double GLVM(cv::Mat& img); //gray-level variance modified
	static double GLVA(cv::Mat& img); //gray-level variance

	static cv::Mat SimpleDeNoise(cv::Mat& img, int window, int step);

	static cv::Mat ImageSharpening(cv::Mat& img, int step);
	static cv::Mat ContrastEnhancement(cv::Mat& img, int step);
	static cv::Mat Saturation(cv::Mat& img, int step);
	static cv::Mat BrightnessChange(cv::Mat& img, int step);
	static cv::Mat Expo(cv::Mat& img, int step);
	static cv::Mat Hue(cv::Mat& img, int step);
	static cv::Mat Temperature(cv::Mat& img, int step);
};