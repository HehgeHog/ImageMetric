#pragma once
#include<string>
#include<vector>
#include<iostream>
#include<opencv2/opencv.hpp>

class Functions
{
public:
	static double ACMO(cv::Mat& img); //Absolute Central Moment Operator
	static double HISE(cv::Mat& img); //histogram entropy
	static double BREN(cv::Mat& img); //Brenner's focus measure 
	static double CONT(cv::Mat& img); //image contrast
	static double HELM(cv::Mat& img); //Helmli's measure
	static double GLVM(cv::Mat& img); //gray-level variance modified
	static double GLVA(cv::Mat& img); //gray-level variance

	static cv::Mat deNoise(cv::Mat& img, int window);
};