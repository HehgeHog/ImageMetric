#include"Functions.h"
#include<cmath>
#include<vector>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>

static int Sum(std::vector<int>& hist)
{
    int sum = 0;

    //подсчет количества всех пикселей
    for (int i = 0; i < hist.size(); i++)
    {
        sum += hist.at(i);
    }

    return sum;
}
static double Probability(std::vector<int>& hist, int sum, int brightness)
{
    return (double)hist.at(brightness)/(double)sum;
}
static double AverageHist(std::vector<int>& hist)
{
    int sum = 0;
    //сумма всех значений пикселей (сумма яркостей)
    for (int i = 0; i < hist.size(); i++)
    {
        sum += hist.at(i) * i;
    }

    return sum / pow(hist.size(),2);
}
static std::vector<int> Hist(cv::Mat& img, int size_hist)
{
    int temporary = 0;

    std::vector <int> hist(size_hist);

    if (hist.empty())
    {
        std::cout << "Error creating histogram" << std::endl;
        std::vector <int> error = {0};
        return error;
    }

    //рассчет значений гистограммы
    unsigned size = img.cols * img.rows * img.channels();
    for (unsigned i = 0; i < size; i++)
    {
        temporary = img.data[i];
        hist.at(temporary) += 1;
    }

    //проверка заполненности вектора
    //for (int i = 0; i < hist.size(); i++)
    //{
    //    std::cout << i << ": " << hist.at(i) << std::endl;
    //}

    return hist;
}
double Functions::ACMO(cv::Mat& img)
{
    int size = 0;

    //определение числа уровней яркости изображения
    switch (img.depth())
    {
    case 0:
    case 1:
        size = pow(2,8); // 8-bit
        break;
    case 2:
    case 3:
        size = pow(2,16);
        break;
    case 4:
    case 5:
        size = pow(2,32);
        break;
    case 6:
        size = pow(2,64);
        break;
    default:
        std::cout << "ACMO: Image depth error" << std::endl;
        break;
    }

    std::vector<int> hist = Hist(img, size); 
    double average = AverageHist(hist);
    int sum = Sum(hist);

    double p = 0;
    double coff = 0;

    for (int i = 0; i < size; i++)
    {
        p = Probability(hist,sum,i);
        coff += std::abs(i - average) * p;
    }

    return coff;
}


