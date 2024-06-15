#include"Functions.h"
#include<cmath>
#include<vector>
#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>

static int SumHist(std::vector<int>& hist)
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

    return double(sum)/double(pow(hist.size(),2));
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
    int sum = SumHist(hist);

    double p = 0;
    double coff = 0;

    for (int i = 0; i < size; i++)
    {
        p = Probability(hist,sum,i);
        coff += std::abs(i - average) * p;
    }

    return coff;
}

double Functions::HISE(cv::Mat& img)
{
    int size = 0;

    //определение числа уровней яркости изображения
    switch (img.depth())
    {
    case 0:
    case 1:
        size = pow(2, 8); // 8-bit
        break;
    case 2:
    case 3:
        size = pow(2, 16);
        break;
    case 4:
    case 5:
        size = pow(2, 32);
        break;
    case 6:
        size = pow(2, 64);
        break;
    default:
        std::cout << "HISE: Image depth error" << std::endl;
        break;
    }

    std::vector<int> hist = Hist(img, size);
    int sum = SumHist(hist);

    double p = 0;
    double coff = 0;

    for (int i = 0; i < size; i++)
    {
        p = Probability(hist, sum, i);
        if (p != 0)
        {
            coff += p * log(p);
        }
    }

    return coff*(-1);
};

int BrenCalc(cv::Mat& img)
{
    int sum = 0;

    for (int i = 0; i < img.rows - 2; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            sum += std::abs(std::pow(img.at<uchar>(i, j) - img.at<uchar>(i + 2, j), 2));
        }
    }

    return sum;
}
double Functions::BREN(cv::Mat& img)
{
    int sum = 0;
    int temp = 0;

    if (img.depth() != 0 && img.depth() != 1) // if != 8 bit
    {
        std::cout << "BREN: Image depth error, not 8 bit" << std::endl;
        return -1;
    }
    
    if (img.channels() == 3)
    {
        cv::Mat channel[3];
        split(img, channel);

        for (int i = 0; i < img.channels(); i++)
        {
            temp = BrenCalc(channel[i]);
            sum += temp;
        }
    }
    else if(img.channels() == 1)
    {
        sum = BrenCalc(img);
    }
    else
    {
        std::cout << "BREN: image must be 1 or 3 channel" << std::endl;
        return -1;
    }

    return (double)sum / (double)((img.cols * img.rows) * img.channels());
}

cv::Mat ContrastCalc(cv::Mat& img)
{
    cv::Mat dst;
    img.copyTo(dst);

    for (int i = 1; i < dst.rows-1; i++)
    {
        for (int j = 1; j < dst.cols-1; j++)
        {
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    if (x != 0 && y != 0)
                    {
                        dst.at<uchar>(i, j) = std::abs(dst.at<uchar>(i, j) - dst.at<uchar>(i - x, j - y));
                    }
                }
            }
        }
    }

    return dst;
}
double Functions::CONT(cv::Mat& img)
{
    cv::Mat cont;
    int sum = 0;

    if (img.depth() != 0 && img.depth() != 1) // if != 8 bit
    {
        std::cout << "CONT: Image depth error, not 8 bit" << std::endl;
        return -1;
    }

    if (img.channels() == 3)
    {
        cv::Mat channel[3];
        split(img, channel);

        for (int i = 0; i < img.channels(); i++)
        {
            cont = ContrastCalc(channel[i]);

            for (unsigned j = 0; j < img.cols * img.rows; j++)
            {
                sum += cont.at<uchar>(j);
            }
        }
    }
    else if(img.channels() == 1)
    {
        cont = ContrastCalc(img);

        for (unsigned i = 0; i < img.cols * img.rows; i++)
        {
            sum += cont.at<uchar>(i);
        }
    }
    else
    {
        std::cout << "CONT: image must be 1 or 3 channel" << std::endl;
        return -1;
    }

    return (double)sum/ (double)((img.cols * img.rows) * img.channels());
}

cv::Mat AverageX15(cv::Mat& img)
{
    cv::Mat average;
    img.copyTo(average);
    int sum = 0;
    int counter = 0;

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            for (int x = -7; x <= 7; x++)
            {
                for (int y = -7; y <= 7; y++)
                {
                    if (x != 0 && y != 0 && i-x >= 0 && i-x < img.rows && j-y >= 0 && j-y < img.cols)
                    {
                        sum += average.at<uchar>(i-x, j-y);
                        counter += 1;
                    }
                }
            }
            average.at<uchar>(i, j) = std::round((double)sum / (double)counter);
            counter = 0;
            sum = 0;
        }
    }

    return average;
}
cv::Mat HelmCalc(cv::Mat& img)
{
    cv::Mat R;
    img.copyTo(R);

    cv::Mat average = AverageX15(img);

    for (int i = 0; i < img.cols * img.rows; i++)
    {
        if (average.at<uchar>(i) >= img.at<uchar>(i))
        {
            R.at<uchar>(i) = average.at<uchar>(i);
        }
        else
        {
            R.at<uchar>(i) = img.at<uchar>(i);
        }
    }

    return R;
}
double Functions::HELM(cv::Mat& img)
{
    double coff = 0;
    cv::Mat R;

    if (img.depth() != 0 && img.depth() != 1) // if != 8 bit
    {
        std::cout << "HELM: Image depth error, not 8 bit" << std::endl;
        return -1;
    }

    if (img.channels() == 3)
    {
        cv::Mat channel[3];
        split(img, channel);

        for (int i = 0; i < img.channels(); i++)
        {
            R = HelmCalc(channel[i]);

            for (int j = 0; j < img.cols * img.rows; j++)
            {
                coff += R.at<uchar>(j);
            }
        }

    }
    else if(img.channels() == 1)
    {
        R = HelmCalc(img);

        for (int i = 0; i < img.cols * img.rows; i++)
        {
            coff += R.at<uchar>(i);
        }
    }
    else
    {
        std::cout << "HELM: image must be 1 or 3 channel" << std::endl;
        return -1;
    }

    return coff/(double)((img.cols * img.rows) * img.channels());
}

double Functions::GLVM(cv::Mat& img)
{
    double coff = 0;
    cv::Mat average;

    if (img.depth() != 0 && img.depth() != 1) // if != 8 bit
    {
        std::cout << "GLVM: Image depth error, not 8 bit" << std::endl;
        return -1;
    }

    if (img.channels() == 3)
    {
        cv::Mat channel[3];
        split(img, channel);

        for (int i = 0; i < img.channels(); i++)
        {
            average = AverageX15(channel[i]);

            for (unsigned j = 0; j < img.cols * img.rows; j++)
            {
                coff += std::pow((channel[i].at<uchar>(j) - average.at<uchar>(j)), 2);
            }
        }
    }
    else if(img.channels() == 1)
    {
        average = AverageX15(img);
        for (unsigned i = 0; i < img.cols * img.rows; i++)
        {
            coff += std::pow((img.at<uchar>(i) - average.at<uchar>(i)), 2);
        }
    }
    else
    {
        std::cout << "GLVM: image must be 1 or 3 channel" << std::endl;
        return -1;
    }

    return coff/(double)((img.cols * img.rows) * img.channels());
}

double AverageImage(cv::Mat& img)
{
    double sum = 0;

    for (unsigned i = 0; i < img.rows * img.cols; i++)
    {
        sum += img.at<uchar>(i);
    }

    return sum /(double)(img.cols * img.rows);
}
double Functions::GLVA(cv::Mat& img)
{
    double coff = 0;
    double average = 0;

    if (img.depth() != 0 && img.depth() != 1) // if != 8 bit
    {
        std::cout << "GLVA: Image depth error, not 8 bit" << std::endl;
        return -1;
    }

    if (img.channels() == 3)
    {
        cv::Mat channel[3];
        split(img, channel);

        for (int i = 0; i < img.channels(); i++)
        {
            average = AverageImage(channel[i]);

            for (unsigned j = 0; j < img.cols * img.rows; j++)
            {
                coff += std::pow((channel[i].at<uchar>(j) - average), 2);
            }
        }
    }
    else if (img.channels() == 1)
    {
        average = AverageImage(img);
        for (unsigned i = 0; i < img.cols * img.rows; i++)
        {
            coff += std::pow((img.at<uchar>(i) - average), 2);
        }
    }
    else
    {
        std::cout << "GLVA: image must be 1 or 3 channel" << std::endl;
        return -1;
    }

    return coff / (double)((img.cols * img.rows) * img.channels());
}

cv::Mat SimpleSmoothing(cv::Mat& img, int window)
{
    cv::Mat res;
    img.copyTo(res);
    int board = std::round((window / 2) - 0.2);
    int sum = 0;
    int counter = 0;

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            for (int x = -board; x <= board; x++)
            {
                for (int y = -board; y <= board; y++)
                {
                    if (x != 0 && y != 0 && i - x >= 0 && i - x < img.rows && j - y >= 0 && j - y < img.cols)
                    {
                        sum += res.at<uchar>(i - x, j - y);
                        counter += 1;
                    }
                }
            }

            res.at<uchar>(i, j) = std::round((double)sum / (double)counter);
            counter = 0;
            sum = 0;
        }
    }

    return res;
}
cv::Mat Functions::deNoise(cv::Mat& img, int window)
{
    cv::Mat res;

    if (img.depth() != 0 && img.depth() != 1) // if != 8 bit
    {
        std::cout << "deNoise: Image depth error, not 8 bit" << std::endl;
        return img;
    }
    if (window < 3 && window % 2 != 0)
    {
        std::cout << "deNoise: window must be odd and greater than 2" << std::endl;
        return img;
    }

    if (img.channels() == 3)
    {
        cv::Mat channel[3];
        cv::Mat temp(cv::Size(img.rows, img.cols), CV_8UC3, 0.0);
        split(img, channel);

        for (int i = 0; i < img.channels(); i++)
        {
            channel[i] = SimpleSmoothing(channel[i], window);
        }

        cv::merge(channel,3, temp);
        res = temp;
    }
    else if(img.channels() == 1)
    {
        res = SimpleSmoothing(img, window);
    }
    else
    {
        std::cout << "deNoise: image must be 1 or 3 channel" << std::endl;
        return img;
    }

    return res;
}