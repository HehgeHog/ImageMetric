#include"Functions.h"
#include<cmath>
#include<vector>
#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>

void Functions::CalcMetrics(cv::Mat& img)
{
    std::vector<std::string> Name = {"ACMO","HISE","BREN"}; // {"ACMO","HISE","BREN", "CONT", "HELM", "GLVM", "GLVA"};
    
    double coffACMO = Functions::ACMO(img);
    double coffHISE = Functions::HISE(img);
    double coffBREN = Functions::BREN(img);
    //double coffCONT = Functions::CONT(img); // долгий расчет
    //double coffHELM = Functions::HELM(img); // долгий расчет
    //double coffGLVM = Functions::GLVM(img); // долгий расчет
    //double coffGLVA = Functions::GLVA(img); // долгий расчет

    std::vector<double> Coff = {coffACMO,coffHISE,coffBREN};

    std::cout << "--------------------" << std::endl;

    for (int i = 0; i < 3; i++)
    {
        show(Coff[i],Name[i]);
    }
}
void Functions::show(double coff, std::string name)
{
    std::cout << name << " : " << coff << std::endl;
}

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
    
    if (img.channels() != 1)
    {
        std::vector<cv::Mat> channel(img.channels());
        split(img, channel);

        for (int i = 0; i < img.channels(); i++)
        {
            temp = BrenCalc(channel[i]);
            sum += temp;
        }
    }
    else
    {
        sum = BrenCalc(img);
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

    if (img.channels() != 1)
    {
        std::vector<cv::Mat> channel(img.channels());
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
    else
    {
        cont = ContrastCalc(img);

        for (unsigned i = 0; i < img.cols * img.rows; i++)
        {
            sum += cont.at<uchar>(i);
        }
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

    if (img.channels() != 1)
    {
        std::vector<cv::Mat> channel(img.channels());
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
    else
    {
        R = HelmCalc(img);

        for (int i = 0; i < img.cols * img.rows; i++)
        {
            coff += R.at<uchar>(i);
        }
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

    if (img.channels() != 1)
    {
        std::vector<cv::Mat> channel(img.channels());
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
    else
    {
        average = AverageX15(img);
        for (unsigned i = 0; i < img.cols * img.rows; i++)
        {
            coff += std::pow((img.at<uchar>(i) - average.at<uchar>(i)), 2);
        }
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

    if (img.channels() != 1)
    {
        std::vector<cv::Mat> channel(img.channels());
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
    else
    {
        average = AverageImage(img);
        for (unsigned i = 0; i < img.cols * img.rows; i++)
        {
            coff += std::pow((img.at<uchar>(i) - average), 2);
        }
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
cv::Mat Functions::SimpleDeNoise(cv::Mat& img, int window)
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

    if (img.channels() != 1)
    {
        std::vector<cv::Mat> channel(img.channels());

        split(img, channel);

        for (int i = 0; i < img.channels(); i++)
        {
            channel[i] = SimpleSmoothing(channel[i], window);
        }

        cv::merge(channel, res);
    }
    else
    {
        res = SimpleSmoothing(img, window);
    }

    return res;
}

cv::Mat Functions::ImageSharpening(cv::Mat& img, int step)
{
    cv::Mat res;

    if (step < 0)
    {
        cv::blur(img, res, cv::Size(-step * 2 + 1, -step * 2 + 1));
    }
    else
    {
        cv::Mat dst;
        img.copyTo(dst);

        float matr[9]{
            -0.0375 - 0.05 * step, -0.0375 - 0.05 * step, -0.0375 - 0.05 * step,
            -0.0375 - 0.05 * step, 1.3 + 0.4 * step, -0.0375 - 0.05 * step, 
            -0.0375 - 0.05 * step, -0.0375 - 0.05 * step, -0.0375 - 0.05 * step
        };
        cv::Mat kernel_matrix = cv::Mat(3, 3, CV_32FC1, &matr);
        cv::filter2D(img, dst, 32, kernel_matrix);      
        res = dst;
    }

    return res;
}
cv::Mat Functions::ContrastEnhancement(cv::Mat& img, int step)
{
    //Контраст определяется в разности яркостей. 
    //Для увеличения контраста нам нужно раздвинуть диапазон яркостей от центра к краям.
    cv::Mat res;

    if (img.depth() != 0 && img.depth() != 1) // if != 8 bit
    {
        std::cout << "ContrastEnhancement: Image depth error, not 8 bit" << std::endl;
        return img;
    }

    std::vector<cv::Mat> channel;
    cv::split(img, channel);
    cv::Mat lut(1, 256, CV_8UC1);
    double contrastLevel = double(100 + step) / 100;

    uchar* p = lut.data;
    double d = 0;

    for (int i = 0; i < 256; i++)
    {
        d = ((double(i) / 255 - 0.5) * contrastLevel + 0.5) * 255;

        if (d > 255)
        {
            d = 255;
        }
            
        if (d < 0)
        {
            d = 0;
        }
            
        p[i] = d;
    }

    LUT(channel[0], lut, channel[0]);
    LUT(channel[1], lut, channel[1]);
    LUT(channel[2], lut, channel[2]);

    cv::merge(channel, res);

    return res;
}
cv::Mat Functions::Saturation(cv::Mat& img, int step)
{
    //Для изменения насыщенности изображение преобразуется в систему цветности HSV и разбивается на слои.
    //К значениям слоя «Sature» прибавляется шаг. 
    cv::Mat res;
    std::vector<cv::Mat> hsv;
    cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_RGB2HSV_FULL);
    cv::split(img, hsv);
    hsv[1] += step * 2;
    cv::merge(hsv, res);
    cv::cvtColor(res, res, cv::ColorConversionCodes::COLOR_HSV2RGB_FULL);
    return res;
}
cv::Mat Functions::BrightnessChange(cv::Mat& img, int step)
{
    cv::Mat res;
    img.copyTo(res);
    cv::Mat kernel_matrix;

    if (step < 0)
    {
        float matr[9]{
             -0.05 - 0.15 * step, 0.02 + 0.05 * step, -0.05 - 0.15 * step,
              0.02 + 0.05 * step, 0.8 + 0.5 * step, 0.02 + 0.05 * step,
             -0.05 - 0.15 * step, 0.02 + 0.05 * step, -0.05 - 0.15 * step
        };
        kernel_matrix = cv::Mat(3, 3, CV_32FC1, &matr);
    }
    else
    {
        float matr[9]{
            -0.05 - 0.15 * step, 0.015 + 0.05 * step, -0.05 - 0.15 * step,
             0.015 + 0.05 * step, 1.3 + 0.5 * step, 0.015 + 0.05 * step,
            -0.05 - 0.15 * step, 0.015 + 0.05 * step, -0.05 - 0.15 * step
        };
        kernel_matrix = cv::Mat(3, 3, CV_32FC1, &matr);
    }
        
    cv::filter2D(img, res, 32, kernel_matrix);

    return res;
}

unsigned char AddDoubleToByte(unsigned char bt, double d)
{

    unsigned char result = bt;
    if (double(result) + d > 255)
        result = 255;
    else if (double(result) + d < 0)
        result = 0;
    else
    {
        result += d;
    }
    return result;
}

cv::Mat GetGammaExpo(int step)
{
    cv::Mat result(1, 256, CV_8UC1);

    uchar* p = result.data;
    for (int i = 0; i < 256; i++)
    {
        p[i] = AddDoubleToByte(i, std::sin(i * 0.01255) * step * 10);
    }

    return result;
}

cv::Mat Functions::Expo(cv::Mat& img, int step)
{
    cv::Mat res;

    std::vector<cv::Mat> hsv;
    cv::cvtColor(img, res, cv::ColorConversionCodes::COLOR_RGB2HSV_FULL);
    cv::Mat lut = GetGammaExpo(step);
    cv::split(res, hsv);
    cv::LUT(hsv[2], lut, hsv[2]);
    cv::merge(hsv, res);
    cv::cvtColor(res, res, cv::ColorConversionCodes::COLOR_HSV2RGB_FULL);
    
    return res;
}

