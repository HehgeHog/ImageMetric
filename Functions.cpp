#include"Functions.h"
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

int* Functions::Hist8(cv::Mat img)
{
    //определение размеров и создание матрицы гистограммы
    int cols = img.cols;
    int rows = img.rows;
    int temporary = 0;

    static int hist[256]{};

    memset(hist,0,256); // заполнение массива нулями

    //рассчет значений гистограммы
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            temporary = img.at<uchar>(i,j);
            hist[temporary] += 1;
        }
    }

    return hist;
}

