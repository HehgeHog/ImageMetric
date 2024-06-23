#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Inversion_CUDA.h"
#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>

__global__ void Inversion_CUDA(unsigned char* Image, int Channels);
__global__ void Sharpening_CUDA(uchar* Image, float* mask, uchar* output, int ImageC, int ImageR);

void Image_Inversion_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels) 
{
	unsigned char* Dev_Input_Image = NULL;

	cudaMalloc((void**)&Dev_Input_Image, Height * Width * Channels);
	
	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * Channels, cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Inversion_CUDA <<<Grid_Image, 1 >>> (Dev_Input_Image, Channels);

	cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * Channels, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(Dev_Input_Image);
}

cv::Mat ImageSharpening(cv::Mat& img, int step)
{
	uchar* input = NULL;
	uchar* output = NULL;
	float matr[9] =
	{
		-0.0375 - 0.05 * step, -0.0375 - 0.05 * step, -0.0375 - 0.05 * step,
		-0.0375 - 0.05 * step, 1.3 + 0.4 * step, -0.0375 - 0.05 * step,
		-0.0375 - 0.05 * step, -0.0375 - 0.05 * step, -0.0375 - 0.05 * step,
	};

	cv::Mat res = cv::Mat(img.rows,img.cols,CV_8UC3);
	uchar* ResData = new uchar[img.rows * img.cols * img.channels()];

	cudaMalloc((void**)&input, img.cols * img.rows * img.channels());
	cudaMalloc((void**)&matr, sizeof(matr));
	cudaMalloc((void**)&output, img.cols * img.rows * img.channels());

	cudaMemcpy(input, img.data, img.cols * img.rows * img.channels(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(matr, img.data, sizeof(matr), cudaMemcpyKind::cudaMemcpyHostToDevice);

	
	//dim3 blockSize(16, 16);
	//dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
	dim3 gridSize(img.cols, img.rows);
	Sharpening_CUDA << <gridSize, 1 >> > (input, matr, output, img.cols, img.rows);

	cudaMemcpy(ResData, output, img.cols * img.rows * img.channels(), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	res.data = ResData;

	cudaFree(output);
	cudaFree(input);
	cudaFree(matr);

	return res;
}

__global__ void Inversion_CUDA(unsigned char* Image, int Channels) 
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) 
	{
		Image[idx + i] = 255 - Image[idx + i];
	}
}
__global__ void Sharpening_CUDA(uchar* img, float* mask, uchar* output, int ImageC, int ImageR)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * 3;

	//for (int i = 0; i < 3; i++)
	//{
	//	output[idx + i] = 255 - img[idx + i];
	//}
	for (int i = 0; i < ImageR; i++)
	{
		for (int j = 0; j < ImageC; j++)
		{
			for (int channel = 0; channel < 3; channel++)
			{
				output[i + j + channel] = 255 - img[i + j + channel];;
			}
		}
	}

	/*int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < ImageR; i++)
	{
		for (int j = 0; j <ImageC; j++)
		{
			for (int channel = 0; channel < 3; channel++)
			{
				output[i * ImageC + j + channel] = 255;
			}
		}
	}*/
}