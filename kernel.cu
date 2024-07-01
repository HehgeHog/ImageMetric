#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "metricks.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

__device__ void sort(unsigned char* filterVector);
__device__ void RGBtoHSV(unsigned char r, unsigned char g, unsigned char b, float* h, float* s, float* v);
__device__ void HSVtoRGB(float h, float s, float v, unsigned char* r, unsigned char* g, unsigned char* b);

__global__ void Inversion(uchar* img, uchar* output);
__global__ void ApplyingMask(uchar3* img, float* mask, uchar3* output, int column, int row);
__global__ void MedianFilter(unsigned char* img, unsigned char* output, int column, int row);
__global__ void sharpeningFilter(unsigned char* img, float* mask, unsigned char* output, unsigned int column, unsigned int row);
__global__ void SaturationKernel(unsigned char* img, float saturation, unsigned char* output, int column, int row);
__global__ void AverageX15(unsigned char* img, unsigned char* output, int column, int row);
__global__ void HELM_calc(unsigned char* img, unsigned char* average, unsigned char* output);
__global__ void SumHistogram(int* output, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red);
__global__ void Histogram(unsigned char* img, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red);

cv::Mat Image_Inversion_CUDA(cv::Mat& img, int step)
{
	if (step == 0)
	{
		return img;
	}

	cv::Mat res(img.size(), img.type());

	uchar* input = NULL;
	uchar* output = NULL;

	cudaMalloc((void**)&input, img.cols * img.rows * img.channels());
	cudaMalloc((void**)&output, img.cols * img.rows * img.channels());

	cudaMemcpy(input, img.data, img.cols * img.rows * img.channels(), cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 gridSize(img.cols * img.rows);
	Inversion << <gridSize, 1 >> > (input, output);

	cudaMemcpy(res.data, output, img.cols * img.rows * img.channels(), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(input);
	cudaFree(output);

	return res;
}
cv::Mat ImageSharpening_CUDA(cv::Mat& img, int step)
{
	if (step == 0)
	{
		return img;
	}

	unsigned char* input = NULL;
	unsigned char* output = NULL;
	float* dmatr;
	float matr[9] =
	{
		-1 * step, -1 * step, -1 * step,
		-1 * step, 9 * step, -1 * step,
		-1 * step, -1 * step, -1 * step
	};

	const int inputSize = img.rows * img.cols * img.channels();

	cv::Mat res(img.size(), img.type());

	cudaMalloc((void**)&input, inputSize);
	cudaMalloc((void**)&dmatr, 9 * sizeof(float));
	cudaMalloc((void**)&output, inputSize);

	cudaMemcpy(input, img.data, inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dmatr, matr, 9 * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
	sharpeningFilter << <gridSize, blockSize >> > (input, dmatr, output, img.cols, img.rows);

	cudaMemcpy(res.data, output, inputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(output);
	cudaFree(input);
	cudaFree(dmatr);

	return res;
}
cv::Mat BrightnessChange_CUDA(cv::Mat& img, int step)
{
	if (step == 0)
	{
		return img;
	}

	uchar3* input = NULL;
	uchar3* output = NULL;
	float* dmatr;

	float matr[9]
	{
		-0.05 - 0.16 * step, 0.015 + 0.05 * step, -0.05 - 0.16 * step,
		0.015 + 0.05 * step, 1.3 + 0.5 * step, 0.015 + 0.05 * step,
		-0.05 - 0.16 * step, 0.015 + 0.05 * step, -0.05 - 0.16 * step
	};

	if (step < 0)
	{
		step = step * (-1);
		float matr[9]
		{
			 -0.05 - 0.15 * step, 0.02 + 0.05 * step, -0.05 - 0.15 * step,
			  0.02 + 0.05 * step, 0.8 + 0.5 * step, 0.02 + 0.05 * step,
			 -0.05 - 0.15 * step, 0.02 + 0.05 * step, -0.05 - 0.15 * step
		};
	}

	cv::Mat res(img.size(), img.type());

	cudaMalloc((void**)&input, img.total() * sizeof(uchar3));
	cudaMalloc((void**)&dmatr, 9 * sizeof(float));
	cudaMalloc((void**)&output, img.total() * sizeof(uchar3));

	cudaMemcpy(input, img.data, img.total() * sizeof(uchar3), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dmatr, matr, 9 * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
	ApplyingMask << <gridSize, blockSize >> > (input, dmatr, output, img.cols, img.rows);

	cudaMemcpy(res.data, output, img.total() * sizeof(uchar3), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(output);
	cudaFree(input);
	cudaFree(dmatr);

	return res;

}
cv::Mat SimpleDeNoise_CUDA(cv::Mat& img, int step)
{
	if (step == 0)
	{
		return img;
	}

	unsigned char* input = NULL;
	unsigned char* output = NULL;

	cv::Mat res(img.size(), img.type());

	const int inputSize = img.rows * img.cols * img.channels();

	cudaMalloc((void**)&input, inputSize);
	cudaMalloc((void**)&output, inputSize);

	cudaMemcpy(input, img.data, inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
	MedianFilter << <gridSize, blockSize >> > (input, output, img.cols, img.rows);

	cudaMemcpy(res.data, output, inputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(output);
	cudaFree(input);

	return res;
}
cv::Mat Saturation_CUDA(cv::Mat& img, float step)
{
	if (step == 1)
	{
		return img;
	}

	unsigned char* input = NULL;
	unsigned char* output = NULL;

	cv::Mat res(img.size(), img.type());

	const int inputSize = img.rows * img.cols * img.channels();

	cudaMalloc((void**)&input, inputSize);
	cudaMalloc((void**)&output, inputSize);

	cudaMemcpy(input, img.data, inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
	SaturationKernel << <gridSize, blockSize >> > (input, step, output, img.cols, img.rows);

	cudaMemcpy(res.data, output, inputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(output);
	cudaFree(input);

	return res;
}

float ACMO_CUDA(cv::Mat& img) 
{
	// создание 3х гистограмм
	unsigned char* input = NULL;

	int Histogram_Blue[256] = { 0 };
	int Histogram_Green[256] = { 0 };
	int Histogram_Red[256] = { 0 };

	int* Dev_Histogram_Blue = NULL;
	int* Dev_Histogram_Green = NULL;
	int* Dev_Histogram_Red = NULL;

	int InputSize = img.cols * img.rows * img.channels();

	cudaMalloc((void**)&input, InputSize);
	cudaMalloc((void**)&Dev_Histogram_Blue, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Green, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Red, 256 * sizeof(int));

	cudaMemcpy(input, img.data, InputSize, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Blue, Histogram_Blue, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Green, Histogram_Green, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Red, Histogram_Red, 256 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 Grid_Image(img.cols, img.rows);
	Histogram << <Grid_Image, 1 >> > (input, Dev_Histogram_Blue, Dev_Histogram_Green, Dev_Histogram_Red);

	cudaMemcpy(Histogram_Blue, Dev_Histogram_Blue, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Histogram_Green, Dev_Histogram_Green, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Histogram_Red, Dev_Histogram_Red, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(Dev_Histogram_Blue);
	cudaFree(Dev_Histogram_Green);
	cudaFree(Dev_Histogram_Red);
	cudaFree(input);

	//--------------
	//сумма гистограмм
	int* Dev_Histogram_Blue1 = NULL;
	int* Dev_Histogram_Green1 = NULL;
	int* Dev_Histogram_Red1 = NULL;

	int* Dev_United = NULL;
	int United[256] = { 0 };

	cudaMalloc((void**)&Dev_Histogram_Blue1, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Green1, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_Histogram_Red1, 256 * sizeof(int));
	cudaMalloc((void**)&Dev_United, 256 * sizeof(int));

	cudaMemcpy(Dev_Histogram_Blue1, Histogram_Blue, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Green1, Histogram_Green, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram_Red1, Histogram_Red, 256 * sizeof(int), cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (256 + threadsPerBlock - 1) / threadsPerBlock;
	SumHistogram << <blocksPerGrid, threadsPerBlock >> > (Dev_United, Dev_Histogram_Blue1, Dev_Histogram_Green1, Dev_Histogram_Red1);

	cudaMemcpy(United, Dev_United, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(Dev_Histogram_Blue1);
	cudaFree(Dev_Histogram_Green1);
	cudaFree(Dev_Histogram_Red1);
	cudaFree(Dev_United);
	//--------------
	//расчет среднего
	int sum = 0;
	float average = 0.0;

	for (int i = 0; i < 256; i++)
	{
		sum += United[i] * i;
		
	}

	average = float(sum) / float(65536);
	//-------------
	//расчет суммы
	int quantity = 0;

	for (int i = 0; i < 256; i++)
	{
		quantity += United[i];
	}
	//------------
	//расчет коэффициента
	float p = 0;
	float coff = 0;

	for (int i = 0; i < 256; i++)
	{
		p = (float)United[i]/(float)quantity;
		coff += std::abs(i - average) * p;
	}

	return coff;
} 
float HELM_CUDA(cv::Mat& img)
{
	//calc average

	unsigned char* input = NULL;
	unsigned char* output = NULL;

	cv::Mat R(img.size(), img.type());

	const int inputSize = img.rows * img.cols * img.channels();

	cudaMalloc((void**)&input, inputSize);
	cudaMalloc((void**)&output, inputSize);

	cudaMemcpy(input, img.data, inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
	AverageX15 << <gridSize, blockSize >> > (input, output, img.cols, img.rows);

	cudaMemcpy(R.data, output, inputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(output);
	cudaFree(input);

	//-------------------------------
	//calc HELM

	unsigned char* average = NULL;
	unsigned char* input2 = NULL;
	unsigned char* output2 = NULL;

	cv::Mat res(img.size(), img.type());

	cudaMalloc((void**)&input2, inputSize);
	cudaMalloc((void**)&average, inputSize);
	cudaMalloc((void**)&output2, inputSize);

	cudaMemcpy(input2, img.data, inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(average, R.data, inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 gridSize1(img.cols, img.rows);
	HELM_calc << <gridSize1, 1 >> > (input2, average, output2);

	cudaMemcpy(res.data, output2, inputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(output2);
	cudaFree(input2);
	cudaFree(average);

	//расчет коэффициента
	float sum = 0;

	for (int i = 0; i < res.channels(); i++)
	{
		sum += cv::sum(res)[i];
	}

	float res_coff = sum / float(inputSize);

	return res_coff;
}
float GLVM_CUDA(cv::Mat& img)
{
	//calc average
	unsigned char* input = NULL;
	unsigned char* output = NULL;

	cv::Mat average(img.size(), img.type());

	const int inputSize = img.rows * img.cols * img.channels();

	cudaMalloc((void**)&input, inputSize);
	cudaMalloc((void**)&output, inputSize);

	cudaMemcpy(input, img.data, inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, (img.rows + blockSize.y - 1) / blockSize.y);
	AverageX15 << <gridSize, blockSize >> > (input, output, img.cols, img.rows);

	cudaMemcpy(average.data, output, inputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaFree(output);
	cudaFree(input);
	//-------------------------------
	//расчет коэффициента
	float sum = 0;

	// Попиксельное вычитание
	cv::Mat diff_image;
	cv::subtract(img, average, diff_image);

	// Возведение в степень 2
	cv::Mat squared_image;
	cv::pow(diff_image, 2, squared_image);

	for (int i = 0; i < squared_image.channels(); i++)
	{
		sum += cv::sum(squared_image)[i];
	}

	float res_coff = sum / float(inputSize);

	return res_coff;
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
	return (double)hist.at(brightness) / (double)sum;
}
static double AverageHist(std::vector<int>& hist)
{
	int sum = 0;
	//сумма всех значений пикселей (сумма яркостей)
	for (int i = 0; i < hist.size(); i++)
	{
		sum += hist.at(i) * i;
	}

	return double(sum) / double(pow(hist.size(), 2));
}
static std::vector<int> Hist(cv::Mat& img, int size_hist)
{
	int temporary = 0;

	std::vector <int> hist(size_hist);

	//рассчет значений гистограммы
	unsigned size = img.cols * img.rows * img.channels();
	for (unsigned i = 0; i < size; i++)
	{
		temporary = img.data[i];
		hist.at(temporary) += 1;
	}

	return hist;
}

float ACMO(cv::Mat& img)
{
	int size = 256;

	std::vector<int> hist = Hist(img, size);
	double average = AverageHist(hist);
	int sum = SumHist(hist);

	double p = 0;
	double coff = 0;

	for (int i = 0; i < size; i++)
	{
		p = Probability(hist, sum, i);
		coff += std::abs(i - average) * p;
	}

	return coff;
}
float HISE(cv::Mat& img)
{
	int size = 256;

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

	return coff * (-1);
};

__global__ void Inversion(uchar* img, uchar* output)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * 3;

	for (int i = 0; i < 3; i++)
	{
		output[idx + i] = 255 - img[idx + i];
	}
}
__global__ void ApplyingMask(uchar3* img, float* mask, uchar3* output, int column, int row)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < column && y < row)
	{
		float3 filterTotal = make_float3(0.0f, 0.0f, 0.0f);
		int halfFilterSize = 3 / 2;

		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				int imgX = x + j - halfFilterSize;
				int imgY = y + i - halfFilterSize;

				if (imgX >= 0 && imgX < column && imgY >= 0 && imgY < row)
				{
					uchar3 pixelValue = img[imgY * column + imgX];
					float filterValue = mask[i * 3 + j];
					filterTotal.x += pixelValue.x * filterValue;
					filterTotal.y += pixelValue.y * filterValue;
					filterTotal.z += pixelValue.z * filterValue;
				}
			}
		}

		output[y * column + x] = make_uchar3(static_cast<uchar>(filterTotal.x),
			static_cast<uchar>(filterTotal.y),
			static_cast<uchar>(filterTotal.z));
	}
}
__global__ void MedianFilter(unsigned char* img, unsigned char* output, int column, int row)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= 3 / 2) && (x < (column - 3 / 2)) && (y >= 3 / 2) && (y < (row - 3 / 2)))
	{
		for (int c = 0; c < 3; c++)
		{
			unsigned char filterVector[9];

			for (int ky = -3 / 2; ky <= 3 / 2; ky++)
			{
				for (int kx = -3 / 2; kx <= 3 / 2; kx++)
				{
					filterVector[ky * 3 + kx] = img[((y + ky) * column + (x + kx)) * 3 + c];
				}
			}

			sort(filterVector);
			output[(y * column + x) * 3 + c] = filterVector[(9) / 2];
		}
	}
}
__global__ void sharpeningFilter(unsigned char* img, float* mask, unsigned char* output, unsigned int column, unsigned int row)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float kernel[3][3];

	int counter = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			kernel[i][j] = mask[counter];
			counter++;
		}
	}

	if ((x >= 3 / 2) && (x < (column - 3 / 2)) && (y >= 3 / 2) && (y < (row - 3 / 2)))
	{
		for (int c = 0; c < 3; c++)
		{
			float sum = 0;

			for (int ky = -3 / 2; ky <= 3 / 2; ky++)
			{
				for (int kx = -3 / 2; kx <= 3 / 2; kx++)
				{
					float fl = img[((y + ky) * column + (x + kx)) * 3 + c];
					sum += fl * kernel[ky + 3 / 2][kx + 3 / 2];
				}
			}
			output[(y * column + x) * 3 + c] = sum;
		}
	}
}
__global__ void SaturationKernel(unsigned char* img, float saturation, unsigned char* output, int column, int row)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < column && y < row)
	{
		int idx = (y * column + x) * 3;

		// Read the pixel
		unsigned char r = img[idx];
		unsigned char g = img[idx + 1];
		unsigned char b = img[idx + 2];

		// Convert to HSV
		float h, s, v;
		RGBtoHSV(r, g, b, &h, &s, &v);

		// Adjust the saturation
		s *= saturation;

		// Convert back to RGB
		HSVtoRGB(h, s, v, &r, &g, &b);

		// Write the pixel back
		output[idx] = r;
		output[idx + 1] = g;
		output[idx + 2] = b;
	}
}
__global__ void AverageX15(unsigned char* img, unsigned char* output, int column, int row)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= 7 / 2) && (x < (column - 7 / 2)) && (y >= 7 / 2) && (y < (row - 7 / 2)))
	{
		for (int c = 0; c < 3; c++)
		{
			unsigned char filterVector[7 * 7];

			for (int ky = -7 / 2; ky <= 7 / 2; ky++)
			{
				for (int kx = -7 / 2; kx <= 7 / 2; kx++)
				{
					filterVector[ky * 7 + kx] = img[((y + ky) * column + (x + kx)) * 3 + c];
				}
			}

			output[(y * column + x) * 3 + c] = filterVector[(7 * 7) / 2];
		}
	}
}
__global__ void HELM_calc(unsigned char* img, unsigned char* average, unsigned char* output)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * 3;

	for (int i = 0; i < 3; i++)
	{
		if (average[idx + i] >= img[idx + i])
		{
			output[idx + i] = average[idx + i];
		}
		else
		{
			output[idx + i] = img[idx + i];
		}
	}
}
__global__ void Histogram(unsigned char* img, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = (x + y * gridDim.x) * 3;

	atomicAdd(&Histogram_Blue[img[Image_Idx]], 1);
	atomicAdd(&Histogram_Green[img[Image_Idx + 1]], 1);
	atomicAdd(&Histogram_Red[img[Image_Idx + 2]], 1);
}
__global__ void SumHistogram(int* output, int* Histogram_Blue, int* Histogram_Green, int* Histogram_Red)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	output[i] = Histogram_Blue[i] + Histogram_Green[i] + Histogram_Red[i];
}

__device__ void sort(unsigned char* filterVector)
{
	for (int i = 0; i < 9; i++)
	{
		for (int j = i + 1; j < 9; j++)
		{
			if (filterVector[i] > filterVector[j])
			{
				unsigned char tmp = filterVector[i];
				filterVector[i] = filterVector[j];
				filterVector[j] = tmp;
			}
		}
	}
}
__device__ void RGBtoHSV(unsigned char r, unsigned char g, unsigned char b, float* h, float* s, float* v)
{
	float red = r / 255.0f;
	float green = g / 255.0f;
	float blue = b / 255.0f;

	float cmax = fmaxf(red, fmaxf(green, blue));
	float cmin = fminf(red, fminf(green, blue));
	float delta = cmax - cmin;

	// Hue calculation
	if (delta == 0)
	{
		*h = 0;
	}
	else if (cmax == red)
	{
		*h = 60.0f * fmodf(((green - blue) / delta), 6.0f);
	}
	else if (cmax == green)
	{
		*h = 60.0f * (((blue - red) / delta) + 2.0f);
	}
	else
	{
		*h = 60.0f * (((red - green) / delta) + 4.0f);
	}

	// Saturation calculation
	*s = (cmax == 0) ? 0 : (delta / cmax);

	// Value calculation
	*v = cmax;
}
__device__ void HSVtoRGB(float h, float s, float v, unsigned char* r, unsigned char* g, unsigned char* b)
{
	float c = v * s;
	float x = c * (1 - fabsf(fmodf(h / 60.0f, 2) - 1));
	float m = v - c;
	float r_, g_, b_;

	if (h >= 0 && h < 60)
	{
		r_ = c, g_ = x, b_ = 0;
	}
	else if (h >= 60 && h < 120)
	{
		r_ = x, g_ = c, b_ = 0;
	}
	else if (h >= 120 && h < 180)
	{
		r_ = 0, g_ = c, b_ = x;
	}
	else if (h >= 180 && h < 240)
	{
		r_ = 0, g_ = x, b_ = c;
	}
	else if (h >= 240 && h < 300)
	{
		r_ = x, g_ = 0, b_ = c;
	}
	else
	{
		r_ = c, g_ = 0, b_ = x;
	}

	*r = (unsigned char)((r_ + m) * 255.0f);
	*g = (unsigned char)((g_ + m) * 255.0f);
	*b = (unsigned char)((b_ + m) * 255.0f);
}

void Changes(std::vector<float> odds_first, std::vector<float> odds_second)
{
	std::cout << std::endl << "---------------------------------------" << std::endl;
	std::cout << "Change: " << std::endl;
	std::cout << "ACMO_CUDA: " << odds_first[0] - odds_second[0] << std::endl;
	std::cout << "HELM_CUDA: " << odds_first[1] - odds_second[1] << std::endl;
	std::cout << "GLVM_CUDA: " << odds_first[2] - odds_second[2] << std::endl;
	std::cout << "ACMO: " << odds_first[3] - odds_second[3] << std::endl;
	std::cout << "HISE: " << odds_first[4] - odds_second[4] << std::endl;
	std::cout << "---------------------------------------" << std::endl;
}
void CalcMetrics(std::vector<int> list, cv::Mat& img, std::vector<float>& odds)
{
	float coffACMO_CUDA = 0.0;
	float coffHELM_CUDA = 0.0;
	float coffGLVM_CUDA = 0.0;
	float coffACMO = 0.0;
	float coffHISE = 0.0;

	for (int i = 0; i < list.size(); i++)
	{
		switch (list[i])
		{
		case 1:
			coffACMO_CUDA = ACMO_CUDA(img);
			break;
		case 2:
			coffHELM_CUDA = HELM_CUDA(img);
			break;
		case 3:
			coffGLVM_CUDA = GLVM_CUDA(img);
			break;
		case 4:
			coffACMO = ACMO(img);
			break;
		case 5:
			coffHISE = HISE(img);
			break;
		case 6:
			coffACMO_CUDA = ACMO_CUDA(img);
			coffHELM_CUDA = HELM_CUDA(img);
			coffGLVM_CUDA = GLVM_CUDA(img);
			coffACMO = ACMO(img);
			coffHISE = HISE(img);
			break;
		default:
			std::cout << "Error" << std::endl;
			break;
		}
	}
	
	odds[0] = coffACMO_CUDA;
	odds[1] = coffHELM_CUDA;
	odds[2] = coffGLVM_CUDA;
	odds[3] = coffACMO;
	odds[4] = coffHISE;
}
void SelectingFunctions(std::vector<int>& dst)
{
	std::vector<std::string> list = {"ACMO_CUDA","HELM_CUDA","GLVM_CUDA","ACMO","HISE"};
	std::vector<int> selected;
	
	std::cout << "Enter the function numbers for their operation (when finished selecting, enter 0): " << std::endl;

	for (int i = 0; i < list.size(); i++)
	{
		std::cout << i + 1 << ". " << list[i] << std::endl;

		if (i + 1 == list.size())
		{
			std::cout << i + 2 << ". " << "Use all functions" << std::endl;
		}
	}

	int choice;
	std::cout << "Enter item number: ";
	std::cin >> choice;

	while (choice != 0)
	{
		if (choice >= 1 && choice <= list.size() + 1)
		{
			selected.push_back(choice);
		}
		else 
		{
			std::cout << "Invalid input" << std::endl;
		}

		std::cout << "Enter next item number (or 0 to complete): ";
		std::cin >> choice;
	}

	dst = selected;
}