#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "metricks.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

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
		-0.05 - 0.15 * step, 0.015 + 0.05 * step, -0.05 - 0.15 * step,
		0.015 + 0.05 * step, 1.3 + 0.5 * step, 0.015 + 0.05 * step,
		-0.05 - 0.15 * step, 0.015 + 0.05 * step, -0.05 - 0.15 * step
	};

	if (step < 0)
	{
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
cv::Mat SimpleDeNoise_CUDA(cv::Mat& img)
{
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
