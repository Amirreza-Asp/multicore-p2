
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <time.h>
#include <chrono>
#include <stdio.h>
#include <windows.h> 
#include "stdafx.h"
#include <opencv.hpp>

using namespace cv;
using namespace std;
#ifdef _DEBUG
#pragma comment (lib, "opencv_world452d.lib")
#else
#pragma comment (lib, "opencv_world452.lib")
#endif

void changeConsoleColor(int desiredColor) {
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), desiredColor);
}

void GetRGBFromInt(int IntegerToConvert, unsigned char&R, unsigned char&G, unsigned char&B)
{
	unsigned char buffer[4];
	memcpy(buffer, (char*)&IntegerToConvert, 3);
	R = buffer[0];
	G = buffer[1];
	B = buffer[2];
}

int ConvertRGBToInt(unsigned char R, unsigned char G, unsigned char B)
{
	int ReturnInt = 0;
	unsigned char Padding = 0;
	unsigned char buffer[3];
	buffer[0] = R;
	buffer[1] = G;
	buffer[2] = B;
	//buffer[3] = Padding;
	memcpy((char*)&ReturnInt, buffer, 3);
	return ReturnInt;
}

#define BLUR_SIZE 5
#define CHANNEL 3
#define IMAGETYPE CV_8UC3
__global__ void bluringKernel(unsigned char *input, unsigned char *blured, int height, int width) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int column = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < height && column < width) {
		int redValue = 0;
		int greenValue = 0;
		int blueValue = 0;
		int pixels = 0;
		for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
			for (int blurColumn = -BLUR_SIZE; blurColumn < BLUR_SIZE + 1; ++blurColumn) {
				int currentRow = row + blurRow;
				int currentColumn = column + blurColumn;
				if (currentRow > -1 && currentRow < height-2 && currentColumn > -1 && currentColumn < width-2) {
					int offset = (currentRow * width + currentColumn) * CHANNEL;
					redValue += input[offset + 2];
					greenValue += input[offset + 1];
					blueValue += input[offset];
					pixels++;
				}
			}
		}
		int resultOffset = (row * width + column) * CHANNEL;
		blured[resultOffset + 2] = (unsigned char)(redValue / pixels);
		blured[resultOffset + 1] = (unsigned char)(greenValue / pixels);
		blured[resultOffset + 0] = (unsigned char)(blueValue / pixels);
	}
}

int main(void) {
	changeConsoleColor(8);
	Mat originalImage = imread("cameraman.tiff", IMREAD_COLOR);//
	if (!originalImage.data || originalImage.empty()) {
		printf("It can't read image\n");
		system("pause");
		return -1;
	}
	imshow("original image", originalImage);

	const int height = originalImage.size().height;
	const int width = originalImage.size().width;
	const int pixelsCount = height * width;
	const int memoryNeeded = pixelsCount * CHANNEL;
	unsigned char *original_host, *blured_host;
	original_host = (unsigned char*)malloc(memoryNeeded);
	blured_host = (unsigned char*)malloc(memoryNeeded);
	int i = 0, j = 0;
	for (i = 0; i < width; i++) {
		for (j = 0;j < height;j++) {
			Vec3b tmp = originalImage.at<Vec3b>(i,j);
			int offset = (i * width + j) * CHANNEL;
			original_host[offset] = tmp[0];
			original_host[offset + 1] = tmp[1];
			original_host[offset + 2] = tmp[2];
		}
	}
	auto startSerial = chrono::high_resolution_clock::now();
	for (int i = 0;i < width;i++) {
		for (int j = 0;j < height;j++) {
			int redValue = 0;
			int greenValue = 0;
			int blueValue = 0;
			int pixels = 0;
			for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
				for (int blurColumn = -BLUR_SIZE; blurColumn < BLUR_SIZE + 1; ++blurColumn) {
					int currentRow = i + blurRow;
					int currentColumn = j + blurColumn;
					if (currentRow > -1 && currentRow < height - 2 && currentColumn > -1 && currentColumn < width - 2 ) {
						int offset = (currentRow * width + currentColumn) * CHANNEL;
						redValue += original_host[offset + 2];
						greenValue += original_host[offset + 1];
						blueValue += original_host[offset];
						pixels++;
					}
				}
			}
			int resultOffset = (i * width + j) * CHANNEL;
			blured_host[resultOffset + 2] = (unsigned char)(redValue / pixels);
			blured_host[resultOffset + 1] = (unsigned char)(greenValue / pixels);
			blured_host[resultOffset + 0] = (unsigned char)(blueValue / pixels);
		}
	}
	auto endSerial = chrono::high_resolution_clock::now();
	double timeTakenSerial = chrono::duration_cast<chrono::nanoseconds>(endSerial - startSerial).count();
	timeTakenSerial *= 1e-9;
	changeConsoleColor(10);
	printf("***************************************************\n");
	printf("Time taken by serial bluring : %f sec\n", timeTakenSerial);
	printf("***************************************************\n");
	Mat serialResult(height, width, IMAGETYPE, (void*)blured_host);
	imshow("serial result", serialResult);

	int dev_count;
	cudaGetDeviceCount(&dev_count);
	cudaDeviceProp dev_prop;
	int minor;
	for (i = 0;i < dev_count;i++) {
		cudaGetDeviceProperties(&dev_prop, i);
		minor = dev_prop.minor;
	}
	blured_host = (unsigned char*)malloc(memoryNeeded);
	unsigned char *original_device, *blured_device;
	cudaMalloc((void**)&original_device, memoryNeeded);
	cudaMemcpy(original_device, original_host, memoryNeeded, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&blured_device, memoryNeeded);
	int threadsCount = pow(2,minor);
	const dim3 dimGrid((int)ceil(width / threadsCount), (int)ceil(height / threadsCount));
	const dim3 dimBlock(threadsCount, threadsCount);
	auto startByGPU = chrono::high_resolution_clock::now();
	bluringKernel << <dimGrid, dimBlock >> >(original_device, blured_device, height, width);
	auto endByGPU = chrono::high_resolution_clock::now();
	double timeTakenGPU = chrono::duration_cast<chrono::nanoseconds>(endByGPU - startByGPU).count();
	timeTakenGPU *= 1e-9;
	changeConsoleColor(11);
	printf("***************************************************\n");
	printf("kernel lunch parameters: dimGrid=(%d,%d), dimBlock=(%d,%d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
	printf("Time taken by using GPU is : %f sec\n", timeTakenGPU);
	printf("***************************************************\n");
	cudaMemcpy(blured_host, blured_device, memoryNeeded, cudaMemcpyDeviceToHost);
	Mat GPUresult(height, width, IMAGETYPE, (void *)blured_host);
	imshow("GPU result", GPUresult);
	imwrite("result.lena512color.tiff", GPUresult);

	cudaFree(original_device);
	cudaFree(blured_device);
	//free(original_host);
	//free(blured_host);

	waitKey(0);
    return 0;
}