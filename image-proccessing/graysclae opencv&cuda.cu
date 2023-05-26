
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

int getSPcores(cudaDeviceProp devProp) {
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2:
		printf("Fermi\n");
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3:
		printf("Kepler\n");
		cores = mp * 192;
		break;
	case 5:
		printf("Maxwell\n");
		cores = mp * 128;
		break;
	case 6:
		printf("Pascal\n");
		if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 7:
		printf(" Volta and Turing\n");
		if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 8:
		printf("Ampere\n");
		if (devProp.minor == 0) cores = mp * 64;
		else if (devProp.minor == 6) cores = mp * 128;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

#define CHANNELS 3
__global__ void grayscaleKernel(unsigned char *rgb, unsigned char *grayscale, int height, int width) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int column = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < height && column < width) {
		int grayOffset = row*width + column;
		int rgbOffset = grayOffset * CHANNELS;
		unsigned char red = rgb[rgbOffset + 2];
		unsigned char green = rgb[rgbOffset + 1];
		unsigned char blue = rgb[rgbOffset];
		grayscale[grayOffset] = 0.21f*red + 0.71f*green + 0.07f*blue;
	}
}

int main(void) {
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	cudaDeviceProp dev_prop;
	changeConsoleColor(9);
	printf("GPU count: %d\n", dev_count);
	int minor;
	for (int i = 0;i < dev_count;i++) {
		printf("=================================================\n");
		cudaGetDeviceProperties(&dev_prop, i);
		printf("Name: %s\n", dev_prop.name);
		printf("cuda core(SPs): %d\n",getSPcores(dev_prop));
		printf("SMs: %d\n",dev_prop.multiProcessorCount);
		printf("max Thread/Block: %d\n", dev_prop.maxThreadsPerBlock);
		printf("max Thread/SM: %d\n", dev_prop.maxThreadsPerMultiProcessor);
		printf("clock rate: %d\n",dev_prop.clockRate);
		printf("max Thread/(dimention of block): (%d,%d,%d)\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
		printf("max grid size: (%d,%d,%d)\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
		printf("sharedMem/Block: %d\n", dev_prop.sharedMemPerBlock);
		printf("warp size: %d\n", dev_prop.warpSize);
		printf("global memory: %d\n", dev_prop.totalGlobalMem);
		printf("constatnt memory: %d\n", dev_prop.totalConstMem);
		printf("l2 Cache: %d\n", dev_prop.l2CacheSize);
		minor = dev_prop.minor;
		printf("minor: %d, major: %d\n",dev_prop.minor,dev_prop.major);
		printf("=================================================\n");
	}
	changeConsoleColor(8);

	Mat originalImage = imread("peppers.tiff", IMREAD_COLOR);
	if (!originalImage.data || originalImage.empty()) {
		printf("It can't read image");
		system("pause");
		return -1;
	}
	imshow("original image", originalImage);
	Mat grayscaleByCV;
	auto startByCV = chrono::high_resolution_clock::now();
	cvtColor(originalImage, grayscaleByCV, COLOR_BGR2GRAY);
	auto endByCV = chrono::high_resolution_clock::now();
	double timeTakenCV = chrono::duration_cast<chrono::nanoseconds>(endByCV - startByCV).count();
	timeTakenCV *= 1e-9;
	changeConsoleColor(10);
	printf("***************************************************\n");
	printf("Time taken by CV converting is : %f sec\n", timeTakenCV);
	printf("***************************************************\n");
	changeConsoleColor(11);

	const int height = originalImage.rows;
	const int width = originalImage.cols;
	const int pixelsCount = height * width;
	const int rgbMemoryNeeded = pixelsCount * CHANNELS * sizeof(unsigned char);
	const int grayscaleMemoryNeeded = pixelsCount * sizeof(unsigned char);
	unsigned char *rgb_host, *grayscale_host;
	rgb_host = (unsigned char*)malloc(rgbMemoryNeeded);
	grayscale_host = (unsigned char*)malloc(grayscaleMemoryNeeded);
	int i = 0, j = 0;
	for (i = 0; i < height; i++) {
		for (j = 0;j < width;j++) {
			Vec3b tmp = originalImage.at<Vec3b>(i, j);
			//printf("(%d,%d): r=%d b=%d g=%d\r\n",i,j, tmp[2],tmp[1],tmp[0]);
			int offset = (i*height + j) * CHANNELS;
			rgb_host[offset] = tmp[0];
			rgb_host[offset + 1] = tmp[1];
			rgb_host[offset + 2] = tmp[2];
		}
	}
	unsigned char *rgb_device, *grayscale_device;
	cudaMalloc((void**)&rgb_device, rgbMemoryNeeded);
	cudaMemcpy(rgb_device, rgb_host, rgbMemoryNeeded, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&grayscale_device, grayscaleMemoryNeeded);
	int threadsCount = pow(2, minor);
	const dim3 dimGrid((int)ceil(width/ threadsCount), (int)ceil(height/ threadsCount));
	const dim3 dimBlock(threadsCount, threadsCount);
	auto startByGPU = chrono::high_resolution_clock::now();
	grayscaleKernel<<<dimGrid, dimBlock>>>(rgb_device, grayscale_device, height, width);
	auto endByGPU = chrono::high_resolution_clock::now();
	double timeTakenGPU = chrono::duration_cast<chrono::nanoseconds>(endByGPU - startByGPU).count();
	timeTakenGPU *= 1e-9;
	printf("***************************************************\n");
	printf("kernel lunch parameters: dimGrid=(%d,%d), dimBlock=(%d,%d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
	printf("Time taken by using GPU is : %f sec\n", timeTakenGPU);
	printf("***************************************************\n");
	cudaMemcpy(grayscale_host, grayscale_device, grayscaleMemoryNeeded, cudaMemcpyDeviceToHost);
	Mat GPUresult(height, width, CV_8UC1, (void *)grayscale_host);
	imshow("GPU result", GPUresult);
	imwrite("result.tiff", GPUresult);

	cudaFree(rgb_device);
	cudaFree(grayscale_device);
	free(rgb_host);
	free(grayscale_host);

	waitKey(0);
    return 0;
}