#include <cuda_runtime.h>
#include <stdio.h>

__global__
void convolve(float *image, float *output, float *kernel, int width, int height, int kernelSize){
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int half_size = kernelSize / 2;
    float sum = 0.0f;
    if(x >= half_size && x < (width - half_size) && y >= half_size && y< (height - half_size)){
        for(int i = -half_size; i <= half_size; i++){
            for(int j = -half_size; j<= half_size; j++){
                const int imgIdx = (y + j) * width + (x + i);
                const int kernelIdx = (j + half_size) * kernelSize + (i + half_size);
                sum += kernel[kernelIdx] * image[imgIdx]; 
            }
        }
        output[y * width + x] = sum; 
    }else if(x < width && y < height){
        const int imgIdx = y * width + x;
        output[imgIdx] = image[imgIdx];
    }
}

int main() {
    int width = 1024, height = 1024, kernelSize = 3;
    size_t imgSize = width * height * sizeof(float);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);

    float *h_image = (float*)malloc(imgSize);
    float *h_output = (float*)malloc(imgSize);
    float *h_kernel = (float*)malloc(kernelSizeBytes);

    float *d_image, *d_output, *d_kernel;
    cudaMalloc(&d_image, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_image, h_image, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    convolve<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_output, d_kernel, width, height, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(h_image);
    free(h_output);
    free(h_kernel);

    return 0;
}