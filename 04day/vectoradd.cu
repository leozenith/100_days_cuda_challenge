#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h> 
#include <chrono>
#include <iostream>
constexpr int warpSize = 32;
__device__ float warp_reduce(float val){
    for(int offset = warpSize/2; offset>0; offset>>=1)
       val += __shfl_down_sync(0xffffffff, val, offset);
    return val; 
}


__global__ void vector_sum(float *input, float* output, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = blockDim.x * blockIdx.x + tid;
    const int warpID = tid / warpSize;
    const int laneID = tid % warpSize;
    extern __shared__ float shm[];
    float sum = 0.0f;
    for(int i = idx; i < N; i += blockDim.x*gridDim.x){
        sum += input[i];
    }
    sum = warp_reduce(sum);
    if(laneID == 0)shm[warpID] = sum;
    __syncthreads();
    if(warpID==0){
        sum = laneID < blockDim.x / warpSize ? shm[laneID]:0;
        sum = warp_reduce(sum);
        if(laneID==0)output[bid] = sum; 
    }
}

void verify_result(float* host_array, float* device_output, int N) {
    float cpu_sum = 0.0f;
    for(int i = 0; i < N; ++i) {
        cpu_sum += host_array[i];
    }
    
    float gpu_sum = 0.0f;
    cudaMemcpy(&gpu_sum, device_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    const float epsilon = 1e-5f;
    if(fabs(cpu_sum - gpu_sum) > epsilon) {
        printf("Verification failed! CPU: %f, GPU: %f\n", cpu_sum, gpu_sum);
    } else {
        printf("Verification passed! Sum: %f\n", gpu_sum);
    }
}

int main() {
    const int N = 1 << 20;  // 1M元素
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    
    // 分配主机内存
    float* host_array = new float[N];
    for(int i = 0; i < N; ++i) {
        host_array[i] = 1.0f;  // 初始化为1便于验证
    }
    
    // 分配设备内存
    float *device_input, *device_output;
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMalloc(&device_output, gridSize * sizeof(float));
    
    // 拷贝数据到设备
    cudaMemcpy(device_input, host_array, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 计算共享内存大小（每个Warp一个float）
    const size_t sharedMemSize = (blockSize / warpSize) * sizeof(float);
    
    // 启动核函数
    vector_sum<<<gridSize, blockSize, sharedMemSize>>>(device_input, device_output, N);
    
    // 可选：二次归约（如果gridSize > 1）
    if(gridSize > 1) {
        float* final_output;
        cudaMalloc(&final_output, sizeof(float));
        vector_sum<<<1, blockSize, sharedMemSize>>>(device_output, final_output, gridSize);
        cudaMemcpy(device_output, final_output, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(final_output);
    }
    
    // 验证结果
    verify_result(host_array, device_output, N);
    
    // 释放资源
    delete[] host_array;
    cudaFree(device_input);
    cudaFree(device_output);
    
    return 0;
}