#include <cuda_runtime.h>
#include <stdio.h>

__global__
void clear_data(float* data, int N){
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        data[idx]=(float)0;
    }
}

__global__ 
void coalescedKernel(float* input, float* output, int N){
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        output[idx] = input[idx] * 2.0f;
    }
}

__global__ 
void nonCoalescedKernel(float* input, float* output, int stride, int N){
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        const int i = (idx * stride) % N;
        output[i] = input[i] * 2.0f;
    }
}

int main() {
    int N = 1 << 40; // 1M elements
    size_t size = N * sizeof(float);
    cudaError_t err;

    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    float *d_input, *d_output, *d_dummy;
    err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess) { fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
    err = cudaMalloc((void**)&d_output, size);
    if (err != cudaSuccess) { fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
    err = cudaMalloc((void**)&d_dummy, size); // For cache flushing
    if (err != cudaSuccess) { fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

    // Configure kernel execution
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int stride = 2;

    // Benchmark setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float millisecondsCoalesced = 0, millisecondsNonCoalesced = 0;

    //warmup
    for(int i=0; i < 10; i++)coalescedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    clear_data<<<blocksPerGrid, threadsPerBlock>>>(d_dummy, N);
    cudaDeviceSynchronize();

    // Benchmark coalesced kernel
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice); // Fresh input
    cudaMemset(d_output, 0, size); // Fresh output
    cudaEventRecord(start);
    coalescedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecondsCoalesced, start, stop);

    // Flush cache with dummy kernel
    clear_data<<<blocksPerGrid, threadsPerBlock>>>(d_dummy, N);
    cudaDeviceSynchronize();

    // Benchmark non-coalesced kernel
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice); // Reset input
    cudaMemset(d_output, 0, size); // Reset output
    cudaEventRecord(start);
    nonCoalescedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, stride);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecondsNonCoalesced, start, stop);

    // Print benchmark results
    printf("Coalesced Kernel Time:    %f ms\n", millisecondsCoalesced);
    printf("Non-Coalesced Kernel Time (stride=%d): %f ms\n", stride, millisecondsNonCoalesced);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_dummy);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
