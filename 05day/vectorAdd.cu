
#include <cuda_runtime.h>
__global__ void vector_add_v1(const float* A, const float* B, float* C, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vector_add_v2(const float* A, const float* B, float* C, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int vec_idx = idx;
    const int scalar_idx = idx * 4;
    
    float4* A_ = (float4*)A;
    float4* B_ = (float4*)B;
    float4* C_ = (float4*)C;
    
    // Process 4 elements at a time
    if(vec_idx < N/4) {
        float4 a4 = A_[vec_idx];
        float4 b4 = B_[vec_idx];
        C_[vec_idx] = make_float4(a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w);
    }
    
    // Handle remaining elements (1-3)
    if(scalar_idx + 3 >= N && scalar_idx < N) {
        for(int i = 0; i < 4 && (scalar_idx + i) < N; i++) {
            C[scalar_idx + i] = A[scalar_idx + i] + B[scalar_idx + i];
        }
    }
}

void solve(const float* A, const float* B, float* C, int N) {
    const int threadsPerBlock = 256;
    
    if(N < 256) {
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vector_add_v1<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    } else {
        // For vectorized version, we need threads = ceil(N/4)
        int blocksPerGrid = (N/4 + threadsPerBlock - 1) / threadsPerBlock;
        vector_add_v2<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    }
    
    cudaDeviceSynchronize();
}