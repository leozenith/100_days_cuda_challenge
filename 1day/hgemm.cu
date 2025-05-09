#include "mma.h"
#include <cuda_fp16.h>  // 定义half类型
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
using namespace nvcuda;  // 启用WMMA接口
#define WMMA_M 16
#define WMMA_N 16 
#define WMMA_K 16
#define WARP_SIZE 32
#define div_ceil(X,Y) ((X+Y-1)/Y)
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
__global__
void hgemm(const half* A, const half* B, half* C, size_t M, size_t N, size_t K){
    const size_t k_tiles = div_ceil(K, WMMA_K);
    const int warp_row = blockIdx.y * WMMA_M;
    const int warp_col = blockIdx.x * WMMA_N;

    //C_fragment
    wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,half> C_frag;
    wmma::fill_fragment(C_frag, 0.0f);
    for(int i = 0; i < k_tiles; ++i){
        wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half, wmma::col_major> B_frag;
        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + K * warp_col + i * WMMA_K, K);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }
    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

void wmmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));

    hgemm<<<grid, block>>>(A, B, C, M, N, K);
}

// 初始化half类型矩阵（FP16）
void init_half_matrix(half *h_mat, int size, float value) {
    for (int i = 0; i < size; ++i) {
        h_mat[i] = __float2half(value);
    }
}

// 验证结果矩阵
bool verify_result(half *h_C, int M, int N, int K) {
    const float expected = K * 1.0f; // A和B初始化为全1时，C的每个元素应为K
    for (int i = 0; i < M*N; ++i) {
        if (fabs(__half2float(h_C[i]) - expected) > 1e-3) {
            printf("Validation failed at C[%d]: %.2f vs %.2f\n", 
            i, __half2float(h_C[i]), expected);
            return false;
        }
    }
    return true;
}

int main() {
    const size_t M = 512, N = 512, K = 512;  // 测试矩阵维度
    
    // 分配主机内存
    half *h_A = new half[M*K];
    half *h_B = new half[K*N];
    half *h_C = new half[M*N];
    
    // 初始化数据
    init_half_matrix(h_A, M*K, 1.0f);
    init_half_matrix(h_B, K*N, 1.0f);
    
    // 分配设备内存
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M*K*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K*N*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M*N*sizeof(half)));
    
    // 数据拷贝到设备
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice));
    
    // 执行内核
    wmmaNaive(d_A, d_B, d_C, M, N, K);
    
    // 同步并检查错误
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M*N*sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证结果
    if (verify_result(h_C, M, N, K)) {
        printf("Validation PASSED!\n");
    } else {
        printf("Validation FAILED!\n");
    }
    
    // 释放资源
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}