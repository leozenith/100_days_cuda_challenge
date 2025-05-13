#include "mma.h"
#include <cuda_fp16.h>  
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
using namespace nvcuda; 
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define DIV(X,Y) ((X+Y-1)/Y)
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__
void hgemm(half* A, half* B, half* C, size_t M, size_t N, size_t K){
    const int warp_row = blockIdx.y * WMMA_N;
    const int warp_col = blockIdx.x * WMMA_M;
    if (warp_row >= M || warp_col >= N) {
        return;
    }
    const int Kiter = (K + WMMA_K - 1) / WMMA_K;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> Cfrag;
    wmma::fill_fragment(Cfrag, 0.0f);
    for(int k = 0; k < Kiter; ++k){
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Afrag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> Bfrag;
        const int offset_a = warp_row * K + k * WMMA_K;
        const int offset_b = warp_col * K + k * WMMA_K;    
        wmma::load_matrix_sync(Afrag, A + offset_a, K);
        wmma::load_matrix_sync(Bfrag, B + offset_b, K);
        wmma::mma_sync(Cfrag, Afrag, Bfrag, Cfrag);
    }
    wmma::store_matrix_sync(C + warp_row * N + warp_col, Cfrag, N, wmma::mem_row_major);
}

void wmmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(DIV(N, WMMA_N), DIV(M, WMMA_M));

    hgemm<<<grid, block>>>(A, B, C, M, N, K);
}


void initMatrix(half* mat, size_t rows, size_t cols, float minVal = -1.0f, float maxVal = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(minVal, maxVal);
    
    for (size_t i = 0; i < rows * cols; ++i) {
        mat[i] = __float2half(dis(gen));
    }
}

bool verifyResult(const half* C, const half* C_ref, size_t M, size_t N, float epsilon = 1e-1f) {
    for (size_t i = 0; i < M * N; ++i) {
        float val = __half2float(C[i]);
        float ref_val = __half2float(C_ref[i]);
        if (fabs(val - ref_val) > epsilon) {
            std::cout << "Mismatch at (" << i / N << "," << i % N << "): "
                      << val << " vs " << ref_val << std::endl;
            return false;
        }
    }
    return true;
}

void cpuMatMul(const half* A, const half* B, half* C, size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += __half2float(A[m * K + k]) * __half2float(B[n * K + k]);
            }
            C[m * N + n] = __float2half(sum);
        }
    }
}

int main() {
    const size_t M = 256, N = 256, K = 256;

    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<half> h_C(M * N);
    std::vector<half> h_C_ref(M * N);

    initMatrix(h_A.data(), M, K);
    initMatrix(h_B.data(), N, K);

    cpuMatMul(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    wmmaNaive(d_A, d_B, d_C, M, N, K);

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    if (verifyResult(h_C.data(), h_C_ref.data(), M, N)) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    return 0;
}