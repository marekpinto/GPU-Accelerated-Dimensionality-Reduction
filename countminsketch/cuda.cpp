#include <cuda_runtime.h>
#include <iostream>

__device__ uint32_t MurmurHash3(uint32_t key, uint32_t seed) {
    key ^= seed;
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}


// Kernel to insert item
__global__ void insert(float* estimates, int N, int K, int item) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        int bucket = MurmurHash3(item, row) % K;
        estimates[row * K + bucket] += 1;

    }
}

__global__ void remove(float* estimates, int N, int K, int item) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        int bucket = MurmurHash3(item, row) % K;
        estimates[row * K + bucket] -= 1;
    }
}

//reduce based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] = std::min(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = std::min(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = std::min(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8) sdata[tid] = std::min(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4) sdata[tid] = std::min(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2) sdata[tid] = std::min(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize> //set at compile time (very efficient)
__global__ void query(float* estimates, float* output, int N, int K, int item) {

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = INT_MAX;

    while (i < N) { 
        int bucket1 = MurmurHash(i);
        int bucket2 = MurmurHash(i+blockSize);
        sdata[tid] = std::min(sdata[tid], std::min(estimates[i][bucket1], g_idata[i+blockSize][bucket2]));
        i += gridSize; 
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = std::min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = std::min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = std::min(sdata[tid], sdata[tid + 64]); } __syncthreads(); } 

    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) output[blockIdx.x] = sdata[0];

}
