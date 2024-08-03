// V0 optimization
// Matrix tiling, 使用shared memory/SMEM, split-by-k


#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

float testError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);
float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}



__global__ void sgemm_V0(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    /*
    dim3 blockDim(BN/TN, BM/TM) = (16, 16)，即一个block中有256个thread
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM) = (4，4)，即一共16个block
    */

    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
    // const int TM = 8;
    // const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x; // thread在对应block内的行id
    const int ty = threadIdx.y; // thread在对应block内的列id
    const int tid = ty * blockDim.x + tx; // thread在对应block中的全局id（从左到右，从上到下，从0开始逐一标）

    /*
    在SMEM上对A和B，分别开辟大小为(BM, BK), (BK, BN)的空间
    */
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // load data
    /*
    例：
    一个thread搬运s_a中的1个数据，s_b中的一个数据,BM=BN=BK=32，和block dim维持一致
    对于tid = 0的thread，以下四个值分别为((0, 0), (0, 0)),
    意味着它负责把s_a(0,0)，s_b(0,0)从global memory加载到SMEM

    对于tid = 1的thread，以下四个值分别为((0, 1), (0, 1)),
    意味着它负责把s_a(0,1)，s_b(0,1)，从global memory加载到SMEM
   
    对于tid = 34的thread，以下四个值分别为((1, 2), (1, 2))，含义同上
    */
    
    // 当前thread负责把A中的相关数据从global memory加载到SMEM，计算该data的row
    int load_a_smem_m = tid / 32;  // row of s_a
    // 当前thread负责加载的数在s_a中的col
    int load_a_smem_k = tid % 32;  // col of s_a


    // 当前thread负责把B中的相关数据从global memory加载到SMEM
    int load_b_smem_k = tid / 32;   // row of s_b
    // 当前thread负责加载的第一个数在s_b中的col
    int load_b_smem_n = tid % 32;  // col of s_b

    /*
    */
    // glocal row: blockIdx.y * blockdim.y + threadIdx.y; 这里threadIdx.y是local row，在s_a里就是刚刚计算出的load_a_smem_m
    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    // glocal col: blockIdx.x * blockDim.x + threadIdx.x; 这里threadIdx.x是local col，在s_b里就是刚刚计算出的load_b_smem_n
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b


    /*
    对每个block，它都要经历K/Bk 次循环，每次循环计算一块s_a * s_b的结果
    这也意味着，对每个block内的每个thread，它的外循环也是16次
    */
    // 外循环block
    float sum = 0.f;
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {

        // 在这次循环中，当前thread从s_a上取的第一个数，其位置在A（位于global memory）上的col，与load_a_gmem_m对应
        int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
        // 在这次循环中，当前thread从s_a上取的数，在A中的地址，即A[load_a_gmem_m][load_a_gmem_k]
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        // 从这个地址开始，从A所在的global memory上，加载到s_a上
        s_a[load_a_smem_m][load_a_smem_k] = a[load_a_gmem_addr];

        // 在这次循环中，当前thread从s_b上取的第一个数，其位置在B（位于global memory）上的row，与load_b_gmem_n对应
        int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
        // 在这次循环中，当前thread从s_b上取的第一个数，在B中的地址，即B[load_b_gmem_k][load_b_gmem_n]
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        // 同理将相关的数据从global memory加载到SMEM上
        s_b[load_b_smem_k][load_b_smem_n] = b[load_b_gmem_addr];

        // 在所有thread间做一次同步，保证在下面的计算开始时，s_a, s_b相关的数据已经全部从global memory搬运到SMEM上了
        __syncthreads();
        
        // 内循环thread, 每个thread计算s_a一行*s_b一列的结果
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            int comp_a_smem_m = load_a_smem_m;
            int comp_b_smem_n = load_b_smem_n;
            sum += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
        }

        __syncthreads();
    }

    /*
    3. 
    此时，所有的block已做完循环，
    我们把当前thread计算出的结果写回global memory上的C矩阵对应位置中
    */
    int store_c_gmem_m = load_a_gmem_m;
    int store_c_gmem_n = load_b_gmem_n;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    c[store_c_gmem_addr] = sum;
}

int main(void) {
    printf("\nKernal = sgemm_V0\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 32, BN = 32;
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = sgemm_V0;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(BN, BM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    
    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN, BM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}


float testError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}


float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}
