// V1 optimization
// 使用register
// FlOAT4 vectorize: Coalesced memory access

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

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


__global__ void sgemm_V1(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    /*
    dim3 blockDim(BN/TN, BM/TM) = (16, 16)，即一个block中有256个thread
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM) = (4，4)，即一共16个block
    */

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

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

    /*
    初始化当前thread所维护的C矩阵（确定长度的数组，应该是定义在寄存器上的）
    */
    float r_c[TM][TN] = {0.0};

    // load data
    /*
    例：
    对于tid = 0的thread，以下四个值分别为((0, 0), (0, 0)),
    意味着它负责把s_a(0,0)开始的连续4个数，s_b(0,0)开始的连续4个数，从global memory加载到SMEM

    对于tid = 1的thread，以下四个值分别为((0, 4), (0, 4)),
    意味着它负责把s_a(0,4)开始的连续4个数，s_b(0,4)开始的连续4个数，从global memory加载到SMEM

    对于tid = 2的thread，以下四个值分别为((1, 0), (0, 8))
    此时s_a第一行的8个数已经被前面的thread取完了，所以现在从s_a第二行开始取，s_b第一行没取完，继续进行
   
    对于tid = 18的thread，以下四个值分别为((9, 0), (0, 72))，含义同上
    */
    
    // 当前thread负责把A中的相关数据从global memory加载到SMEM，
    // 这里在计算该thread负责加载的第一个数在s_a中的row
    // 一个thread load 4个data，s_a 每行8个data，所以row = tid/2
    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    // 当前thread负责加载的第一个数在s_a中的col
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a


    // 当前thread负责把B中的相关数据从global memory加载到SMEM，
    // 这里在计算该thread负责加载的第一个数在s_b中的row
    // 一个thread load 4个data，s_b 每行128个data，所以row = tid/32
    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    // 当前thread负责加载的第一个数在s_b中的col
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    /*
    例：
    对于tid = 0的thread，以下两个值为(256, 128)，
    表示该thread从s_a上取的第一个数，其位置在A（位于global memory）上的row 256
    该thread从s_b上取的第一个数，其位置在B（位于global memory）上的col 128
   
    对于tid = 18的thread，以下两个值为(265, 200)，道理同上
    */
    // glocal row: blockIdx.y * blockdim.y + threadIdx.y; 这里threadIdx.y是local row，在s_a里就是刚刚计算出的load_a_smem_m
    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    // glocal col: blockIdx.x * blockDim.x + threadIdx.x; 这里threadIdx.x是local col，在s_b里就是刚刚计算出的load_b_smem_n
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b


    /*
    对每个block，它都要经历K/Bk = 128/8 = 16次循环，每次循环计算一块s_a * s_b的结果
    这也意味着，对每个block内的每个thread，它的外循环也是16次
    */
    // 外循环block
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        /*
        1. 在block的单次循环中，需要把对应的s_a（高亮红）和s_b(高亮黄)从global memory
        加载到SMEM上，因此每个thread负责加载一部分s_a, s_b的数据，最后的__syncthreads()
        是保证thread们在正式计算前，都干完了自己加载的活，即完整的s_a, s_b已被加载到SMEM上
        */
       // 在这次循环中，当前thread从s_a上取的第一个数，其位置在A（位于global memory）上的col，与load_a_gmem_m对应
        int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
        // 在这次循环中，当前thread从s_a上取的第一个数，在A中的地址，即A[load_a_gmem_m][load_a_gmem_k]
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        // 从这个地址开始，取出连续的4个数，将其从A所在的global memory上，加载到s_a上
        // 注：采用FLOAT4的好处是便于连续访存。如果存储的4个数在地址上不连续，你就发4条指令。float4的数据类型就只要发1条指令，一起取出
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);

        // 在这次循环中，当前thread从s_b上取的第一个数，其位置在B（位于global memory）上的row，与load_b_gmem_n对应
        int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
        // 在这次循环中，当前thread从s_b上取的第一个数，在B中的地址，即B[load_b_gmem_k][load_b_gmem_n]
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        // 同理将相关的数据从global memory加载到SMEM上
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

        // 在所有thread间做一次同步，保证在下面的计算开始时，s_a, s_b相关的数据已经全部从global memory搬运到SMEM上了
        __syncthreads();
        
        // 内循环thread
        /*
        2. 在block的单次循环中，每个thread采用split-by-k的方式，
        逐步累加计算当前thread所维护的(TM, TN)块的结果
        */
        #pragma unroll
        for (int k = 0; k < BK; k++) {  //遍历每一个[TM, TN]需要的a和b
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m; // row in block
                    int comp_b_smem_n = tx * TN + n; // col in block
                    // 每次从SMEM上，各加载子block上的1个元素，到register，然后再计算
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }
        
        // 做一次同步，保证所有的thread都计算完当前所维护的（TM, TN）块
        __syncthreads();
    }

    /*
    3. 
    此时，所有的block已做完循环，
    我们把当前thread计算出的结果（存放在r_c中，尺寸为(Tm, Tn)）写回
    global memory上的C矩阵对应位置中
    */
    
    // 遍历当前thread结果矩阵的每一行
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        // 计算该thread结果矩阵的这一行，在C矩阵上对应的全局row
        int store_c_gmem_m = by * BM + ty * TM + i;

        // 以4个数为1组，遍历该thread结果矩阵的每一列
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            // 计算这4个数中的第一个数在C矩阵上对应的全局col
            int store_c_gmem_n = bx * BN + tx * TN + j;
            // 将这4个数以FLOAT4写回global memory
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

int main(void) {
    printf("\nKernal = sgemm_V1\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = sgemm_V1;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(BN / TN, BM / TM);
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

        dim3 blockDim(BN / TN, BM / TM);
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
