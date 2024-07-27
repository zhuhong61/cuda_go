#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void flash_attention_2_forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float* L,
    float* O
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for L

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int i = 0; i < Tr; i++) { // Q is outer loop

        // Load Qi from HBM to SRAM
        for (int x = 0; x < d; x++) {
            Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
        }
        // Initialize l and m
        float row_m_prev = -INFINITY;
        float row_l_prev = 0;

        for (int j = 0; j < Tc; j++) { // K, V is inner loop
            // Load Kj, Vj from HBM to SRAM
            for (int x = 0; x < d; x++) {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }
            __syncthreads();

            // compute S = QK^T
            // S_i^j = softmax_scale * QiKj^T
            // S_i^j[tx][y] = softmax_scale * Sum_{x = 0}^{d-1} Qi[tx][x] * Kj[y][x]
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++)
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // compute local max
            // m_i^j = max(m_i^j-1, row_max(S_i^j))
            float row_m_new = max(row_m_prev, row_m);


            // compute local sum
            // P_i^j = exp(S_i^j - m_i^j)
            // P_i^j[tx][y] = exp(S_i^j[tx][y] - m_i^j)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m_new);
                row_l += S[(Bc * tx) + y];
            }

            // l_i^j = (exp(m_i^j-1 - m_i^j) * l_i^j-1) + row_sum(P_i^j)
            float row_m_exp = __expf(row_m_prev - row_m_new);
            float row_l_new = (row_m_exp * row_l_prev) + row_l;

            // compute P@V and O
            // O_i^j = diag(exp(m_i^j-1 - m_i^j))^-1 * O_i^j-1 + P_i^jVj
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = \
                    row_m_exp * O[qkv_offset + (tile_size * i) + (tx * d) + x] + pv;
            }

            // Update m and l
            row_m_prev = row_m_new;
            row_l_prev = row_l_new;
        } // end K, V loop

        // O_i = diag(l_i^{Tc})^-1 * O_i^{Tc}
        for (int x = 0; x < d; x++)
            O[qkv_offset + (tile_size * i) + (tx * d) + x] /= row_l_prev;
        // L_i = m_i^{Tc} + log(l_i^{Tc})
        L[lm_offset + (Br * i) + tx] = row_m_prev + __logf(row_l_prev);
    }
}

std::vector<torch::Tensor> flash_attention_2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, L to HBM
    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B, nh, N});
    torch::Device device(torch::kCUDA);
    L = L.to(device);

    // Calculate SRAM size needed per block
    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    const int sram_size =
        (2 * col_tile_size * sizeof(float))  // SRAM size for Kj, Vj
        + (row_tile_size * sizeof(float))  // SRAM size for Qi
        + (Bc * Br * sizeof(float));  // SRAM size for S
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Br);  // Br threads per block

    flash_attention_2_forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        L.data_ptr<float>(), O.data_ptr<float>()
    );
    return {O, L};
}
~

