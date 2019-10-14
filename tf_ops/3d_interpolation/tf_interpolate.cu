#include <stdio.h>
#include <stdlib.h>

#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define TOTAL_THREADS 1024

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int n, int c) {
    const int x_threads = opt_n_threads(n); // max to 512
    const int y_threads = max(min(opt_n_threads(c), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);

    return block_config;
}

#define CUDA_CHECK_ERRORS()                                                    \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (cudaSuccess != err) {                                              \
            fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",     \
                    cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__,    \
                    __FILE__);                                                 \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
__global__ void three_nn_kernel(int b, int n, int m,
    const float *__restrict__ unknown,
    const float *__restrict__ known,
    float *__restrict__ dist2,
    int *__restrict__ idx) {
    int batch_index = blockIdx.x;
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * 3;
    idx += batch_index * n * 3;

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int j = index; j < n; j += stride) {
        float ux = unknown[j * 3 + 0];
        float uy = unknown[j * 3 + 1];
        float uz = unknown[j * 3 + 2];

        double best1 = 1e40, best2 = 1e40, best3 = 1e40;
        int besti1 = 0, besti2 = 0, besti3 = 0;
        for (int k = 0; k < m; ++k) {
            float x = known[k * 3 + 0];
            float y = known[k * 3 + 1];
            float z = known[k * 3 + 2];
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
            if (d < best1) {
                best3 = best2;
                besti3 = besti2;
                best2 = best1;
                besti2 = besti1;
                best1 = d;
                besti1 = k;
            } else if (d < best2) {
                best3 = best2;
                besti3 = besti2;
                best2 = d;
                besti2 = k;
            } else if (d < best3) {
                best3 = d;
                besti3 = k;
            }
        }
        dist2[j * 3 + 0] = best1;
        dist2[j * 3 + 1] = best2;
        dist2[j * 3 + 2] = best3;

        idx[j * 3 + 0] = besti1;
        idx[j * 3 + 1] = besti2;
        idx[j * 3 + 2] = besti3;
    }
}

void threennLauncher(int b, int n, int m, const float *xyz1,
    const float *xyz2, float *dist2, int *idx) {

    three_nn_kernel<<<b, opt_n_threads(n)>>>(b, n, m, xyz1, xyz2, dist2, idx);

    CUDA_CHECK_ERRORS();
}

// input: points (b,m,c), idx (b,n,3), weight (b,n,3)
// output: out (b,n,c)
__global__ void three_interpolate_kernel(int b, int m, int c, int n,
    const float *__restrict__ points,
    const int *__restrict__ idx,
    const float *__restrict__ weight,
    float *__restrict__ out) {
    int batch_index = blockIdx.x;

    points += batch_index * m * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    out += batch_index * n * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    // for each thread
    for (int i = index; i < n * c; i += stride) {
        const int j = i / c; // should be i / c in range [0, n]
        const int l = i % c; // should be i % c in range [0, c]
        // each n of idx and weight
        float w1 = weight[j * 3 + 0];
        float w2 = weight[j * 3 + 1];
        float w3 = weight[j * 3 + 2];

        int i1 = idx[j * 3 + 0];
        int i2 = idx[j * 3 + 1];
        int i3 = idx[j * 3 + 2];

        out[i] = points[i1 * c + l] * w1 +  // should be points[i1*c + l] * w1
                 points[i2 * c + l] * w2 +  // should be points[i2*c + l] * w2
                 points[i3 * c + l] * w3;   // should be points[i3*c + l] * w3
    }
}

void threeinterpolateLauncher(int b, int m, int c, int n, 
    const float *points, const int *idx, 
    const float *weight, float *out) {
    three_interpolate_kernel<<<b, opt_block_config(n, c)>>>(
        b, m, c, n, points, idx, weight, out);
    
    CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, n), idx(b, n, 3), weight(b, n, 3)
// output: grad_points(b, c, m)

// input: grad_out(b, n, c), idx(b, n, 3), weight(b, n, 3)
// output: grad_points(b, m, c)
__global__ void three_interpolate_grad_kernel(int b, int n, int c, int m, 
    const float *__restrict__ grad_out,
    const int *__restrict__ idx, 
    const float *__restrict__ weight,
    float *__restrict__ grad_points) {
    int batch_index = blockIdx.x;
    
    grad_out += batch_index * n * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    grad_points += batch_index * m * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * n; i += stride) {
        const int j = i / c; // should be i / c in range [0, n]
        const int l = i % c; // should be i % c in range [0, c]
        float w1 = weight[j * 3 + 0];
        float w2 = weight[j * 3 + 1];
        float w3 = weight[j * 3 + 2];

        int i1 = idx[j * 3 + 0];
        int i2 = idx[j * 3 + 1];
        int i3 = idx[j * 3 + 2];

        atomicAdd(grad_points + i1 * c + l, grad_out[i] * w1);
        atomicAdd(grad_points + i2 * c + l, grad_out[i] * w2);
        atomicAdd(grad_points + i3 * c + l, grad_out[i] * w3);
    }
}

void threeinterpolateGradLauncher(int b, int n, int c, int m,
    const float *grad_out,
    const int *idx, const float *weight,
    float *grad_points) {
    
    three_interpolate_grad_kernel<<<b, opt_block_config(n, c)>>>(
            b, n, c, m, grad_out, idx, weight, grad_points);

    CUDA_CHECK_ERRORS();
}