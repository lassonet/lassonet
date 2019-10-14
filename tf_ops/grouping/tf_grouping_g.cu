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



// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_gpu(int b, int n, int m, float radius, 
        int nsample, 
        const float * __restrict__ xyz1, // xyz
        const float * __restrict__ xyz2, // new_xyz
        int * __restrict__ idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    const int index = threadIdx.x;
    const int stride = blockDim.x;
    
    float radius2 = radius * radius;
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        float x2=xyz2[j*3+0];
        float y2=xyz2[j*3+1];
        float z2=xyz2[j*3+2];
        // only pick the FIRST nsample points in the ball
        for (int k=0;k<n && cnt < nsample;++k) {
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
    	    float d2=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d2<radius2) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = k;
                }
                idx[j*nsample+cnt] = k;
                ++cnt;
            }
        }
        pts_cnt[j] = cnt;
    }
}

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)

// input: points (b,n,c), idx (b, m, nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, 
                                int nsample, 
                                const float * __restrict__ points, 
                                const int * __restrict__ idx, 
                                float * __restrict__ out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[(j*nsample + k)*c+l] = points[ii*c+l];
            }
        }
    }
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)

// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, 
                                     int nsample, 
                                     const float *__restrict__ grad_out, 
                                     const int *__restrict__ idx, 
                                     float *__restrict__ grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                 atomicAdd(grad_points + ii*c+l, grad_out[(j*nsample+k)*c+l]);
            }
        }
    }
}

// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
__global__ void selection_sort_gpu(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
void queryBallPointLauncher(int b, int n, int m, float radius, 
        int nsample, const float *xyz1, const float *xyz2, 
        int *idx, int *pts_cnt) {
    query_ball_point_gpu<<<b,opt_n_threads(m)>>>(b,n,m,radius,nsample,xyz1,xyz2,idx,pts_cnt);
    //cudaDeviceSynchronize();
    CUDA_CHECK_ERRORS();
}
void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    selection_sort_gpu<<<b,256>>>(b,n,m,k,dist,outi,out); 
    //cudaDeviceSynchronize();
}
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
    group_point_gpu<<<b,opt_block_config(m, c)>>>(b,n,c,m,nsample,points,idx,out);
    //cudaDeviceSynchronize();
    CUDA_CHECK_ERRORS();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points){
    group_point_grad_gpu<<<b,opt_block_config(m, c)>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //group_point_grad_gpu<<<1,1>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
    CUDA_CHECK_ERRORS();
}
