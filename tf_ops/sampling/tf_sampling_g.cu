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


__global__ void cumsumKernel(int b,int n,const float * __restrict__ inp,float * __restrict__ out){
  const int BlockSize=2048;
  const int paddingLevel=5;
  __shared__ float buffer4[BlockSize*4];
  __shared__ float buffer[BlockSize+(BlockSize>>paddingLevel)];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    float runningsum=0,runningsum2=0;
    for (int j=0;j<n;j+=BlockSize*4){
      int n24_i=min(n-j,BlockSize*4);
      int n24=(n24_i+3)&~3;
      int n2=n24>>2;
      for (int k=threadIdx.x*4;k<n24_i;k+=blockDim.x*4){
        if (k+3<n24_i){
          float v1=inp[i*n+j+k];
          float v2=inp[i*n+j+k+1];
          v2+=v1;
          float v3=inp[i*n+j+k+2];
          float v4=inp[i*n+j+k+3];
          v4+=v3;
          v3+=v2;
          v4+=v2;
          buffer4[k]=v1;
          buffer4[k+1]=v2;
          buffer4[k+2]=v3;
          buffer4[k+3]=v4;
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v4;
        }else{
          float v=0;
          for (int k2=k;k2<n24_i;k2++){
            v+=inp[i*n+j+k2];
            buffer4[k2]=v;
          }
          for (int k2=n24_i;k2<n24;k2++){
            buffer4[k2]=v;
          }
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v;
        }
      }
      int u=0;
      for (;(2<<u)<=n2;u++){
        __syncthreads();
        for (int k=threadIdx.x;k<int(n2>>(u+1));k+=blockDim.x){
          int i1=(((k<<1)+2)<<u)-1;
          int i2=(((k<<1)+1)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      u--;
      for (;u>=0;u--){
        __syncthreads();
        for (int k=threadIdx.x;k<int((n2-(1<<u))>>(u+1));k+=blockDim.x){
          int i1=(((k<<1)+3)<<u)-1;
          int i2=(((k<<1)+2)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      __syncthreads();
      for (int k=threadIdx.x*4;k<n24;k+=blockDim.x*4){
        if (k!=0){
          int k2=((k>>2)-1)+(((k>>2)-1)>>paddingLevel);
          buffer4[k]+=buffer[k2];
          buffer4[k+1]+=buffer[k2];
          buffer4[k+2]+=buffer[k2];
          buffer4[k+3]+=buffer[k2];
        }
      }
      __syncthreads();
      for (int k=threadIdx.x;k<n24_i;k+=blockDim.x){
        out[i*n+j+k]=buffer4[k]+runningsum;
      }
      float t=buffer[(n2-1)+((n2-1)>>paddingLevel)]+runningsum2;
      float r2=runningsum+t;
      runningsum2=t-(r2-runningsum);
      runningsum=r2;
      __syncthreads();
    }
  }
}

__global__ void binarysearchKernel(int b,int n,int m,const float * __restrict__ dataset,const float * __restrict__ query, int * __restrict__ result){
  int base=1;
  while (base<n)
    base<<=1;
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      float q=query[i*m+j]*dataset[i*n+n-1];
      int r=n-1;
      for (int k=base;k>=1;k>>=1)
        if (r>=k && dataset[i*n+r-k]>=q)
          r-=k;
      result[i*m+j]=r;
    }
  }
}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
  int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void farthestpointsamplingKernel(int b,int n,int m,
    const float * __restrict__ dataset,
    float * __restrict__ temp, int * __restrict__ idxs){
      if (m <= 0)
        return;
      __shared__ float dists[block_size];
      __shared__ int dists_i[block_size];
    
      int batch_index = blockIdx.x;
      dataset += batch_index * n * 3;
      temp += batch_index * n;
      idxs += batch_index * m;
  
      const int tid = threadIdx.x;
      const int stride = block_size;
  
      int old = 0;
      if (threadIdx.x == 0)
        idxs[0] = old;
    
      __syncthreads();
      for (int j = 1; j < m; j++) {
        int besti = 0;
        float best = -1;
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];
        for (int k = tid; k < n; k += stride) {
            float x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            if (mag <= 1e-3)
              continue;
      
            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) +
                (z2 - z1) * (z2 - z1);
      
            float d2 = min(d, temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();
    
        if (block_size >= 512) {
            if (tid < 256) {
              __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
              __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
              __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
              __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
              __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
              __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
              __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
              __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
      if (block_size >= 2) {
          if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
          }
          __syncthreads();
      }
    
      old = dists_i[0];
      if (tid == 0)
          idxs[j] = old;
    }
}

// input: points(b, m, 3) idx(b, m)
// output: out(b, n, 3)
__global__ void gatherpointKernel(int b,int n,int m,
                                  const float * __restrict__ inp,
                                  const int * __restrict__ idx,
                                  float * __restrict__ out){
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=tid;j<m;j+=stride){
      int a=idx[i*m+j];
      out[(i*m+j)*3+0]=inp[(i*n+a)*3+0];
      out[(i*m+j)*3+1]=inp[(i*n+a)*3+1];
      out[(i*m+j)*3+2]=inp[(i*n+a)*3+2];
    }
  }
}

// input: out_g(b, m, 3) idx(b, m)
// output: inp_g(b, n, 3)
__global__ void scatteraddpointKernel(int b,int n,int m,
                                      const float * __restrict__ out_g,
                                      const int * __restrict__ idx,
                                      float * __restrict__ inp_g){
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=tid;j<m;j+=stride){
      int a=idx[i*m+j];
      atomicAdd(inp_g + (i*n+a)*3+0, out_g[(i*m+j)*3+0]);
      atomicAdd(inp_g + (i*n+a)*3+1, out_g[(i*m+j)*3+1]);
      atomicAdd(inp_g + (i*n+a)*3+2, out_g[(i*m+j)*3+2]);
    }
  }
}

void cumsumLauncher(int b,int n,const float * inp,float * out){
  cumsumKernel<<<32,512>>>(b,n,inp,out);
}
//require b*n working space
void probsampleLauncher(int b,int n,int m,const float * inp_p,const float * inp_r,float * temp,int * out){
  cumsumKernel<<<32,512>>>(b,n,inp_p,temp);
  binarysearchKernel<<<dim3(32,8,1),512>>>(b,n,m,temp,inp_r,out);
}
//require 32*n working space
void farthestpointsamplingLauncher(int b,int n,int m,
  const float * dataset, float * temp, int * idxs) {
  // farthestpointsamplingKernel<<<32,512>>>(b,n,m,inp,temp,out);
    unsigned int n_threads = opt_n_threads(n);
    switch (n_threads) {
    case 1024: farthestpointsamplingKernel<1024>
	    <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
	    break;
    case 512: farthestpointsamplingKernel<512>
	    <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
	    break;
    case 256:
	    farthestpointsamplingKernel<256>
	      <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
	    break;
    case 128:
      farthestpointsamplingKernel<128>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
	    break;
    case 64:
      farthestpointsamplingKernel<64>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      farthestpointsamplingKernel<32>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      farthestpointsamplingKernel<16>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      farthestpointsamplingKernel<8>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      farthestpointsamplingKernel<4>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      farthestpointsamplingKernel<2>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 1:
      farthestpointsamplingKernel<1>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      farthestpointsamplingKernel<512>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
    }

    CUDA_CHECK_ERRORS();
}

void gatherpointLauncher(int b,int n,int m,
                         const float * inp,const int * idx,
                         float * out){
  gatherpointKernel<<<b, opt_n_threads(m)>>>(b,n,m,inp,idx,out);
  CUDA_CHECK_ERRORS();
}

void scatteraddpointLauncher(int b,int n,int m,
                             const float * out_g,
                             const int * idx,
                             float * inp_g){
  scatteraddpointKernel<<<b, opt_n_threads(m)>>>(b,n,m,out_g,idx,inp_g);
}

