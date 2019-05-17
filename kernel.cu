#include "kernel.h"
#define TX 32
#define TY 32
#define RAD 1

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
float clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
int idxClip(int idx, int idxMax) {
return idx > (idxMax-1) ? (idxMax-1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int width, int height) {
return idxClip(col, width) + idxClip(row, height)*width;
}

__global__
void sharpenKernel(float4 *d_out, const float4 *d_in,const float *d_filter, int w, int h) {

const int c = threadIdx.x + blockDim.x * blockIdx.x;
const int r = threadIdx.y + blockDim.y * blockIdx.y;

if ((c >= w) || (r >= h)) return;
  const int i = flatten(c, r, w, h);
  const int fltSz = 2*RAD + 1;
  float rgb[3] = {0.f, 0.f, 0.f};

for (int rd = -RAD; rd <= RAD; ++rd) {
  for (int cd = -RAD; cd <= RAD; ++cd) {
    int imgIdx = flatten(c + cd, r + rd, w, h);
    int fltIdx = flatten(RAD + cd, RAD + rd, fltSz, fltSz);
    float4 color = d_in[imgIdx];
    float weight = d_filter[fltIdx];
    rgb[0] += weight*color.x;
    rgb[1] += weight*color.y;
    rgb[2] += weight*color.z;
  }
}

d_out[i].x = clip(rgb[0]);
d_out[i].y = clip(rgb[1]);
d_out[i].z = clip(rgb[2]);

}

void sharpenParallel(float4 *arr, int w, int h) {

const int fltSz = 2 * RAD + 1;

const float filter[9] = {-0.5, 1.0, 0.5,
			1.0, -4.0, 1.0,
			0.5, 1.0, -0.5};

float4 *d_in = 0, *d_out = 0;
float *d_filter = 0;

cudaMalloc(&d_in, w*h*sizeof(float4));

cudaMemcpy(d_in, arr, w*h*sizeof(float4), cudaMemcpyHostToDevice);

cudaMalloc(&d_out, w*h*sizeof(float4));

cudaMalloc(&d_filter, fltSz*fltSz*sizeof(float));

cudaMemcpy(d_filter, filter, fltSz*fltSz*sizeof(float),cudaMemcpyHostToDevice);

const dim3 blockSize(TX, TY);
const dim3 gridSize(divUp(w, blockSize.x), divUp(h, blockSize.y));

sharpenKernel<<<gridSize, blockSize>>>(d_out, d_in, d_filter, w, h);

cudaMemcpy(arr, d_out, w*h*sizeof(float4), cudaMemcpyDeviceToHost);

cudaFree(d_in);
cudaFree(d_out);
cudaFree(d_filter);

}














