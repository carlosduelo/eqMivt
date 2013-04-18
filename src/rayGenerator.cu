/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "rayGenerator.h"
#include "cuda_help.h"
#include "cutil_math.h"

namespace eqMivt
{

__global__ void cuda_generateRays(float * rays, float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH)
{
	int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	int numElements = (pvpW*pvpH);

	if (tid < numElements)
	{
		float3 	ray = LB - origin;
    	int is = tid % pvpW;
		int js = tid / pvpW;
		ray += js*h*up + is*w*right;
		ray = normalize(ray);

		rays[tid] = ray.x;
		rays[numElements + tid] = ray.y;
		rays[2*numElements + tid] = ray.z;
	}
}

void generateRays_CUDA(float * rays, float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, cudaStream_t stream)
{
	dim3 threads = getThreads(pvpW*pvpH);
	dim3 blocks = getBlocks(pvpW*pvpH);

	cuda_generateRays<<<blocks, threads, 0, stream>>>(rays, origin, LB, up, right, w, h, pvpW, pvpH);

}

}
