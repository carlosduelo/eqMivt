/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "ray-casting-sphere.h" 
#include <cuda_runtime.h>    
#include <cutil_math.h>
#include <iostream>

namespace eqMivt
{

#define BLOCK_SIZE 128

inline dim3 getBlocks(int dim)
{
	if (dim <= BLOCK_SIZE)
	{
		dim3 blocks(1);//,0,0);
		return blocks;
	}
	else// if (dim<=(BLOCK_SIZE*BLOCK_SIZE))
	{
		int numBlocks = dim / BLOCK_SIZE;
		if (dim % BLOCK_SIZE !=0) numBlocks++;
		int bpA = sqrt(numBlocks);
		int bp  = floorf(bpA) + 1;
		dim3 blocks(bp,bp);//,0); 
		return blocks;
	}
}

inline dim3 getThreads(int dim)
{
	int t = 32;
	while(dim>t && t<BLOCK_SIZE)
	{
		t+=32;
	}

	dim3 threads(t);//,0,0);
	return threads;
}

inline __device__ bool intersection(float3 ray, float3 posR, float r, float * t)
{
	float3 posE = make_float3(0.0f, 0.0f, 0.0f);
	float3 d = posR -posE;
	//Compute A, B and C coefficients
	float a = dot(ray, ray);
	float b = 2 * dot(ray, d);
	float c = dot(d,d) - (r * r);

	//Find discriminant
	float disc = b * b - 4 * a * c;

	// if discriminant is negative there are no real roots, so return 
	// false as ray misses sphere
	if (disc < 0)
		return false;

	// compute q as described above
	float distSqrt = sqrtf(disc);
	float q;
	if (b < 0)
		q = (-b - distSqrt);
	else
		q = (-b + distSqrt);

	// compute t0 and t1
	float t0 = q / (2.0f*a);
	float t1 = q / (2.0f*a);

	// make sure t0 is smaller than t1
	if (t0 > t1)
	{
		// if t0 is bigger than t1 swap them around
		float temp = t0;
		t0 = t1;
		t1 = temp;
	}

	// if t1 is less than zero, the object is in the ray's negative direction
	// and consequently the ray misses the sphere
	if (t1 < 0)
		return false;

	// if t0 is less than zero, the intersection point is at t1
	if (t0 < 0)
	{
		*t = t1;
		return true;
	}
	// else the intersection point is at t0
	else
	{
		*t = t0;
		return true;
	}
}

//__global__ void kernel_render_sphere(float3 pos, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH)
__global__ void kernel_render_sphere(float * buffer, int pvpW, int pvpH, float3 pos, float3 LB, float3 up, float3 right, float w, float h, float2 jitter)
{
    unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < (pvpW*pvpH))
    {
    	int i = tid % pvpW;
		int j = tid / pvpW;

		float3 ray = LB - pos;
		ray += (j*h*up+jitter.y) + (i*w+jitter.x)*right;
		ray = normalize(ray);

		float hit = 100.0f;
		if (intersection(ray, pos, 1.0f , &hit))
		{
			float3 ph = pos + hit * ray;
			float3 n = ph;
			n = normalize(n);
			float3 l = make_float3(pos.x - ph.x, pos.y - ph.y, pos.z - ph.z);
			l = normalize(l);
			float3 k = cross(n,l);
			float dif = fabs(n.x*l.x + n.y*l.y + n.z*l.z);
			buffer[3*tid] = dif*0.3f; 
			buffer[3*tid+1] = dif*0.5f;
			buffer[3*tid+2] = dif*0.3f;
		}
    }
}


	void render_sphere(GLuint pbo, int pvpW, int pvpH, float posx, float posy, float posz,  float LBx, float LBy, float LBz, float upx, float upy, float upz, float rightx, float righty, float rightz, float w, float h, float jitterX, float jitterY)
{
    struct cudaGraphicsResource *cuda_pbo_resource;
    if (cudaSuccess != cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard))
    {
    	std::cerr<<"Error cudaGraphicsGLRegisterBuffer"<<std::endl;
    }
    // map PBO to get CUDA device pointer
    float  *d_output;
    // map PBO to get CUDA device pointer
    if (cudaSuccess != cudaGraphicsMapResources(1, &cuda_pbo_resource, 0))
    {
    	std::cerr<<"Error cudaGraphicsMapResources"<<std::endl;
    }
    size_t num_bytes;
    if (cudaSuccess != cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsResourceGetMappedPointer"<<std::endl;
    }
    std::cout<<"CUDA MAPPED "<<num_bytes<<std::endl;
    if (cudaSuccess != cudaMemset(d_output, (int)1.0f, num_bytes))
    {
    	std::cerr<<"Error cudaMemSet"<<std::endl;
    }

    cudaStream_t stream;
    if (cudaSuccess != cudaStreamCreate(&stream))
    {
	    std::cerr<<"Error cudaStreamCreate"<<std::endl;
    }

    dim3 threads = getThreads(pvpW*pvpH);
    dim3 blocks = getBlocks(pvpW*pvpH);
    kernel_render_sphere<<<blocks, threads, 0, stream>>>(d_output, pvpW, pvpH, make_float3(posx,posy,posz), make_float3(LBx, LBy, LBz), make_float3(upx,upy,upz),make_float3(rightx,righty,rightz), w, h, make_float2(jitterX, jitterY));

    if (cudaSuccess !=  cudaStreamSynchronize(stream))
    {
	    std::cerr<<"Error cudaSreamSynchronize"<<std::endl;
    }

    std::cerr<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;

    if (cudaSuccess != cudaStreamDestroy(stream))
    {
	    std::cerr<<"Error cudaStreamDestroy"<<std::endl;
    }

    if (cudaSuccess != cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0))
    {
    	std::cerr<<"Error cudaGraphicsUnmapResources"<<std::endl;
    }
    if (cudaSuccess != cudaGraphicsUnregisterResource(cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsUnregisterResource"<<std::endl;
    }
}
}
