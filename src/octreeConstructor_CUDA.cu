/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#define BLOCK_SIZE 128

#include "octreeConstructor_CUDA.h"
#include "mortonCodeUtil.h"
#include "cuda_help.h"
#include <cutil_math.h>

#include "cuda_runtime.h"

#include <iostream>

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

namespace eqMivt
{
	size_t	octreeConstructorGetFreeMemory()
	{
		size_t total = 0;
		size_t free = 0;

		if (cudaSuccess != cudaMemGetInfo(&free, &total))
		{
			std::cerr<<"LRUCache: Error getting memory info"<<std::endl;
			return false;
		}
		
		return free;
	}

	float * octreeConstructorCreateCube(int cubeDim)
	{
		float * cubeGPU = 0;
		size_t size = (cubeDim*cubeDim*cubeDim)*sizeof(float); 

		if (cudaSuccess != (cudaMalloc((void **)&cubeGPU, size)))
		{
			return 0;
		}

		return cubeGPU;
	}

	bool	octreeConstructorCopyCube(float * cubeGPU, float * cube, int cubeDim)
	{
		size_t size = (cubeDim*cubeDim*cubeDim)*sizeof(float); 

		if (cudaSuccess != (cudaMemcpy((void*)cubeGPU, (void*)cube, size, cudaMemcpyHostToDevice)))
		{
			if (cubeGPU != 0)
				cudaFree(cubeGPU);
			return false;
		}
		return true;
	}
	bool	octreeConstructorCopyCube3D(float * cubeGPU, float * cube, int cubeDimC, int cubeDimG, int xoffset, int yoffset, int zoffset)
	{
			cudaMemcpy3DParms paramsG = {0};
			paramsG.srcPtr = make_cudaPitchedPtr((void*)cube, cubeDimC*sizeof(float), cubeDimC, cubeDimC);
			paramsG.dstPtr = make_cudaPitchedPtr((void*)cubeGPU, cubeDimG*sizeof(float), cubeDimG, cubeDimG);
			paramsG.extent =  make_cudaExtent(cubeDimG*sizeof(float), cubeDimG, cubeDimG);
			paramsG.srcPos = make_cudaPos(zoffset*sizeof(float), yoffset, xoffset);
			paramsG.dstPos = make_cudaPos(0,0,0);
			paramsG.kind =  cudaMemcpyHostToDevice;

			if (cudaSuccess != cudaMemcpy3D(&paramsG))
			{
				if (cubeGPU != 0)
					cudaFree(cubeGPU);
				return false;
			}
		return true;
	}

	index_node_t * octreeConstructorCreateResult(int size)
	{
		index_node_t * result  = 0;
		size_t sizeR = (size)*sizeof(index_node_t); 

		if (cudaSuccess != (cudaMalloc((void **)&result, sizeR)))
		{
			return 0;
		}

		return result;
	}

	bool	octreeConstructorCopyResult(index_node_t * cpuResult, index_node_t * gpuResult, int size)
	{
		size_t sizeR = (size)*sizeof(index_node_t); 

		if (cudaSuccess != (cudaMemcpy((void*)cpuResult, (void*)gpuResult, sizeR, cudaMemcpyDeviceToHost)))
		{
			return false;
		}
		return true;
	}

	void	octreeConstructorDestroyCube(float * cube)
	{
		if (cube != 0)
			cudaFree(cube);
	}
	void	octreeConstructorDestroyResult(index_node_t * result)
	{
		if (result != 0)
			cudaFree(result);
	}

	__device__ bool cuda_checkIsosurface(int x, int y, int z, int dim, float * cube, float isosurface)
	{
		bool sign = (cube[posToIndex(x, y, z, dim)] - isosurface) < 0;

		if (((cube[posToIndex(x, y, z+1, dim)] - isosurface) < 0) != sign)
			return true;
		if (((cube[posToIndex(x, y+1, z, dim)] - isosurface) < 0) != sign)
			return true;
		if (((cube[posToIndex(x, y+1, z+1, dim)] - isosurface) < 0) != sign)
			return true;
		if (((cube[posToIndex(x+1, y, z, dim)] - isosurface) < 0) != sign)
			return true;
		if (((cube[posToIndex(x+1, y, z+1, dim)] - isosurface) < 0) != sign)
			return true;
		if (((cube[posToIndex(x+1, y+1, z, dim)] - isosurface) < 0) != sign)
			return true;
		if (((cube[posToIndex(x+1, y+1, z+1, dim)] - isosurface) < 0) != sign)
			return true;

		return false;
	}

	__global__ void	cuda_octreeConstructorComputeCube(index_node_t * cubes, int size, index_node_t startID, float iso, float * cube, int nodeLevel, int nLevels, int dimNode, int cubeDim, int3 coorCubeStart)
	{
		unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < size)
		{
			index_node_t id = startID + tid;

			int3 coorNodeStart = getMinBoxIndex2(id, nodeLevel, nLevels);
			int3 coorNodeFinish= coorNodeStart + dimNode - 1;
			coorNodeStart = coorNodeStart-coorCubeStart;
			coorNodeFinish = coorNodeFinish-coorCubeStart;

			for(int x=coorNodeStart.x; x<=coorNodeFinish.x; x++)
			{
				for(int y=coorNodeStart.y; y<=coorNodeFinish.y; y++)
				{
					for(int z=coorNodeStart.z; z<=coorNodeFinish.z; z++)
					{	
							if ( cuda_checkIsosurface(x, y, z, cubeDim, cube, iso))
							{
								cubes[tid] = id;
								return;
							}
					}
				}
			}

			cubes[tid] = 0;
			return;
		}
	}

	void	octreeConstructorComputeCube(index_node_t * cubes, int size, index_node_t startID, float iso, float * cube, int nodeLevel, int nLevels, int dimNode, int cubeDim, int  coorCubeStart[3])
	{
		dim3 threads = getThreads(size);
		dim3 blocks = getBlocks(size);
		cuda_octreeConstructorComputeCube<<<blocks, threads>>>(cubes, size, startID, iso, cube, nodeLevel, nLevels, dimNode, cubeDim, make_int3(coorCubeStart[0], coorCubeStart[1], coorCubeStart[2]));
		return;
	}

}
