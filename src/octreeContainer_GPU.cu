/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <octreeContainer_GPU.h>

#include <sstream>

namespace eqMivt
{

__global__ void insertOctreePointers(index_node_t ** octreeGPU, int * sizes, index_node_t * memoryGPU)
{
	int offset = 0;
	for(int i=0;i<threadIdx.x; i++)
		offset+=sizes[i];

	octreeGPU[threadIdx.x] = &memoryGPU[offset];
}



bool Create_OctreeContainer(int device, index_node_t ** octreeCPU, int * sizesCPU, int maxLevel, index_node_t *** octree, index_node_t ** memoryGPU, int ** sizes, std::string * result)
{
	std::stringstream ss (std::stringstream::in | std::stringstream::out);

	ss<< "Setting device "<<device<<" "<<std::endl;;
	if (cudaSuccess != (cudaSetDevice(device)))
	{
		ss<< "Error setting device\n";
		*result = ss.str();
		return false;
	}


	int total = 0;
	for(int i=0; i<=maxLevel; i++)
		total+=sizesCPU[i];

	ss<< "Allocating memory octree CUDA octree ";
	ss<<(maxLevel+1)*sizeof(index_node_t*)/1024.0f/1024.0f;
	ss<< " MB: \n";
	if (cudaSuccess != (cudaMalloc(octree, (maxLevel+1)*sizeof(index_node_t*))))
	{
		ss<< "Octree: error allocating octree in the gpu\n";
		*result = ss.str();
		return false;
	}

	ss<< "Allocating memory octree CUDA memory ";
	ss<< total*sizeof(index_node_t)/1024.0f/1024.0f;
	ss<< " MB:\n";
	if (cudaSuccess != (cudaMalloc(memoryGPU, total*sizeof(index_node_t))))
	{
		ss<< "Octree: error allocating octree in the gpu\n";
		*result = ss.str();
		return false;
	}
	ss<< "Allocating memory octree CUDA sizes ";
	ss<< (maxLevel+1)*sizeof(int)/1024.0f/1024.0f;
	ss<< " MB:\n";
	if (cudaSuccess != (cudaMalloc(sizes,   (maxLevel+1)*sizeof(int))))
	{
		ss<< "Octree: error allocating octree in the gpu\n";
		*result = ss.str();
		return false;
	}

	/* Compiando sizes */
	ss<< "Octree: coping to device the sizes ";
	if (cudaSuccess != (cudaMemcpy((void*)*sizes, (void*)sizesCPU, (maxLevel+1)*sizeof(int), cudaMemcpyHostToDevice)))
	{
		ss<< "Fail\n";
		*result = ss.str();
		return false;
	}
	else
		ss<< "OK\n";

	/* end sizes */

	/* Copying octree */

	int offset = 0;
	for(int i=0; i<=maxLevel; i++)
	{
		ss<< "Coping to device level ";
		ss<< i;
		ss<<": ";
		if (cudaSuccess != (cudaMemcpy((void*)((*memoryGPU)+offset), (void*)octreeCPU[i], sizesCPU[i]*sizeof(index_node_t), cudaMemcpyHostToDevice)))
		{
			ss<<"Fail\n";
			*result = ss.str();
			return false;
		}
		else
			ss<< "OK\n";

		offset+=sizesCPU[i];
	}

	dim3 blocks(1);
	dim3 threads(maxLevel+1);

	insertOctreePointers<<<blocks,threads>>>(*octree, *sizes,*memoryGPU);
	//      (*result)<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;
	ss<< "Octree: sorting pointers ";
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		ss<< "Fail\n";
		*result = ss.str();
		return false;
	}
	else
		ss<< "OK\n";

	ss<< "End copying octree to GPU\n";
	*result = ss.str();

	return true;
}


bool Destroy_OctreeContainer(int device, index_node_t ** octree, index_node_t * memoryGPU, int * sizes)
{
	if (cudaSuccess != (cudaSetDevice(device)))
	{
		std::cerr<< "Error setting device "<<device<<std::endl;
		return false;
	}

	cudaFree(memoryGPU);
	cudaFree(octree);
	cudaFree(sizes);

	return true;
}

}
