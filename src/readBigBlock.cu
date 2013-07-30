#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>

#include "mortonCodeUtil_CPU.h"

#include "cuda_runtime.h"

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

int main(int argc, char ** argv)
{


	int nLevels = atoi(argv[1]);
	int levelC = atoi(argv[2]);
	int levelG = atoi(argv[3]);

	int dim		= exp2((float)nLevels);
	int dimC	= exp2((float)nLevels - levelC); 
	int dimG	= exp2((float)nLevels - levelG); 

	printf("nLevels %d dimension %d\nLevel cube CPU %d dimension %d\nLevel cube GPU %d dimension %d\n\n",nLevels, dim, levelC, dimC, levelG, dimG);

	float * volume = 0;
	float * cubeCPU = 0;
	float * cubeGPU = 0;

	fprintf(stdout, "Allocating %f MB volume\n", (dim*dim*dim*sizeof(float))/1024.0f/1024.0f);
	if (cudaSuccess != cudaHostAlloc((void**)&volume, dim*dim*dim*sizeof(float), cudaHostAllocDefault))
	{
		fprintf(stderr, "Error allocating memory volume %ld \n", dim*dim*dim*sizeof(float));
		return 0;
	}
	fprintf(stdout, "Allocating %f MB cube cpu\n", (dimC*dimC*dimC*sizeof(float))/1024.0f/1024.0f);
	if (cudaSuccess != cudaHostAlloc((void**)&cubeCPU, dimC*dimC*dimC*sizeof(float), cudaHostAllocDefault))
	{
		fprintf(stderr, "Error allocating memory cube cpu %ld \n", dimC*dimC*dimC*sizeof(float));
		return 0;
	}
	fprintf(stdout, "Allocating %f MB cube gpu\n", (dimG*dimG*dimG*sizeof(float))/1024.0f/1024.0f);
	if (cudaSuccess != cudaHostAlloc((void**)&cubeGPU, dimG*dimG*dimG*sizeof(float), cudaHostAllocDefault))
	{
		fprintf(stderr, "Error allocating memory cube GPU %ld \n", dimG*dimG*dimG*sizeof(float));
		return 0;
	}

	for(int i=0; i<dim; i++)
		for(int j=0; j<dim; j++)
			for(int k=0; k<dim; k++)
				volume[posToIndex(i,j,k,dim)] = (float)posToIndex(i,j,k,dim);


	std::cout<<std::endl;
	
	eqMivt::index_node_t startC		= eqMivt::coordinateToIndex(vmml::vector<3, int>(0,0,0), levelC, nLevels);
	eqMivt::index_node_t finishC	= eqMivt::coordinateToIndex(vmml::vector<3, int>(dim-1, dim-1, dim-1), levelC, nLevels);

	std::cout<<"Index cube cpu "<<startC<<" "<<finishC<<std::endl;
	for(eqMivt::index_node_t id=startC; id<=finishC; id++)
	{
		vmml::vector<3, int> start = eqMivt::getMinBoxIndex2(id, levelC, nLevels); 
		vmml::vector<3, int>  end = start + vmml::vector<3, int> (dimC, dimC, dimC);
		std::cout<<"Copying volume to cpu"<<start<<" to "<<end<<std::endl;

		cudaMemcpy3DParms params = {0};
		params.srcPtr = make_cudaPitchedPtr((void*)(volume), dim*sizeof(float), dim, dim);
		params.dstPtr = make_cudaPitchedPtr((void*)cubeCPU, dimC*sizeof(float), dimC, dimC);
		params.extent =  make_cudaExtent(dimC*sizeof(float), dimC, dimC);
		params.srcPos = make_cudaPos(start.z()*sizeof(float), start.y(), start.x());
		params.dstPos = make_cudaPos(0,0,0);
		params.kind =  cudaMemcpyHostToHost;

		if (cudaSuccess != cudaMemcpy3D(&params))
		{
			fprintf(stderr, "Error cudaMemcpy3D: %s \n",cudaGetErrorString(cudaGetLastError()));
			return 0;
		}

		for(int i=0; i<dimC; i++)
			for(int j=0; j<dimC; j++)
				for(int k=0; k<dimC; k++)
				{
					if (cubeCPU[posToIndex(i,j,k,dimC)] != volume[posToIndex(start.x()+i, start.y()+j, start.z()+k, dim)])
					{
						fprintf(stderr, "Error (%d,%d,%d) != (%d,%d,%d) ==> ",i,j,k, start.x()+i, start.y()+j, start.z()+k);
						std::cerr<<cubeCPU[posToIndex(i,j,k,dimC)]<<" != "<<volume[posToIndex(start.x()+i, start.y()+j, start.z()+k, dim)]<<std::endl; 
						return 0;
					}
				}
		
		eqMivt::index_node_t startG		= eqMivt::coordinateToIndex(start, levelG, nLevels);
		eqMivt::index_node_t finishG	= eqMivt::coordinateToIndex(start+vmml::vector<3, int>(dimC-1, dimC-1, dimC-1), levelG, nLevels);
		std::cout<<"\tIndex cube gpu "<<startG<<" "<<finishG<<std::endl;

		for(eqMivt::index_node_t idG = startG; idG<=finishG; idG++)
		{
			vmml::vector<3, int> startG = eqMivt::getMinBoxIndex2(idG, levelG, nLevels); 
			vmml::vector<3, int>  endG = startG + vmml::vector<3, int> (dimG, dimG, dimG);
			std::cout<<"\tCopying volume to gpu"<<startG<<" to "<<endG<<std::endl;
			
			cudaMemcpy3DParms paramsG = {0};
			paramsG.srcPtr = make_cudaPitchedPtr((void*)cubeCPU, dimC*sizeof(float), dimC, dimC);
			paramsG.dstPtr = make_cudaPitchedPtr((void*)cubeGPU, dimG*sizeof(float), dimG, dimG);
			paramsG.extent =  make_cudaExtent(dimG*sizeof(float), dimG, dimG);
			paramsG.srcPos = make_cudaPos((startG.z()-start.z())*sizeof(float), startG.y()-start.y(), startG.x()-start.x());
			paramsG.dstPos = make_cudaPos(0,0,0);
			paramsG.kind =  cudaMemcpyHostToHost;

			if (cudaSuccess != cudaMemcpy3D(&paramsG))
			{
				fprintf(stderr, "Error cudaMemcpy3D: %s \n",cudaGetErrorString(cudaGetLastError()));
				return 0;
			}
			for(int i=0; i<dimG; i++)
				for(int j=0; j<dimG; j++)
					for(int k=0; k<dimG; k++)
					{
						if (cubeGPU[posToIndex(i,j,k,dimG)] != volume[posToIndex(startG.x()+i, startG.y()+j, startG.z()+k, dim)])
						{
							fprintf(stderr, "Error (%d,%d,%d) != (%d,%d,%d) ==> ",i,j,k, startG.x()+i, startG.y()+j, startG.z()+k);
							std::cerr<<cubeCPU[posToIndex(i,j,k,dimC)]<<" != "<<volume[posToIndex(startG.x()+i, startG.y()+j, startG.z()+k, dim)]<<std::endl; 
							return 0;
						}
					}
		}

		std::cout<<std::endl;
	}

	if (cudaSuccess != cudaFreeHost((void*)volume))
	{
		fprintf(stderr, "Error free memory volume\n");
		return 0;
	}
	if (cudaSuccess != cudaFreeHost((void*)cubeCPU))
	{
		fprintf(stderr, "Error free memory cube cpu\n");
		return 0;
	}
	if (cudaSuccess != cudaFreeHost((void*)cubeGPU))
	{
		fprintf(stderr, "Error free memory cube cpu\n");
		return 0;
	}

	return 0;
}
