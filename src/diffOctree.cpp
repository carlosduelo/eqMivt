#include "typedef.h"
#include <iostream>
#include <fstream>
#include <cmath>

int main(int argc, char ** argv)
{
	std::ifstream file1;
	std::ifstream file2;

	try
	{
		file1.open(argv[1], std::ifstream::binary);
	}
	catch(...)
	{
		std::cerr<<"Octree: error opening octree file "<<argv[1]<<std::endl;
		return -1;
	}

	int magicWord;

	file1.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<argv[1]<<std::endl;
		return -1;
	}

	try
	{
		file2.open(argv[2], std::ifstream::binary);
	}
	catch(...)
	{
		std::cerr<<"Octree: error opening octree file "<<argv[2]<<std::endl;
		return -1;
	}

	magicWord = 0;

	file2.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<argv[2]<<std::endl;
		return -1;
	}

	int numOctree[2];

	file1.read((char*)&numOctree[0],sizeof(int));
	file2.read((char*)&numOctree[1],sizeof(int));

	if (numOctree[0] != numOctree[1])
	{
		std::cerr<<"Different number of octrees"<<std::endl;
		return 0;
	}

	int realDimensionV1[3];
	int realDimensionV2[3];

	file1.read((char*)realDimensionV1, 3*sizeof(int));
	file2.read((char*)realDimensionV2, 3*sizeof(int));

	if (realDimensionV1[0] != realDimensionV2[0] ||
		realDimensionV1[1] != realDimensionV2[1] ||
		realDimensionV1[2] != realDimensionV2[2])
	{
		std::cerr<<"Different volume size"<<std::endl;
		return 0;
	}

	for(int j=0; j<3; j++)
	{
		float * grid1 = new float[realDimensionV1[j]];
		float * grid2 = new float[realDimensionV1[j]];
		file1.read((char*)grid1,realDimensionV1[j]*sizeof(float));
		file2.read((char*)grid2,realDimensionV1[j]*sizeof(float));
		for(int i=0; i<realDimensionV1[j]; i++)
		{
			if (grid1[i] != grid2[i])
			{
				std::cerr<<"Error different grids"<<std::endl;
				return 0;
			}
		}
		delete[] grid1;
		delete[] grid2;
	}

	int nLevels1[numOctree[0]];
	int nLevels2[numOctree[0]];
	int maxLevel1[numOctree[0]]; 
	int maxLevel2[numOctree[0]]; 
	int dimension1[numOctree[0]]; 
	int dimension2[numOctree[0]]; 
	int realDim1[3*numOctree[0]];
	int startC1[3*numOctree[0]];
	int finishC1[3*numOctree[0]]; 
	int realDim2[3*numOctree[0]];
	int startC2[3*numOctree[0]];
	int finishC2[3*numOctree[0]]; 

	int rest = 0;
	while(rest < numOctree[0])
	{
		int n = 0;
		int nL = 0;
		int mL = 0;
		int s[3];
		int f[3];
		int d[3];
		file1.read((char*)&n,sizeof(int));
		file1.read((char*)s,3*sizeof(int));
		file1.read((char*)f,3*sizeof(int));
		file1.read((char*)&nL,sizeof(int));
		file1.read((char*)&mL,sizeof(int));
		d[0] = f[0] - s[0];
		d[1] = f[1] - s[1];
		d[2] = f[2] - s[2];
		for(int j=0; j<n; j++)
		{
			nLevels1[rest+j] = nL;
			maxLevel1[rest+j] = mL;
			dimension1[rest+j] = exp2(nL);
			startC1[3*(rest+j)] = s[0];
			startC1[3*(rest+j)+1] = s[1];
			startC1[3*(rest+j)+2] = s[2];
			finishC1[3*(rest+j)] = f[0];
			finishC1[3*(rest+j)+1] = f[1];
			finishC1[3*(rest+j)+2] = f[2];
			realDim1[3*(rest+j)] = d[0];
			realDim1[3*(rest+j)+1] = d[1];
			realDim1[3*(rest+j)+2] = d[2];
		}
		rest += n;
	}
	rest = 0;
	while(rest < numOctree[0])
	{
		int n = 0;
		int nL = 0;
		int mL = 0;
		int s[3];
		int f[3];
		int d[3];
		file2.read((char*)&n,sizeof(int));
		file2.read((char*)s,3*sizeof(int));
		file2.read((char*)f,3*sizeof(int));
		file2.read((char*)&nL,sizeof(int));
		file2.read((char*)&mL,sizeof(int));
		d[0] = f[0] - s[0];
		d[1] = f[1] - s[1];
		d[2] = f[2] - s[2];
		for(int j=0; j<n; j++)
		{
			nLevels2[rest+j] = nL;
			maxLevel2[rest+j] = mL;
			dimension2[rest+j] = exp2(nL);
			startC2[3*(rest+j)] = s[0];
			startC2[3*(rest+j)+1] = s[1];
			startC2[3*(rest+j)+2] = s[2];
			finishC2[3*(rest+j)] = f[0];
			finishC2[3*(rest+j)+1] = f[1];
			finishC2[3*(rest+j)+2] = f[2];
			realDim2[3*(rest+j)] = d[0];
			realDim2[3*(rest+j)+1] = d[1];
			realDim2[3*(rest+j)+2] = d[2];
		}
		rest += n;
	}

	for(int i=0; i<numOctree[0]; i++)
	{
		if (nLevels1[i] != nLevels2[i])
		{
			std::cerr<<"Octree with different nLevels"<<std::endl;
			return 0;
		}
		if (maxLevel1[i] != maxLevel2[i])
		{
			std::cerr<<"Octree with different maxLevels"<<std::endl;
			return 0;
		}
		if (dimension1[i] != dimension2[i])
		{
			std::cerr<<"Octree with different dimension"<<std::endl;
			return 0;
		}
		if (startC1[3*i] != startC2[3*i] || 
			startC1[3*i+1] != startC2[3*i+1] || 
			startC1[3*i+2] != startC2[3*i+2])
		{
			std::cerr<<"Different start coordinate"<<std::endl;
			return 0;
		}
		if (finishC1[3*i] != finishC2[3*i] || 
			finishC1[3*i+1] != finishC2[3*i+1] || 
			finishC1[3*i+2] != finishC2[3*i+2])
		{
			std::cerr<<"Different finish coordinate"<<std::endl;
			return 0;
		}
		if (realDim1[3*i] != realDim2[3*i] || 
			realDim1[3*i+1] != realDim2[3*i+1] || 
			realDim1[3*i+2] != realDim2[3*i+2])
		{
			std::cerr<<"Different real dimension"<<std::endl;
			return 0;
		}

	}

	float isos1[numOctree[0]];
	float isos2[numOctree[0]];
	int desp1[numOctree[0]];
	int desp2[numOctree[0]];
	file1.read((char*)isos1, numOctree[0]*sizeof(float));
	file1.read((char*)desp1, numOctree[0]*sizeof(int));
	file2.read((char*)isos2, numOctree[0]*sizeof(float));
	file2.read((char*)desp2, numOctree[0]*sizeof(int));
	for(int i=0; i<numOctree[0]; i++)
	{
		if (isos1[i] != isos2[i])
		{
			std::cerr<<"Differents isosurfaces"<<std::endl;
			return 0;
		}
		if (desp1[i] != desp2[i])
		{
			std::cerr<<"Possible differences... "<<std::endl;
		}
	}


	for(int i=0; i<numOctree[0]; i++)
	{
		std::cout<<std::endl;
		std::cout<<"Octree "<<i<<std::endl;
		std::cout<<"nLevels "<<nLevels1[i]<<std::endl;
		std::cout<<"maxLevel "<<maxLevel1[i]<<std::endl;
		std::cout<<"Isosurface "<<isos1[i]<<std::endl;
		std::cout<<"Start coordinates "<<startC1[3*i]<<" "<<startC1[3*i+1]<<" "<<startC1[3*i+2]<<std::endl;
		std::cout<<"Finish coordinates "<<finishC1[3*i]<<" "<<startC1[3*i+1]<<" "<<startC1[3*i+2]<<std::endl;
		std::cout<<"Real dimension "<<realDim1[3*i]<<" "<<realDim1[3*i+1]<<" "<<realDim1[3*i+2]<<std::endl;
		std::cout<<"Offset in file "<<desp1[i]<<std::endl;
		std::cout<<"Checking...... ";
		
		int * numCubes1 = new int[maxLevel1[i]+1];
		int * numCubes2= new int[maxLevel1[i]+1];
		int * sizes1 = new int[maxLevel1[i]+1];
		int * sizes2 = new int[maxLevel1[i]+1];
		int   maxH1;
		int   maxH2;

		file1.seekg(desp1[0], std::ios_base::beg);
		for(int d=1; d<=i; d++)
			file1.seekg(desp1[d], std::ios_base::cur);
		file2.seekg(desp2[0], std::ios_base::beg);
		for(int d=1; d<=i; d++)
			file2.seekg(desp2[d], std::ios_base::cur);
		file1.read((char*)&maxH1, sizeof(int));
		file1.read((char*)numCubes1, (maxLevel1[i]+1)*sizeof(int));
		file1.read((char*)sizes1, (maxLevel1[i]+1)*sizeof(int));
		file2.read((char*)&maxH2, sizeof(int));
		file2.read((char*)numCubes2, (maxLevel1[i]+1)*sizeof(int));
		file2.read((char*)sizes2, (maxLevel1[i]+1)*sizeof(int));

		if (maxH1 != maxH2)
			std::cout<<" different max height ";


		int diff = false;

		for(int j=0; j<=maxLevel1[i]; j++)
		{
			//std::cout<<std::endl<<sizes1[j]<<" "<<sizes2[j]<<std::endl;
			if (sizes1[j] != sizes2[j])
			{
				std::cout<<" different sizes"<<std::endl<<" Fail"<<std::endl;
				diff = true;
				break;
			}
			if (numCubes1[j] != numCubes2[j])
			{
				std::cout<<" different num cubes"<<std::endl<<" Fail"<<std::endl;
				diff = true;
				break;
			}
		}

		for(int j=0; !diff && j<=maxLevel1[i]; j++)
		{
			eqMivt::index_node_t * l1 = new eqMivt::index_node_t[sizes1[j]];	
			eqMivt::index_node_t * l2 = new eqMivt::index_node_t[sizes2[j]];	
			file1.read((char*)l1, sizes1[j]*sizeof(eqMivt::index_node_t));
			file2.read((char*)l2, sizes2[j]*sizeof(eqMivt::index_node_t));

			for(int k=0; k<sizes1[j]; k++)
			{
				if (l1[k] != l2[k])
				{
					std::cout<<" different octree nodes "<<l1[k]<<" "<<l2[k]<<std::endl<<" Fail"<<std::endl;
					diff = true;
					break;
				}
			}

			delete[] l1;
			delete[] l2;
		}

		delete[] numCubes1;
		delete[] numCubes2;
		delete[] sizes1;
		delete[] sizes2;
		
		if (!diff)
			std::cout<<" OK"<<std::endl;
		else
			std::cout<<std::endl;
	}

	return 0;
}
