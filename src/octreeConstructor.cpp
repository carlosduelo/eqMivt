/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "octreeConstructor.h"

#include "fileFactory.h"

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

namespace eqMivt
{
	class octree
	{
		private: 
			std::vector<index_node_t> *	_octree;
			int		*					_numCubes;
			int							_maxLevel;
			int							_nLevels;

			void _addElement(index_node_t id, int level)
			{
				int size = _octree[level].size();

				// Firts
				if (size == 0)
				{
					_numCubes[level] = 1;
					_octree[level].push_back(id);
					_octree[level].push_back(id);
				}
				else if (_octree[level].back() == (id - (index_node_t)1))
				{
					_numCubes[level] += 1;
					_octree[level][size-1] = id;
				}
				else if(_octree[level].back() == id)
				{
					//std::cout<<"repetido in level "<<level<<" "<< id <<std::endl;
				}
				else if(_octree[level].back() > id)
				{
					std::cout<<"=======>   ERROR: insert index in order "<< id <<" (in level "<<level<<") last inserted "<<_octree[level].back()<<std::endl;
					throw;
				}
				else
				{
					_numCubes[level] += 1;
					_octree[level].push_back(id);
					_octree[level].push_back(id);
				}
			}

		public:
			octree(int nLevels, int maxLevel)
			{

				_octree		= new std::vector<index_node_t>[maxLevel + 1];
				_numCubes	= new int[maxLevel + 1];	

				_maxLevel = maxLevel;
				_nLevels = nLevels;
			}

			~octree()
			{
				delete[] _octree;
				delete[] _numCubes;
			}

			void addVoxel(index_node_t id)
			{
				for(int i=_maxLevel; i>=0; i--)
				{
					_addElement(id, i);
					id >>= 3;
				}

			}

			void printTree()
			{
				for(int i=_maxLevel; i>=0; i--)
				{
					std::vector<index_node_t>::iterator it;

					std::cout<<"Level: "<<i<<std::endl;
					for ( it=_octree[i].begin() ; it != _octree[i].end(); it++ )
					{
						std::cout << "From " << *it;
						it++;
						std::cout << " to " << *it<<std::endl;
					}
				}
			}
	};

	inline index_node_t dilateInteger(index_node_t x)
	{
		x = (x | (x << 20)) & 0x000001FFC00003FF;
		x = (x | (x << 10)) & 0x0007E007C00F801F;
		x = (x | (x << 4))  & 0x00786070C0E181C3;
		x = (x | (x << 2))  & 0x0199219243248649;
		x = (x | (x << 2))  & 0x0649249249249249;
		x = (x | (x << 2))  & 0x1249249249249249;
		return x;
	}

	inline  index_node_t coordinateToIndex(vmml::vector<3, int> pos, int level, int nLevels)
	{
		if (level==0)
			return 1;

		index_node_t code	= (index_node_t)1 << (nLevels*3);
		index_node_t xcode	= dilateInteger((index_node_t) pos.x()) << 2;
		index_node_t ycode	= dilateInteger((index_node_t) pos.y()) << 1;
		index_node_t zcode	= dilateInteger((index_node_t) pos.z());

		code = code | xcode | ycode | zcode;

		code>>=(nLevels-level)*3;

		   if (code==0)
		   {
			   std::cout<<"Error, index cannot be zero "<<pos.x()<<","<<pos.y()<<","<<pos.z()<<" level "<<level<<std::endl;
			   std::exception();
		   }
		return code;
	}

	inline  vmml::vector<3, int> getMinBoxIndex(index_node_t index, int level, int nLevels)
	{
		vmml::vector<3, int> minBox(0,0,0);

		if (index == 1)
			return minBox;// minBOX (0,0,0) and level 0

		index_node_t mask = 1;

		for(int l=level; l>0; l--)
		{
			minBox[2] +=  (index & mask) << (nLevels-l); index>>=1;
			minBox[1] +=  (index & mask) << (nLevels-l); index>>=1;
			minBox[0] +=  (index & mask) << (nLevels-l); index>>=1;
		}

		if (index!=1)
			std::cerr<<"Error getting minBox from index"<<std::endl;

		return minBox;

	}

	bool _checkIso(float * voxel, float isosurface)
	{
			bool has = false; 
			bool sign = (voxel[0] - isosurface) < 0;
			for (int i = 1; i < 8 && !has; ++i)
				if (((voxel[i] - isosurface) < 0) != sign)
					has = true;

			return has;
	}


	void _getVoxel(int x, int y, int z, int dim, float * cube, float * voxel)
	{
		voxel[0]	= cube[posToIndex(x, y, z, dim)];
		voxel[1]	= cube[posToIndex(x, y, z+1, dim)];
		voxel[2]	= cube[posToIndex(x, y+1, z, dim)];
		voxel[3]	= cube[posToIndex(x, y+1, z+1, dim)];
		voxel[4]	= cube[posToIndex(x+1, y, z, dim)];
		voxel[5]	= cube[posToIndex(x+1, y, z+1, dim)];
		voxel[6]	= cube[posToIndex(x+1, y+1, z, dim)];
		voxel[7]	= cube[posToIndex(x+1, y+1, z+1, dim)];
	}


	void _checkCube(std::vector<octree*> octrees, std::vector<float> isos, int nLevels, int nodeLevel, int dimNode, index_node_t idCube, int cubeLevel, int cubeDim, float * cube)
	{
		vmml::vector<3, int> coorCubeStart = getMinBoxIndex(idCube, cubeLevel, nLevels);
		vmml::vector<3, int> coorCubeFinish = coorCubeStart + cubeDim - 2;

		std::cout<<"       "<<coorCubeStart<<" "<<coorCubeFinish<<std::endl;
		index_node_t start = idCube << 3*(nodeLevel - cubeLevel); 
		index_node_t finish = coordinateToIndex(coorCubeFinish, nodeLevel, nLevels);

		std::cout<<"       "<<idCube<<" "<<start<<" "<<finish<<std::endl;

		float voxel[8];
		int		nodesToCheck = octrees.size();
		bool *  checkNode = new bool[octrees.size()];

		while(start<=finish)
		{

			vmml::vector<3, int> coorNodeStart = getMinBoxIndex(start, nodeLevel, nLevels);
			vmml::vector<3, int> coorNodeFinish = coorNodeStart + dimNode - 1;
			#if 0
			std::cout<<"--------------------------"<<std::endl;
			std::cout<<start<<"      "<< coorNodeStart<< " "<<coorNodeFinish<<std::endl;
			std::cout<<start<<"      "<< coorNodeStart-coorCubeStart<< " "<<coorNodeFinish-coorCubeStart<<std::endl;
			std::cout<<"--------------------------"<<std::endl;
			#endif
			coorNodeStart = coorNodeStart-coorCubeStart;
			coorNodeFinish = coorNodeFinish-coorCubeStart;

			int     nodesToCheck = octrees.size();
			for(int i=0; i<octrees.size(); i++)
				checkNode[i] = true;


			for(int x=coorNodeStart.x(); x<=coorNodeFinish.x(); x++)
			{
				for(int y=coorNodeStart.y(); y<=coorNodeFinish.y(); y++)
				{
					for(int z=coorNodeStart.z(); z<=coorNodeFinish.z(); z++)
					{	
						if (nodesToCheck==0)
						{
							x = coorNodeFinish.x();
							y = coorNodeFinish.y();
							z = coorNodeFinish.z();
							break;
						}
						_getVoxel(x, y, z, cubeDim, cube, voxel);

						for(int i=0; i<octrees.size(); i++)
						{

							if (checkNode[i] && _checkIso(voxel, isos[i]))
							{
								checkNode[i] = false;
								nodesToCheck--;
								octrees[i]->addVoxel(start);
								//std::cout<<start<<" :) "<< isos[i]<<std::endl;
							}
						}
					}
				}
			}

			start++;
		}

		delete[] checkNode;
	}

	bool createOctree(std::string type_file, std::vector<std::string> file_params, int maxLevel, std::vector<float> isosurfaceList, std::string octree_file)
	{
		vmml::vector<3, int> cubeDim(2,2,2);
		vmml::vector<3, int> cubeInc(0,0,0);
		FileManager * file = eqMivt::CreateFileManage(type_file, file_params, maxLevel, maxLevel, cubeDim, cubeInc); 
		vmml::vector<3, int> realDim = file->getRealDimension();
		delete file;

		int dimension;
		int nLevels;

		if (realDim[0]>realDim[1] && realDim[0]>realDim[2])
			dimension = realDim[0];
		else if (realDim[1]>realDim[2])
			dimension = realDim[1];
		else
			dimension = realDim[2];

		/* Calcular dimension del árbol*/
		float aux = logf(dimension)/logf(2.0);
		float aux2 = aux - floorf(aux);
		nLevels = aux2>0.0 ? aux+1 : aux;
		dimension = pow(2,nLevels);
	
		if (maxLevel > nLevels)
		{
			std::cerr<<"MaxLevel has to be <= "<<nLevels<<std::endl;
			return false;
		}

		int levelCube = nLevels >= 9 ? ((nLevels-9) > maxLevel ? maxLevel : ((nLevels-9))): maxLevel;
		int dimCube = pow(2, nLevels - levelCube) + 1;
		cubeDim.set(dimCube, dimCube, dimCube);

		std::cout<<"Creating octree in file "<<octree_file<<std::endl;
		std::cout<<"Octree dimension "<<dimension<<"x"<<dimension<<"x"<<dimension<<" levels "<<nLevels<<std::endl;
		std::cout<<"Octree maximum level "<<maxLevel<<" dimension "<<pow(2, nLevels - maxLevel)<<"x"<<pow(2, nLevels - maxLevel)<<"x"<<pow(2, nLevels - maxLevel)<<std::endl;
		std::cout<<"Reading in block "<<cubeDim<<" level of cube "<<levelCube<<std::endl;
		file = eqMivt::CreateFileManage(type_file, file_params, levelCube, nLevels, cubeDim, cubeInc); 

		float * dataCube = new float[dimCube*dimCube*dimCube];

		std::vector<octree*> octrees(isosurfaceList.size());
		for(int i=0; i<isosurfaceList.size(); i++)
			octrees[i] = new octree(nLevels, maxLevel);

		index_node_t idStart = coordinateToIndex(vmml::vector<3, int>(0,0,0), levelCube, nLevels);
		index_node_t idFinish = coordinateToIndex(vmml::vector<3, int>(dimension-1, dimension-1, dimension-1), levelCube, nLevels);;

		std::cout<<"Start id "<<idStart<<" finish id "<<idFinish<<std::endl;

		while(idStart <= idFinish)
		{
			vmml::vector<3, int> currentBox = getMinBoxIndex(idStart, levelCube, nLevels);
			if (currentBox.x() < realDim[0] && currentBox.y() < realDim[1] && currentBox.z() < realDim[2])
			{
				file->readCube(idStart, dataCube);
				std::cout<<currentBox<<std::endl;
				_checkCube(octrees, isosurfaceList, nLevels, maxLevel, pow(2,nLevels-maxLevel), idStart, levelCube, dimCube, dataCube);
			}

			idStart++;
		}

		delete[] dataCube;

		for(int i=0; i<octrees.size(); i++)
		{
			octrees[i]->printTree();
			delete octrees[i];
		}

		delete file;

		return true;
	}

}
