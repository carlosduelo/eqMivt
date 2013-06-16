/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "octreeConstructor.h"
#include "octreeConstructor_CUDA.h"

#include "fileFactory.h"

#include "memoryCheck.h"

#include <omp.h>
#include <algorithm>
#include <lunchbox/clock.h>
#include <lunchbox/lock.h>
#include <boost/progress.hpp>
#include <iostream>
#include <fstream>

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

namespace eqMivt
{
	class octree
	{
		private: 
			lunchbox::Lock				_lock;
			std::vector<index_node_t> 	_lastLevel;
			std::vector<index_node_t> *	_octree;
			int		*					_numCubes;
			int							_maxHeight;
			int							_maxLevel;
			int							_nLevels;
			float						_iso;

			bool _addElement(index_node_t id, int level)
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
					return true;
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

				return false;
			}

		public:
			octree(int nLevels, int maxLevel, float iso)
			{

				_octree		= new std::vector<index_node_t>[maxLevel + 1];
				_numCubes	= new int[maxLevel + 1];	

				_iso		= iso;
				_maxLevel	= maxLevel;
				_nLevels	= nLevels;
				_maxHeight	= 0;
			}

			~octree()
			{
				delete[] _octree;
				delete[] _numCubes;
			}

			void addVoxel(index_node_t id)
			{
				_lock.set();
				_lastLevel.push_back(id);
				_lock.unset();
			}

			void completeOctree()
			{
				std::sort(_lastLevel.begin(), _lastLevel.end());
				for (std::vector<index_node_t>::iterator it=_lastLevel.begin(); it!=_lastLevel.end(); ++it)
				{
					index_node_t id = *it;
					for(int i=_maxLevel; i>=0; i--)
					{
						if (_addElement(id, i))
							break;
						id >>= 3;
					}
				}
				_lastLevel.clear();
			}

			void reportHeight(int height)
			{
				if (_maxHeight < height)
					_maxHeight = height;
			}

			float getIso()
			{
				return _iso;
			}

			int getMaxHeight()
			{
				return _maxHeight; 
			}

			int getSize()
			{
				int size = (2*(_maxLevel+1) + 1)*sizeof(int);

				for(int i=0; i<=_maxLevel; i++)
					size +=_octree[i].size()*sizeof(index_node_t); 

				return size;
			}

			void writeToFile(std::ofstream * file)
			{
				file->write((char*)&_maxHeight, sizeof(int));
				file->write((char*)_numCubes, (_maxLevel+1)*sizeof(int));
				for(int i=0; i<=_maxLevel; i++)
				{
					int s = _octree[i].size();
					file->write((char*)&s, sizeof(int));
				}

				for(int i=0; i<=_maxLevel; i++)
				{
					file->write((char*)_octree[i].data(), _octree[i].size()*sizeof(index_node_t));
				}
			}

			void printTree()
			{
				std::cout<<"Isosurface "<<_iso<<std::endl;
				std::cout<<"Maximum height "<<_maxHeight<<std::endl;
				for(int i=_maxLevel; i>=0; i--)
				{
					std::vector<index_node_t>::iterator it;

					std::cout	<<"Level: "<<i<<" num cubes "<<_numCubes[i]<<" size "<<_octree[i].size()<<" porcentaje "<<100.0f - ((_octree[i].size()*100.0f)/(float)_numCubes[i])
								<<" cube dimension "<<pow(2,_nLevels-i)<<"^3 "<<" memory needed for level "<< (_numCubes[i]*pow(pow(2,_nLevels-i),3)*sizeof(float))/1024.0f/1024.0f<<" MB"<<std::endl;
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

	bool _checkIsosurface(int x, int y, int z, int dim, float * cube, float isosurface)
	{
		bool sign = (cube[posToIndex(x, y, z, dim)] -isosurface) < 0;

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


	void _checkCube_cuda(std::vector<octree*> octrees, std::vector<float> isos, int nLevels, int nodeLevel, int dimNode, index_node_t idCube, int cubeLevel, int cubeDim, float * cube)
	{
		vmml::vector<3, int> coorCubeStart = getMinBoxIndex(idCube, cubeLevel, nLevels);
		vmml::vector<3, int> coorCubeFinish = coorCubeStart + cubeDim - 2;
		int coorCubeStartV[3] = {coorCubeStart.x(), coorCubeStart.y(), coorCubeStart.z()};

		index_node_t start = idCube << 3*(nodeLevel - cubeLevel); 
		index_node_t finish = coordinateToIndex(coorCubeFinish, nodeLevel, nLevels);

		double freeMemory = octreeConstructorGetFreeMemory();
		int inc = finish - start + 1;
		int rest = inc / 10;
		while(inc*sizeof(index_node_t) > freeMemory)
		{
			inc = inc - rest;
		}
		index_node_t * resultCPU = new index_node_t[inc];
		index_node_t * resultGPU = 0;
		resultGPU = octreeConstructorCreateResult(inc);
		if (resultGPU == 0)
		{
			std::cerr<<"Error creating a structure in a cuda device"<<std::endl;
			throw;
		}

		for(index_node_t id=start; id<=finish; id+=inc)
		{
			for(int i=0; i<octrees.size(); i++)
			{
				octreeConstructorComputeCube(resultGPU, inc, id, isos[i], cube, nodeLevel, nLevels, dimNode, cubeDim, coorCubeStartV);
				if (!octreeConstructorCopyResult(resultCPU, resultGPU, inc))
				{
					std::cerr<<"Error copying structures from a cuda device"<<std::endl;
					throw;
				}
				for(int j=0; j<inc; j++)
				{
					if (resultCPU[j] != (index_node_t)0)
					{
						vmml::vector<3, int> coorNodeStart = getMinBoxIndex(resultCPU[j], nodeLevel, nLevels);
						vmml::vector<3, int> coorNodeFinish = coorNodeStart + dimNode - 1;
						octrees[i]->addVoxel(resultCPU[j]);
						octrees[i]->reportHeight(coorNodeFinish.y());
					}
				}
			}
		}

		delete[] resultCPU;
		octreeConstructorDestroyResult(resultGPU);
	}

	void _checkCube(std::vector<octree*> octrees, std::vector<float> isos, int nLevels, int nodeLevel, int dimNode, index_node_t idCube, int cubeLevel, int cubeDim, float * cube)
	{
		vmml::vector<3, int> coorCubeStart = getMinBoxIndex(idCube, cubeLevel, nLevels);
		vmml::vector<3, int> coorCubeFinish = coorCubeStart + cubeDim - 2;

		index_node_t start = idCube << 3*(nodeLevel - cubeLevel); 
		index_node_t finish = coordinateToIndex(coorCubeFinish, nodeLevel, nLevels);

		#pragma omp parallel for
		for(index_node_t id=start; id<=finish; id++)
		{

			vmml::vector<3, int> coorNodeStart = getMinBoxIndex(id, nodeLevel, nLevels);
			vmml::vector<3, int> coorNodeFinish = coorNodeStart + dimNode - 1;
			coorNodeStart = coorNodeStart-coorCubeStart;
			coorNodeFinish = coorNodeFinish-coorCubeStart;

			int     nodesToCheck = octrees.size();
			bool *  checkNode = new bool[nodesToCheck];
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
							delete[] checkNode;
							checkNode = 0;
							break;
						}

						for(int i=0; i<octrees.size(); i++)
						{

							if (checkNode[i] && _checkIsosurface(x, y, z, cubeDim, cube, isos[i]))
							{
								checkNode[i] = false;
								nodesToCheck--;
								octrees[i]->addVoxel(id);
								octrees[i]->reportHeight(coorNodeFinish.y());
							}
						}
					}
				}
			}
			if (checkNode != 0)
				delete[] checkNode;
		}

	}

	void _writeToFile(std::vector<octree*> octrees, std::vector<int> maxLevel, std::vector<int> nLevel, std::vector<int> numOctrees, std::vector< vmml::vector<3, int> > startCoordinates, std::vector< vmml::vector<3, int> > finishCoordinates, vmml::vector<3, int> realDim, double * xGrid, double * yGrid, double * zGrid, std::string octree_file)
	{
		std::ofstream file(octree_file.c_str(), std::ofstream::binary);

		int magicWord = 919278872;

		int numO = octrees.size();

		file.write((char*)&magicWord,  sizeof(magicWord));
		file.write((char*)&numO,sizeof(numO));
		file.write((char*)&realDim.array,3*sizeof(int));
		file.write((char*)xGrid, realDim[0]*sizeof(double));
		file.write((char*)yGrid, realDim[1]*sizeof(double));
		file.write((char*)zGrid, realDim[2]*sizeof(double));

		int nO = 0;
		for(int i=0; i<maxLevel.size(); i++)
		{
			file.write((char*)&numOctrees[i], sizeof(int));
			file.write((char*)&startCoordinates[i].array,3*sizeof(int));
			file.write((char*)&finishCoordinates[i].array,3*sizeof(int));
			file.write((char*)&nLevel[i], sizeof(int));
			file.write((char*)&maxLevel[i], sizeof(int));

		}

		for(int i=0; i<numO; i++)
		{
			float iso = octrees[i]->getIso();;
			file.write((char*)&iso,sizeof(float));
		}

		// offset from start
		int desp =	5*sizeof(int) + realDim[0]*sizeof(double) + realDim[1]*sizeof(double) + realDim[2]*sizeof(double) +
					numOctrees.size() * 9 *sizeof(int) + numO * sizeof(float) + numO*sizeof(int);
		for(int i =0; i<numO; i++)
		{
			std::cout<<desp<<std::endl;
			file.write((char*)&desp,sizeof(desp));
			desp = octrees[i]->getSize(); 
		}

		for(int i =0; i<numO; i++)
		{
			octrees[i]->writeToFile(&file);
		}

		file.close();	

		return;
	}

 bool createOctree(std::string type_file, std::vector<std::string> file_params, std::vector<int> maxLevel, std::vector< std::vector<float> > isosurfaceList, std::vector<int> numOctrees, std::vector< vmml::vector<3, int> > startCoordinates, std::vector< int > octreeDimension, std::string octree_file, bool useCUDA)
	{

		FileManager * file = eqMivt::CreateFileManage(type_file, file_params);
		vmml::vector<3, int> realDim = file->getRealDimension();

		// get grid vectors
		double * xGrid = 0;
		double * yGrid = 0;
		double * zGrid = 0;
		if (!file->getxGrid(&xGrid) || 
			!file->getyGrid(&yGrid) ||
			!file->getzGrid(&zGrid))
		{
			std::cerr<<"Error reading grid"<<std::endl;
			return false;
		}

		lunchbox::Clock		completeCreationClock;
		completeCreationClock.reset();

		std::vector<int>	_nLevels;
		std::vector<octree*> octrees;
		std::vector< vmml::vector<3, int> > finishCoordinates;
		for(int i=0; i<numOctrees.size(); i++)
		{
			int nO = numOctrees[i];
			int mxLevel = maxLevel[i];
			std::vector< float > isos = isosurfaceList[i];
			std::vector<octree*> _octrees(nO);

			vmml::vector<3, int> start  = startCoordinates[i];
			int dimension = octreeDimension[i];
			vmml::vector<3, int> finish = start + dimension * vmml::vector<3, int>(1,1,1); 
			finish[0] = finish[0] >= realDim[0] ? realDim[0] - 1 : finish[0];
			finish[1] = finish[1] >= realDim[1] ? realDim[1] - 1 : finish[1];
			finish[2] = finish[2] >= realDim[2] ? realDim[2] - 1: finish[2];
			finishCoordinates.push_back(finish);

			int nLevels = 0;
			/* Calcular dimension del árbol*/
			float aux = logf(dimension)/logf(2.0);
			float aux2 = aux - floorf(aux);
			nLevels = aux2>0.0 ? aux+1 : aux;
	
			if (mxLevel > nLevels)
			{
				std::cerr<<"MaxLevel has to be <= "<<nLevels<<std::endl;
				return false;
			}

			_nLevels.push_back(nLevels);

			int levelCube = 0; 
			double memoryCPU = getMemorySize();
			if (memoryCPU == 0)
			{
				std::cerr<<"Not possible, check memory aviable (the call failed due to OS limitations)"<<std::endl;
				levelCube = nLevels >= 9 ? ((nLevels-9) > mxLevel ? mxLevel : ((nLevels-9))): 0;
			}
			else
			{
				if (useCUDA)
				{
					double memoryGPU = octreeConstructorGetFreeMemory();
					for(int l=10; l>0; l--)
					{
						double bigCubeDim = pow(pow(2, l) + 1, 3)*(float)sizeof(float);
						if ((2.0*bigCubeDim) < memoryGPU)
						{
							levelCube = nLevels >= l ? ((nLevels-l) > mxLevel ? mxLevel : ((nLevels-l))): 0;
							break;
						}
					}
				}
				else if (nLevels <= 9)
				{
						levelCube = nLevels >= 9 ? ((nLevels-9) > mxLevel ? mxLevel : ((nLevels-9))): 0;
				}
				else
				{
					for(int l=10; l>0; l--)
					{
						double bigCubeDim = pow(pow(2, l) + 1, 3)*(float)sizeof(float);
						if ((2.0*bigCubeDim) < memoryCPU)
						{
							levelCube = nLevels >= l ? ((nLevels-l) > mxLevel ? mxLevel : ((nLevels-l))): 0;
							break;
						}
					}
				}
			}


			int dimCube = pow(2, nLevels - levelCube) + 1;
			vmml::vector<3, int> cubeDim(dimCube, dimCube, dimCube);
			vmml::vector<3, int> cubeInc(0,0,0);
			vmml::vector<3, int> realcubeDim	= cubeDim + 2 * cubeInc;

			std::cout<<"Octree dimension "<<dimension<<"x"<<dimension<<"x"<<dimension<<" levels "<<nLevels<<std::endl;
			std::cout<<"Octree maximum level "<<mxLevel<<" dimension "<<pow(2, nLevels - mxLevel)<<"x"<<pow(2, nLevels - mxLevel)<<"x"<<pow(2, nLevels - mxLevel)<<std::endl;
			std::cout<<"Reading in block "<<cubeDim<<" level of cube "<<levelCube<<std::endl;
			std::cout<<"Coordinates from "<<start<<" to "<<finish<<" Isosurfaces: ";
			for(int j=0;j<nO; j++)
			{
				_octrees[j] = new octree(nLevels, mxLevel, isos[j]);
				std::cout<<isos[j]<<" ";
			}
			std::cout<<std::endl;
			
			float * dataCube = new float[dimCube*dimCube*dimCube];
			float * dataCubeGPU = 0;
			if (useCUDA)
			{
				//Create CUDA memory
				dataCubeGPU = octreeConstructorCreateCube(dimCube);
				if (dataCubeGPU == 0)
				{
					std::cerr<<"Error allocating memory in a cuda device"<<std::endl;
					throw;
				}
			}

			index_node_t idStart = coordinateToIndex(vmml::vector<3, int>(0,0,0), levelCube, nLevels);
			index_node_t idFinish = coordinateToIndex(vmml::vector<3, int>(dimension-1, dimension-1, dimension-1), levelCube, nLevels);

			boost::progress_display show_progress(idFinish - idStart + 1);

			lunchbox::Clock		readingClock;
			lunchbox::Clock		computinhClock;
			double				readingTime = 0.0;
			double				computingTime = 0.0;

			file->setOffset(start);

			for(index_node_t id=idStart; id<= idFinish; id++)
			{
				vmml::vector<3, int> currentBox = getMinBoxIndex(id, levelCube, nLevels);
				if ((start[0] + currentBox.x()) < finish[0] && (start[1] + currentBox.y()) < finish[1] && (start[2] + currentBox.z()) < finish[2])
				{
					readingClock.reset();
					file->readCube(id, dataCube, levelCube, nLevels, cubeDim, cubeInc, realcubeDim);
					readingTime += readingClock.getTimed();
					computinhClock.reset();
					if (useCUDA)
					{
						if(!octreeConstructorCopyCube(dataCubeGPU, dataCube, dimCube))
						{
							std::cerr<<"Error copying cube to cuda device"<<std::endl;
							throw;
						}
						_checkCube_cuda(_octrees, isos, nLevels, mxLevel, pow(2,nLevels-mxLevel), id, levelCube, dimCube, dataCubeGPU);
					}
					else
					{
						_checkCube(_octrees, isos, nLevels, mxLevel, pow(2,nLevels-mxLevel), id, levelCube, dimCube, dataCube);
					}

					computingTime += computinhClock.getTimed();
				}
				++show_progress;
			}

			delete[] dataCube;
			if (useCUDA)
				octreeConstructorDestroyCube(dataCubeGPU);

			computinhClock.reset();
			#pragma omp parallel for
			for(int i=0; i<_octrees.size(); i++)
				_octrees[i]->completeOctree();
			double completeTime = computinhClock.getTimed();

			for(int j=0;j<nO; j++)
				octrees.push_back(_octrees[j]);

			std::cout<<"Hard disk reading time: "<<readingTime/1000.0<<" seconds."<<std::endl;
			std::cout<<"Computing time: "<<computingTime/1000.0<<" seconds, "<<"time in complete octree "<<completeTime/1000.0<<" seconds."<<std::endl;
		}

		std::cout<<"Creating octree in file "<<octree_file<<std::endl;
		lunchbox::Clock		writingClock;
		writingClock.reset();
		_writeToFile(octrees, maxLevel, _nLevels, numOctrees, startCoordinates, finishCoordinates, realDim, xGrid, yGrid, zGrid, octree_file);
		double writeFileTime = writingClock.getTimed();

		double time = completeCreationClock.getTimed();
		std::cout<<"Toltal time: "<<time/1000.0<<" seconds."<<std::endl;
		std::cout<<"Time writing file "<<writeFileTime/1000.0<<" seconds."<<std::endl;

		delete file;
		delete[] xGrid;
		delete[] yGrid;
		delete[] zGrid;

		for(int i=0; i<octrees.size(); i++)
		{
			//octrees[i]->printTree();
			delete octrees[i];
		}

		return true;
	}

}
