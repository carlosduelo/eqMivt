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
#include <iterator>
#include <lunchbox/clock.h>
#include <lunchbox/lock.h>
#include <boost/progress.hpp>
#include <iostream>
#include <fstream>

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))


namespace eqMivt
{

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
			std::cerr<<"Error getting minBox from index nLevels  "<<nLevels<<" leve "<<level<<" id "<<index<<std::endl;

		return minBox;

	}

	class octree
	{
		private: 
			lunchbox::Lock				_lock;
			std::vector<index_node_t> *	_octree;
			int		*					_numCubes;
			int							_dim;
			int							_maxHeight;
			int							_maxLevel;
			int							_nLevels;
			float						_iso;
			vmml::vector<3, int>		_start;
			vmml::vector<3, int>		_finish;
			int							_numElements;
			std::string					_nameFile;
			std::ofstream				_tempFile;

			bool _addElement(index_node_t id, int level)
			{
				int size = _octree[level].size();

				try
				{
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
				}
				catch (...)
				{
					std::cerr<<"No enough memory aviable"<<std::endl;
					throw;
				}

				return false;
			}


		public:
			octree(int nLevels, int maxLevel, float iso, vmml::vector<3, int> start, vmml::vector<3, int> finish)
			{

				_octree		= new std::vector<index_node_t>[maxLevel + 1];
				_numCubes	= new int[maxLevel + 1];	
				bzero(_numCubes, (maxLevel + 1)*sizeof(int));
				_numElements = 0;

				_iso		= iso;
				_maxLevel	= maxLevel;
				_nLevels	= nLevels;
				_maxHeight	= 0;
				_dim = exp2(_nLevels - _maxLevel);

				_start = start;
				_finish = finish;

				std::ostringstream convert;
				convert <<rand() <<nLevels << maxLevel << iso << ".tmp";
				_nameFile = convert.str();

				_tempFile.open(_nameFile.c_str(), std::ofstream::binary | std::ofstream::trunc);
			}

			~octree()
			{
				remove(_nameFile.c_str());
				if (_octree != 0)
					delete[] _octree;
				if (_numCubes != 0)
					delete[] _numCubes;
			}

			void completeOctree()
			{
				try
				{
					_tempFile.close();
					std::vector<index_node_t> lastLevel;
					std::ifstream File(_nameFile.c_str(), std::ifstream::binary);

					for(int i=0; i< _numElements; i++)
					{
						index_node_t a = 0;
						File.read((char*) &a, sizeof(index_node_t));
						lastLevel.push_back(a);
					}

					std::sort(lastLevel.begin(), lastLevel.end());
					lastLevel.erase( std::unique( lastLevel.begin(), lastLevel.end() ), lastLevel.end() );
					for (std::vector<index_node_t>::iterator it=lastLevel.begin(); it!=lastLevel.end(); ++it)
					{
						index_node_t id = *it;

						vmml::vector<3, int> coorFinishStart = getMinBoxIndex(id, _maxLevel, _nLevels) + (vmml::vector<3, int>(1,1,1)*_dim);
						if (coorFinishStart.y() > _maxHeight)
							_maxHeight = coorFinishStart.y();

						for(int i=_maxLevel; i>=0; i--)
						{
							if (_addElement(id, i))
								break;
							id >>= 3;
						}
					}
					lastLevel.clear();

				}
				catch (...)
				{
					std::cerr<<"Not enough memory aviable"<<std::endl;
					throw;
				}
			}

			void addVoxel(index_node_t id)
			{
				_lock.set();
				_numElements++;
				_tempFile.write((char*)&id, sizeof(index_node_t));
				_lock.unset();
			}

			float getIso()
			{
				return _iso;
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
				remove(_nameFile.c_str());
				delete[] _octree; _octree = 0;
				delete[] _numCubes; _numCubes = 0;

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
				int num = (id + inc) > finish ? finish - id  + 1: inc;
				octreeConstructorComputeCube(resultGPU, num, id, isos[i], cube, nodeLevel, nLevels, dimNode, cubeDim, coorCubeStartV);

				bzero(resultCPU, inc*sizeof(index_node_t));

				if (!octreeConstructorCopyResult(resultCPU, resultGPU, num))
				{
					std::cerr<<"Error copying structures from a cuda device"<<std::endl;
					throw;
				}
				for(int j=0; j<num; j++)
				{
					if (resultCPU[j] != (index_node_t)0)
					{
						vmml::vector<3, int> coorNodeStart = getMinBoxIndex(resultCPU[j], nodeLevel, nLevels);
						vmml::vector<3, int> coorNodeFinish = coorNodeStart + dimNode - 1;
						octrees[i]->addVoxel(resultCPU[j]);
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
							}
						}
					}
				}
			}
			if (checkNode != 0)
				delete[] checkNode;
		}

	}

	void _writeToFile(std::vector<octree*> octrees, std::vector<int> maxLevel, std::vector<int> nLevel, std::vector<int> numOctrees, std::vector< vmml::vector<3, int> > startCoordinates, std::vector< vmml::vector<3, int> > finishCoordinates, vmml::vector<3, int> realDim, float * xGrid, float * yGrid, float * zGrid, std::string octree_file)
	{
		std::ofstream file(octree_file.c_str(), std::ofstream::binary);

		int magicWord = 919278872;

		int numO = octrees.size();

		file.write((char*)&magicWord,  sizeof(magicWord));
		file.write((char*)&numO,sizeof(numO));
		file.write((char*)&realDim.array,3*sizeof(int));
		file.write((char*)xGrid, realDim[0]*sizeof(float));
		file.write((char*)yGrid, realDim[1]*sizeof(float));
		file.write((char*)zGrid, realDim[2]*sizeof(float));

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
		int initDesp =	5*sizeof(int) + realDim[0]*sizeof(float) + realDim[1]*sizeof(float) + realDim[2]*sizeof(float) +
					numOctrees.size() * 9 *sizeof(int) + numO * sizeof(float);

		int desp	= 5*sizeof(int) + realDim[0]*sizeof(float) + realDim[1]*sizeof(float) + realDim[2]*sizeof(float) +
						numOctrees.size() * 9 *sizeof(int) + numO * sizeof(float) + numO*sizeof(int);

		int offsets[numO];
		offsets[0] = desp;

		for(int i=0; i<numO; i++)
		{
			file.seekp(initDesp, std::ios_base::beg);
			file.write((char*)&desp, sizeof(desp));
			initDesp += sizeof(int);

			octrees[i]->completeOctree();	
			desp = octrees[i]->getSize();
			if (i < numO-1)
				offsets[i+1] = desp;

			file.seekp(offsets[0], std::ios_base::beg);
			for(int d=1; d<=i; d++)
				file.seekp(offsets[d], std::ios_base::cur);

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
		double * dxGrid = 0;
		double * dyGrid = 0;
		double * dzGrid = 0;
		if (!file->getxGrid(&dxGrid) || 
			!file->getyGrid(&dyGrid) ||
			!file->getzGrid(&dzGrid))
		{
			std::cerr<<"Error reading grid"<<std::endl;
			return false;
		}
		float * xGrid = new float[realDim.x()];
		float * yGrid = new float[realDim.y()];
		float * zGrid = new float[realDim.z()];
		for(int i=0; i<realDim.x(); i++)
			xGrid[i]=(float)dxGrid[i];
		for(int i=0; i<realDim.y(); i++)
			yGrid[i]=(float)dyGrid[i];
		for(int i=0; i<realDim.z(); i++)
			zGrid[i]=(float)dzGrid[i];

		delete[] dxGrid;
		delete[] dyGrid;
		delete[] dzGrid;

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
				std::cerr<<"MaxLevel "<<mxLevel<<" has to be <= "<<nLevels<<std::endl;
				return false;
			}

			_nLevels.push_back(nLevels);

			int levelCubeGPU = 0; 
			int levelCubeCPU = 0;
			double memoryCPU = getMemorySize();
			if (useCUDA)
			{
				double memoryGPU = octreeConstructorGetFreeMemory();
				for(int l=0; l <= mxLevel; l++)
				{
					double bigCubeDim = pow(pow(2, nLevels - l), 3)*(float)sizeof(float);
					if ((1.2f*bigCubeDim) < memoryGPU)
					{
						levelCubeGPU = l;
						break;
					}
				}
			}
			if (memoryCPU == 0)
			{
				std::cerr<<"Not possible, check memory aviable (the call failed due to OS limitations)"<<std::endl;
				levelCubeCPU = nLevels >= 10 ? ((nLevels-10) > mxLevel ? mxLevel : ((nLevels-10))): 0;
			}
			else
			{
				for(int l=0; l <= mxLevel; l++)
				{
					double bigCubeDim = pow(pow(2, nLevels - l), 3)*(float)sizeof(float);
					if ((1.7f*bigCubeDim) < memoryCPU)
					{
						levelCubeCPU = l;
						break;
					}
				}
			}

			int dimCubeCPU = pow(2, nLevels - levelCubeCPU) + 1;
			vmml::vector<3, int> cubeDimCPU(dimCubeCPU, dimCubeCPU, dimCubeCPU);
			vmml::vector<3, int> cubeInc(0,0,0);
			vmml::vector<3, int> realcubeDimCPU	= cubeDimCPU + 2 * cubeInc;
			int dimCubeGPU = 0;
			vmml::vector<3, int> cubeDimGPU;
			vmml::vector<3, int> realcubeDimGPU;
			if (useCUDA)
			{
				dimCubeGPU = pow(2, nLevels - levelCubeGPU) + 1;
				cubeDimGPU.set(dimCubeGPU, dimCubeGPU, dimCubeGPU);
				realcubeDimGPU	= cubeDimGPU + 2 * cubeInc;
			}

			std::cout<<"Octree dimension "<<dimension<<"x"<<dimension<<"x"<<dimension<<" levels "<<nLevels<<std::endl;
			std::cout<<"Octree maximum level "<<mxLevel<<" dimension "<<pow(2, nLevels - mxLevel)<<"x"<<pow(2, nLevels - mxLevel)<<"x"<<pow(2, nLevels - mxLevel)<<std::endl;
			std::cout<<"Reading in block "<<cubeDimCPU<<" level of cube "<<levelCubeCPU<<std::endl;
			if (useCUDA)
				std::cout<<"CUDA size cube "<<cubeDimGPU<<" level of cube "<<levelCubeGPU<<std::endl;
			std::cout<<"Coordinates from "<<start<<" to "<<finish<<" Isosurfaces: ";

			for(int j=0;j<nO; j++)
			{
				_octrees[j] = new octree(nLevels, mxLevel, isos[j], start, finish);
				std::cout<<isos[j]<<" ";
			}
			std::cout<<std::endl;
			
			float * dataCube = new float[dimCubeCPU*dimCubeCPU*dimCubeCPU];
			float * dataCubeGPU = 0;
			if (useCUDA)
			{
				//Create CUDA memory
				dataCubeGPU = octreeConstructorCreateCube(dimCubeGPU);
				if (dataCubeGPU == 0)
				{
					std::cerr<<"Error allocating memory in a cuda device"<<std::endl;
					throw;
				}
			}

			index_node_t idStart = coordinateToIndex(vmml::vector<3, int>(0,0,0), levelCubeCPU, nLevels);
			index_node_t idFinish = coordinateToIndex(vmml::vector<3, int>(dimension-1, dimension-1, dimension-1), levelCubeCPU, nLevels);

			#ifdef NDEBUG
			boost::progress_display show_progress(idFinish - idStart + 1);
			#endif

			lunchbox::Clock		readingClock;
			lunchbox::Clock		computinhClock;
			double				readingTime = 0.0;
			double				computingTime = 0.0;

			file->setOffset(start);

			for(index_node_t id=idStart; id<= idFinish; id++)
			{
				#ifndef NDEBUG
				std::cout<<"Iterations "<<id<<" "<<idFinish<<std::endl;
				#endif

				vmml::vector<3, int> currentBoxCPU = getMinBoxIndex(id, levelCubeCPU, nLevels);
				if ((start[0] + currentBoxCPU.x()) < finish[0] && (start[1] + currentBoxCPU.y()) < finish[1] && (start[2] + currentBoxCPU.z()) < finish[2])
				{
					readingClock.reset();
					file->readCube(id, dataCube, levelCubeCPU, nLevels, cubeDimCPU, cubeInc, realcubeDimCPU);
					readingTime += readingClock.getTimed();
					computinhClock.reset();
					if (useCUDA)
					{
						index_node_t startG		= coordinateToIndex(currentBoxCPU, levelCubeGPU, nLevels);
						index_node_t finishG	= coordinateToIndex(currentBoxCPU + vmml::vector<3, int>(dimCubeCPU-2, dimCubeCPU-2, dimCubeCPU-2), levelCubeGPU, nLevels);

						#ifndef NDEBUG
						std::cout<<"\tIndex cube gpu "<<startG<<" "<<finishG<<std::endl;
						#endif

						for(index_node_t idG = startG; idG<=finishG; idG++)
						{
							vmml::vector<3, int> startGPU = getMinBoxIndex(idG, levelCubeGPU, nLevels); 
							vmml::vector<3, int>  endGPU = startGPU + vmml::vector<3, int> (dimCubeGPU, dimCubeGPU, dimCubeGPU);
							vmml::vector<3, int> offset = startGPU - currentBoxCPU;  

							#ifndef NDEBUG
							std::cout<<"\tCopying volume to gpu"<<startGPU<<" to "<<endGPU<<std::endl;
							#endif

							if(!octreeConstructorCopyCube3D(dataCubeGPU, dataCube, dimCubeCPU, dimCubeGPU, offset.x(), offset.y() , offset.z()))
							{
								std::cerr<<"Error copying cube to cuda device"<<std::endl;
								return 0;
							}

							_checkCube_cuda(_octrees, isos, nLevels, mxLevel, pow(2,nLevels-mxLevel), idG, levelCubeGPU, dimCubeGPU, dataCubeGPU);
						}
					}
					else
					{
						_checkCube(_octrees, isos, nLevels, mxLevel, pow(2,nLevels-mxLevel), id, levelCubeCPU, dimCubeCPU, dataCube);
					}

					computingTime += computinhClock.getTimed();
				}
				#ifdef NDEBUG
				++show_progress;
				#endif
			}

			delete[] dataCube;
			if (useCUDA)
				octreeConstructorDestroyCube(dataCubeGPU);

			for(int j=0;j<nO; j++)
				octrees.push_back(_octrees[j]);

			std::cout<<"Hard disk reading time: "<<readingTime/1000.0<<" seconds."<<std::endl;
			std::cout<<"Computing time: "<<computingTime/1000.0<<" seconds"<<std::endl;
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
			delete octrees[i];
		}

		return true;
	}

}
