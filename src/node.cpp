/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "node.h"

#include "config.h"
#include "error.h"

#define MAX_WORKERS 32
#define CUBE_INC 2

namespace eqMivt
{
	bool Node::configInit( const eq::uint128_t& initID )
	{
		// All render data is static or multi-buffered, we can run asynchronously
		if( getIAttribute( IATTR_THREAD_MODEL ) == eq::UNDEFINED )
			setIAttribute( IATTR_THREAD_MODEL, eq::ASYNC );

		if( !eq::Node::configInit( initID ))
			return false;

		Config* config = static_cast< Config* >( getConfig( ));
		if( !config->loadData( initID ))
		{
			setError( ERROR_EQ_MIVT_FAILED );
			return false;
		}

		// Init cpu Cache
		const InitData& initData = config->getInitData();

		_status = true;

		_status =	_octreeManager.init(initData.getOctreeFilename());
		if (!_status)
		{
			LBERROR<<"Node: error creating octree manager"<<std::endl;
		}
		else
		{
			_status = _cacheManager.init(initData.getDataTypeFile(), initData.getDataFilename(), CUBE_INC); 
			if (!_status)
				LBERROR<<"Node: error creating cache manager"<<std::endl;
		}

		config->setOctreeManager(&_octreeManager);

		return _status;
	}

	bool Node::configExit()
	{
		return eq::Node::configExit();
	}
	
	vmml::vector<3, float>    Node::getVolumeCoord()
	{ 
		return _octreeManager.getRealDimVolume(); 
	}

	vmml::vector<3, float>    Node::getStartCoord()
	{
		return _octreeManager.getCurrentStartCoord();
	}

	vmml::vector<3, float>    Node::getFinishCoord()
	{
		return _octreeManager.getCurrentFinishCoord();
	}

	Octree *	Node::getOctree(uint32_t device)
	{
		if (_status)
		{
			Octree * o = _octreeManager.getOctree(device);
			_status = o == 0 ? false : true;
			return o;
		}
	}

	bool Node::getCacheHandler(uint32_t device, CacheHandler * cacheHandler)
	{
		if (_status)
			if (!cacheHandler->isValid())
				_status = _cacheManager.getCache(device, cacheHandler);
		
		return _status;
	}
	
	bool Node::updateStatus(uint32_t device, CacheHandler * cacheHandler, int currentOctree, bool grid, bool renderCubes)
	{
		if (_status)
		{
			bool forceResizeGPUCache = false;
			// Octree set Current Octree
			if (!_octreeManager.setCurrentOctree(currentOctree, grid, renderCubes, &forceResizeGPUCache))
			{
				_status = false;
				LBERROR<<"Node: Error setting current octree"<<std::endl;
			}
			if (_status && forceResizeGPUCache)
			{
				_status = _cacheManager.forceResize(cacheHandler);
				if (!_status)
					LBERROR<<"Node: Error forcing resize cache gpu"<<std::endl;
			}

			Config* config = static_cast< Config* >( getConfig( ));
			config->setOctreeManager(&_octreeManager);
			const InitData& initData = config->getInitData();
			// Set Size cache manager for CPU
			int levelCube = _octreeManager.getBestCubeLevel();
			int nL = _octreeManager.getNLevels();
			int numElements = initData.getMaxCubesCacheGPU();
			int numElementsCPU = initData.getMaxCubesCacheCPU();
			int levelCubeCPU  = _octreeManager.getBestCubeLevelCPU();
			if (_status)
			{
				_status =	_cacheManager.reSize(levelCube, nL, numElements, numElementsCPU, levelCubeCPU) &&
							_cacheManager.setOffset(_octreeManager.getCurrentOffset());
				if (!_status)
					LBERROR<<"Node: Error resizing cache cpu"<<std::endl;
				else
				{
					_status =  _octreeManager.checkStatus(device);
					if (!_status)
						LBERROR<<"Node: Error checking octree"<<std::endl;
					else
					{
						_status =  _cacheManager.checkStatus(cacheHandler);
						if (!_status)
							LBERROR<<"Node: Error checking cache gpu"<<std::endl;
					}
				}
			}
		}
		return _status;
	}

}
