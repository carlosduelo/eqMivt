/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "node.h"

#include "config.h"
#include "error.h"

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
		return true;
	}

	bool Node::configExit()
	{
	    for (std::map<int , eqMivt::OctreeContainer *>::iterator it = _octrees.begin(); it!=_octrees.end(); it++)
	        delete it->second;
	}

	bool Node::registerPipeResources(int device)
	{
	    _lock.set();

	    // Check octree
	    std::map<int , eqMivt::OctreeContainer *>::iterator it;
	    it = _octrees.find(device);

	    if (it ==  _octrees.end())
	    {
	        _octrees[device] = new eqMivt::OctreeContainer(device);
		
		Config* config = static_cast< Config* >( getConfig( ));
		const InitData& initData = config->getInitData();

		if (!_octrees[device]->readOctreeFile(initData.getOctreeFilename(), initData.getOctreeMaxLevel()))
		{
		    LBERROR<<"Error: creating octree in node"<<std::endl;
		    _lock.unset();
		    return false;
		}
	    }
	    
	    _lock.unset();
	    return true;
	}

	index_node_t **	Node::getOctreePointer(int device)
	{
		return _octrees[device]->getOctree();
	}

	int *	Node::getOctreeSizesPointer(int device)
	{
		return _octrees[device]->getSizes();
	}

}
