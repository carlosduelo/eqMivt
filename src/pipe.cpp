/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "pipe.h"

#include "config.h"
#include "node.h"
#include <eq/eq.h>

namespace eqMivt
{
bool Pipe::configInit( const eq::uint128_t& initID )
{
    if( !eq::Pipe::configInit( initID ))
        return false;

    Config*         config      = static_cast<Config*>( getConfig( ));
    const InitData& initData    = config->getInitData();
    const eq::uint128_t&  frameDataID = initData.getFrameDataID();

    return config->mapObject( &_frameData, frameDataID );
}

bool Pipe::configExit()
{
    eq::Config* config = getConfig();
    config->unmapObject( &_frameData );

    return eq::Pipe::configExit();
}

void Pipe::frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber)
{
    eq::Pipe::frameStart( frameID, frameNumber );
    _frameData.sync( frameID );
}

Render * Pipe::getRender()
{
    Node*       node = static_cast<Node*>( getNode( ));
    // Check for CUDA RESOURCES
    if (!_render.checkCudaResources())
    {
        if (!node->registerPipeResources(getDevice()))
        {
    	    LBERROR<<"Error creating pipe"<<std::endl;
    	    return 0;
        }
		_render.setCudaResources(node->getOctreeContainer(getDevice()), node->getCubeCache(getDevice()), node->getNewId());
    }

	return &_render;
}

}
