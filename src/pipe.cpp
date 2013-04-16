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
}
