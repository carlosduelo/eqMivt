/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CHANNEL_H
#define EQ_MIVY_CHANNEL_H

#include "eqMivt.h"

#include <eq/eq.h>


namespace eqMivt
{
    class FrameData;
    class InitData;

    class Channel : public eq::Channel
    {
    public:
        Channel( eq::Window* parent );

        bool stopRendering() const;

    protected:
        virtual ~Channel() {}

        virtual bool configInit( const eq::uint128_t& initID );
        virtual bool configExit();
        virtual void frameDraw( const eq::uint128_t& frameID );

    private:
        const FrameData& _getFrameData() const;

        uint32_t 		_frameRestart;
	eq::PixelViewport 	_lastViewport;
    };
}

#endif // EQ_MIVT_CHANNEL_H

