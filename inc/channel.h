/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CHANNEL_H
#define EQ_MIVY_CHANNEL_H

#include "eqMivt.h"

//#define GL_GLEXT_PROTOTYPES
#include "render.h"

#include <eq/eq.h>

#include "ray-casting-sphere.h"

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
		virtual void frameClear( const eq::uint128_t& frameID );
        virtual void frameDraw( const eq::uint128_t& frameID );
		virtual void frameAssemble( const eq::uint128_t& frameID );
		virtual void frameReadback( const eq::uint128_t& frameID );
		virtual void frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber );
		virtual void frameViewStart( const eq::uint128_t& frameID );
		virtual void frameFinish( const eq::uint128_t& frameID,const uint32_t frameNumber );
		virtual void frameViewFinish( const eq::uint128_t& frameID );

		virtual eq::Vector2f getJitter() const;

		virtual void notifyStopFrame( const uint32_t lastFrameNumber )
            { _frameRestart = lastFrameNumber + 1; }
    private:
        const FrameData& _getFrameData() const;

		GLuint _pbo;
		GLuint _texture;

		void _createPBO();
		void _createTexture();
		void _destroyPBO();
		void _destroyTexture();
		void _draw();

		uint32_t			_frameRestart;
		eq::PixelViewport 	_currentPVP;
		eq::PixelViewport 	_lastViewport;

		bool _initAccum();
		bool _isDone() const;
		void _initJitter();
		eq::Vector2i _getJitterStep() const;

		void _saveFrameBuffer(const eq::uint128_t& frameID);

		eq::Vector3i	_dimCube;
		void _drawCube();

		struct Accum
		{
			Accum() : buffer( 0 ), step( 0 ), stepsDone( 0 ), transfer( false )
			{}

			eq::util::Accum* buffer;
			int32_t step;
			uint32_t stepsDone;
			bool transfer;
		}
		_accum[ eq::NUM_EYES ];
    };
}

#endif // EQ_MIVT_CHANNEL_H

