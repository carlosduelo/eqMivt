/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "channel.h"

#include "initData.h"
#include "config.h"
#include "pipe.h"
#include "view.h"
#include "configEvent.h"

namespace eqMivt
{

Channel::Channel( eq::Window* parent )
        : eq::Channel( parent )
        , _frameRestart( 0 )
{
    _lastViewport.h = 0;
    _lastViewport.w = 0;
    _pbo = -1;
    _texture = -1;
}

bool Channel::configInit( const eq::uint128_t& initID )
{
    if( !eq::Channel::configInit( initID ))
        return false;

    setNearFar( 0.1f, 10.0f );

    return true;
}

bool Channel::configExit()
{
	for( size_t i = 0; i < eq::NUM_EYES; ++i )
	{
		delete _accum[ i ].buffer;
		_accum[ i ].buffer = 0;
	}

    return eq::Channel::configExit();
}

void Channel::frameClear( const eq::uint128_t& frameID )
{
	if( stopRendering( ))
		return;

	_initJitter();
	resetRegions();

	const FrameData& frameData = _getFrameData();
	const int32_t eyeIndex = lunchbox::getIndexOfLastBit( getEye() );
	if( _isDone() && !_accum[ eyeIndex ].transfer )
		return;

	applyBuffer();
	applyViewport();

	const eq::View* view = getView();
	if( view && frameData.getCurrentViewID() == view->getID( ))
		glClearColor( 1.f, 1.f, 1.f, 0.f );
#ifndef NDEBUG
	else if( getenv( "EQ_TAINT_CHANNELS" ))
	{
		const eq::Vector3ub color = getUniqueColor();
		glClearColor( color.r()/255.f, color.g()/255.f, color.b()/255.f, 0.f );
	}
#endif // NDEBUG
	else
		glClearColor( 0.f, 0.f, 0.f, 0.0f );

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
}

void Channel::frameDraw( const eq::uint128_t& frameID )
{
    if( stopRendering( ))
        return;

	_initJitter();
	if( _isDone( ))
		return;

    applyViewport();
    applyBuffer();

    Pipe* pipe = static_cast<Pipe*>( getPipe( ));

    std::cout<<getName()<<" Device: "<<pipe->getDevice()<<std::endl;
    std::cout<<getName()<<" Port: "<<pipe->getPort()<<std::endl;
    
	Render * render = pipe->getRender();
	if (render == 0)
		return;

    // Check viewport
    const eq::PixelViewport& pvp = getPixelViewport();
    if (pvp.w != _lastViewport.w || pvp.h != _lastViewport.h)
    {
        _lastViewport.w = pvp.w;
		_lastViewport.h = pvp.h;

		_destroyPBO();
		_destroyTexture();
		_createPBO();
		_createTexture();

		render->resizeViewport(_lastViewport.w, _lastViewport.h, _pbo);
    }

    const FrameData& frameData = _getFrameData();

    // Compute cull matrix
    const eq::Matrix4f& rotation = frameData.getCameraRotation();
    eq::Matrix4f positionM = eq::Matrix4f::IDENTITY;
    positionM.set_translation( frameData.getCameraPosition());

    const eq::Matrix4f model = getHeadTransform() * (positionM * rotation);

    const eq::Frustumf& frustum = getFrustum();
    eq::Vector4f pos;
    pos.set(0.0f, 0.0f, 0.0f, 1.0f);
    pos = model*pos;

    eq::Vector4f p1; p1.set(frustum.right(),frustum.bottom(),frustum.near_plane(),1.0f); p1 = model * p1; 
    eq::Vector4f p2; p2.set(frustum.right(),frustum.top(),frustum.near_plane(),1.0f);  p2 = model * p2;
    eq::Vector4f p3; p3.set(frustum.left(),frustum.top(),frustum.near_plane(),1.0f);  p3 = model * p3;
    eq::Vector4f p4; p4.set(frustum.left(),frustum.bottom(),frustum.near_plane(),1.0f);  p4 = model * p4;
    /************************
     *********FRUSTUM********
     ****p3------------p2****
     *****|             |****
     *****|             |****
     ****p4------------p1****
     ************************
    */

    eq::Vector4f up = p3 - p4;
    eq::Vector4f right = p1 - p4;
    up.normalize();
    right.normalize();
    float w = frustum.get_width()/(float)pvp.w;
    float h = frustum.get_height()/(float)pvp.h;

    //render_sphere(_pbo, pvp.w, pvp.h, pos.x(), pos.y(), pos.z(), p4.x(), p4.y(), p4.z(), up.x(), up.y(), up.z(), right.x(), right.y(), right.z(), w, h);

	render->frameDraw(pos, p4, up, right, w, h, pvp.w, pvp.h, getJitter());

    _draw();

	Accum& accum = _accum[ lunchbox::getIndexOfLastBit( getEye()) ];
	accum.stepsDone = LB_MAX( accum.stepsDone,
			getSubPixel().size * getPeriod( ));
	accum.transfer = true;
}

void Channel::frameAssemble( const eq::uint128_t& frameID )
{
	if( stopRendering( ))
		return;

	if( _isDone( ))
		return;

	Accum& accum = _accum[ lunchbox::getIndexOfLastBit( getEye()) ];

	if( getPixelViewport() != _currentPVP)
	{
		accum.transfer = true;

		if( accum.buffer && !accum.buffer->usesFBO( ))
		{
			LBWARN << "Current viewport different from view viewport, "
				<< "idle anti-aliasing not implemented." << std::endl;
			accum.step = 0;
		}

		eq::Channel::frameAssemble( frameID );
		return;
	}
	// else

	accum.transfer = true;
	const eq::Frames& frames = getInputFrames();

	for( eq::Frames::const_iterator i = frames.begin(); i != frames.end(); ++i )
	{
		eq::Frame* frame = *i;
		const eq::SubPixel& curSubPixel = frame->getSubPixel();

		if( curSubPixel != eq::SubPixel::ALL )
			accum.transfer = false;

		accum.stepsDone = LB_MAX( accum.stepsDone, frame->getSubPixel().size *
				frame->getPeriod( ));
	}

	applyBuffer();
	applyViewport();
	setupAssemblyState();

	try
	{
		eq::Compositor::assembleFrames( getInputFrames(), this, accum.buffer );
	}
	catch( const co::Exception& e )
	{
		LBWARN << e.what() << std::endl;
	}

	resetAssemblyState();
}

void Channel::frameReadback( const eq::uint128_t& frameID )
{
	if( stopRendering() || _isDone( ))
		return;

	const FrameData& frameData = _getFrameData();
	const eq::Frames& frames = getOutputFrames();
	for( eq::FramesCIter i = frames.begin(); i != frames.end(); ++i )
	{
		eq::Frame* frame = *i;
		// OPT: Drop alpha channel from all frames during network transport
		frame->setAlphaUsage( false );

		if( frameData.isIdle( ))
			frame->setQuality( eq::Frame::BUFFER_COLOR, 1.f );
		else
			frame->setQuality( eq::Frame::BUFFER_COLOR, 1.f );
			//frame->setQuality( eq::Frame::BUFFER_COLOR, frameData.getQuality());

		//if( frameData.useCompression( ))
			frame->useCompressor( eq::Frame::BUFFER_COLOR, EQ_COMPRESSOR_AUTO );
		//else
		//	frame->useCompressor( eq::Frame::BUFFER_COLOR, EQ_COMPRESSOR_NONE );
	}

	eq::Channel::frameReadback( frameID );
}

void Channel::frameStart( const eq::uint128_t& frameID,	const uint32_t frameNumber )
{
	if( stopRendering( ))
		return;

	for( size_t i = 0; i < eq::NUM_EYES; ++i )
		_accum[ i ].stepsDone = 0;

	eq::Channel::frameStart( frameID, frameNumber );
}

void Channel::frameViewStart( const eq::uint128_t& frameID )
{
	if( stopRendering( ))
		return;

	_currentPVP= getPixelViewport();
	_initJitter();
	eq::Channel::frameViewStart( frameID );
}

void Channel::frameFinish( const eq::uint128_t& frameID,const uint32_t frameNumber )
{
	if( stopRendering( ))
		return;

	for( size_t i = 0; i < eq::NUM_EYES; ++i )
	{
		Accum& accum = _accum[ i ];
		if( accum.step > 0 )
		{
			if( int32_t( accum.stepsDone ) > accum.step )
				accum.step = 0;
			else
				accum.step -= accum.stepsDone;
		}
	}

	eq::Channel::frameFinish( frameID, frameNumber );
}

void Channel::frameViewFinish( const eq::uint128_t& frameID )
{
	if( stopRendering( ))
		return;

	applyBuffer();

	const FrameData& frameData = _getFrameData();
	Accum& accum = _accum[ lunchbox::getIndexOfLastBit( getEye()) ];

	if( accum.buffer )
	{
		const eq::PixelViewport& pvp = getPixelViewport();
		const bool isResized = accum.buffer->resize( pvp.w, pvp.h );

		if( isResized )
		{
			const View* view = static_cast< const View* >( getView( ));
			accum.buffer->clear();
			accum.step = view->getIdleSteps();
			accum.stepsDone = 0;
		}
		else if( frameData.isIdle( ))
		{
			setupAssemblyState();

			if( !_isDone() && accum.transfer )
				accum.buffer->accum();
			accum.buffer->display();

			resetAssemblyState();
		}
	}

	applyViewport();
//	_drawOverlay();
//	_drawHelp();

//	if( frameData.useStatistics())
//		drawStatistics();

	int32_t steps = 0;
	if( frameData.isIdle( ))
	{
		for( size_t i = 0; i < eq::NUM_EYES; ++i )
			steps = LB_MAX( steps, _accum[i].step );
	}
	else
	{
		const View* view = static_cast< const View* >( getView( ));
		steps = view ? view->getIdleSteps() : 0;
	}

	// if _jitterStep == 0 and no user redraw event happened, the app will exit
	// FSAA idle mode and block on the next redraw event.
	eq::Config* config = getConfig();
	config->sendEvent( IDLE_AA_LEFT ) << steps;

	eq::Channel::frameViewFinish( frameID );
}

void Channel::_draw()
{
    glEnable( GL_TEXTURE_2D );

    glBindTexture( GL_TEXTURE_2D, _texture );
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _lastViewport.w, _lastViewport.h, GL_RGB, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBegin(GL_QUADS);
	glTexCoord2f(0.0f,0.0f); glVertex2f(-1.0f,-1.0f);
	glTexCoord2f(1.0f,0.0f); glVertex2f( 1.0f,-1.0f);
	glTexCoord2f(1.0f,1.0f); glVertex2f( 1.0f, 1.0f);
	glTexCoord2f(0.0f,1.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

void Channel::_createPBO()
{
    // Creating pbo
    // create pixel buffer object for display
    glGenBuffers(1, &_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, _lastViewport.w*_lastViewport.h*sizeof(float)*3, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void Channel::_createTexture()
{
    // allocate a texture name
    glGenTextures( 1, &_texture );
    // select our current texture
    glBindTexture( GL_TEXTURE_2D, _texture );
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, _lastViewport.w, _lastViewport.h, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);  //Always set the base and max mipmap levels of a texture.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);
}
void Channel::_destroyPBO()
{
    glDeleteBuffers(1, &_pbo);
}
void Channel::_destroyTexture()
{
    glDeleteTextures( 1, &_texture );
}


const FrameData& Channel::_getFrameData() const
{
    const Pipe* pipe = static_cast<const Pipe*>( getPipe( ));
    return pipe->getFrameData();
}

bool Channel::stopRendering() const
{
    return getPipe()->getCurrentFrame() < _frameRestart;
}

bool Channel::_initAccum()
{
	View* view = static_cast< View* >( getNativeView( ));
	if( !view ) // Only alloc accum for dest
		return true;

	const eq::Eye eye = getEye();
	Accum& accum = _accum[ lunchbox::getIndexOfLastBit( eye ) ];

	if( accum.buffer ) // already done
		return true;

	if( accum.step == -1 ) // accum init failed last time
		return false;

	// Check unsupported cases
	if( !eq::util::Accum::usesFBO( glewGetContext( )))
	{
		for( size_t i = 0; i < eq::NUM_EYES; ++i )
		{
			if( _accum[ i ].buffer )
			{
				LBWARN << "glAccum-based accumulation does not support "
					<< "stereo, disabling idle anti-aliasing."
					<< std::endl;
				for( size_t j = 0; j < eq::NUM_EYES; ++j )
				{
					delete _accum[ j ].buffer;
					_accum[ j ].buffer = 0;
					_accum[ j ].step = -1;
				}

				view->setIdleSteps( 0 );
				return false;
			}
		}
	}

	// set up accumulation buffer
	accum.buffer = new eq::util::Accum( glewGetContext( ));
	const eq::PixelViewport& pvp = getPixelViewport();
	LBASSERT( pvp.isValid( ));

	if( !accum.buffer->init( pvp, getWindow()->getColorFormat( )) ||
			accum.buffer->getMaxSteps() < 256 )
	{
		LBWARN <<"Accumulation buffer initialization failed, "
			<< "idle AA not available." << std::endl;
		delete accum.buffer;
		accum.buffer = 0;
		accum.step = -1;
		return false;
	}

	// else
	LBVERB << "Initialized "
		<< (accum.buffer->usesFBO() ? "FBO accum" : "glAccum")
		<< " buffer for " << getName() << " " << getEye()
		<< std::endl;

	view->setIdleSteps( accum.buffer ? 256 : 0 );
	return true;
}

bool Channel::_isDone() const
{
	const FrameData& frameData = _getFrameData();
	if( !frameData.isIdle( ))
		return false;

	const eq::SubPixel& subpixel = getSubPixel();
	const Accum& accum = _accum[ lunchbox::getIndexOfLastBit( getEye()) ];
	return int32_t( subpixel.index ) >= accum.step;
}

void Channel::_initJitter()
{
	if( !_initAccum( ))
		return;

	const FrameData& frameData = _getFrameData();
	if( frameData.isIdle( ))
		return;

	const View* view = static_cast< const View* >( getView( ));
	if( !view )
		return;

	const int32_t idleSteps = view->getIdleSteps();
	if( idleSteps == 0 )
		return;

	// ready for the next FSAA
	Accum& accum = _accum[ lunchbox::getIndexOfLastBit( getEye()) ];
	if( accum.buffer )
		accum.buffer->clear();
	accum.step = idleSteps;
}

static const uint32_t _primes[100] = {
	739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829,
	839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
	947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033,
	1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109,
	1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213,
	1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291,
	1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399,
	1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451 };

eq::Vector2i Channel::_getJitterStep() const
{
	const eq::SubPixel& subPixel = getSubPixel();
	const uint32_t channelID = subPixel.index;
	const View* view = static_cast< const View* >( getView( ));
	if( !view )
		return eq::Vector2i::ZERO;

	const uint32_t totalSteps = uint32_t( view->getIdleSteps( ));
	if( totalSteps != 256 )
		return eq::Vector2i::ZERO;

	const Accum& accum = _accum[ lunchbox::getIndexOfLastBit( getEye()) ];
	const uint32_t subset = totalSteps / getSubPixel().size;
	const uint32_t index = ( accum.step * _primes[ channelID % 100 ] )%subset +
		( channelID * subset );
	const uint32_t sampleSize = 16;
	const int dx = index % sampleSize;
	const int dy = index / sampleSize;

	return eq::Vector2i( dx, dy );
}

eq::Vector2f Channel::getJitter() const
{
	const FrameData& frameData = _getFrameData();
	const Accum& accum = _accum[ lunchbox::getIndexOfLastBit( getEye()) ];

	if( !frameData.isIdle() || accum.step <= 0 )
		return eq::Channel::getJitter();

	const View* view = static_cast< const View* >( getView( ));
	if( !view || view->getIdleSteps() != 256 )
		return eq::Vector2f::ZERO;

	const eq::Vector2i jitterStep = _getJitterStep();
	if( jitterStep == eq::Vector2i::ZERO )
		return eq::Vector2f::ZERO;

	const eq::PixelViewport& pvp = getPixelViewport();
	const float pvp_w = float( pvp.w );
	const float pvp_h = float( pvp.h );
	const float frustum_w = float(( getFrustum().get_width( )));
	const float frustum_h = float(( getFrustum().get_height( )));

	const float pixel_w = frustum_w / pvp_w;
	const float pixel_h = frustum_h / pvp_h;

	const float sampleSize = 16.f; // sqrt( 256 )
	const float subpixel_w = pixel_w / sampleSize;
	const float subpixel_h = pixel_h / sampleSize;

	// Sample value randomly computed within the subpixel
	lunchbox::RNG rng;
	const eq::Pixel& pixel = getPixel();

	const float i = ( rng.get< float >() * subpixel_w +
			float( jitterStep.x( )) * subpixel_w ) / float( pixel.w );
	const float j = ( rng.get< float >() * subpixel_h +
			float( jitterStep.y( )) * subpixel_h ) / float( pixel.h );

	return eq::Vector2f( i, j );
}

}
