/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "channel.h"

#include "initData.h"
#include "config.h"
#include "pipe.h"

#include "eq/client/gl.h"

namespace eqMivt
{

Channel::Channel( eq::Window* parent )
        : eq::Channel( parent )
        , _frameRestart( 0 )
{
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
    return eq::Channel::configExit();
}

void Channel::frameDraw( const eq::uint128_t& frameID )
{
    if( stopRendering( ))
        return;
    eq::Channel::frameDraw( frameID ); // Setup OpenGL state

    const FrameData& frameData = _getFrameData();
    const eq::Vector3f& position = frameData.getCameraPosition();

    EQ_GL_CALL( glMultMatrixf( frameData.getCameraRotation().array ) );
    EQ_GL_CALL( glTranslatef( position.x(), position.y(), position.z() ) );

    eq::PixelViewport  viewport = getPixelViewport();

    std::cout<<getName()<<" "<<" .............>"<<viewport.x<<" "<<viewport.y<<" "<<viewport.h<<" "<<viewport.w<<std::endl;
    float modelview[16];

    std::cout<<position<<std::endl;

    EQ_GL_CALL( glGetFloatv(GL_MODELVIEW_MATRIX , modelview) );
    std::cout<<modelview[0]<<" "<<modelview[1]<<" "<<modelview[2]<<std::endl;
    std::cout<<modelview[4]<<" "<<modelview[5]<<" "<<modelview[6]<<std::endl;
    std::cout<<modelview[8]<<" "<<modelview[9]<<" "<<modelview[10]<<std::endl;

    const eq::Frustumf& frustum = getFrustum();
    std::cout<<frustum.left()<<" "<<frustum.bottom()<<std::endl;

#if 0
//if (viewport.x== 0 && viewport.y == 0)
{
    glLineWidth(1); 
	glBegin(GL_LINES); 
	glVertex2f(0.0f,0.0f); 
	glVertex2f(64.0f,64.0f); 
	//glVertex2f(frustum.left(),frustum.bottom() ); 
	//glVertex2f(frustum.right(),frustum.top() ); 
	glEnd(); 
}
#endif

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
}
