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

    const FrameData& frameData = _getFrameData();

    // Compute cull matrix
    const eq::Matrix4f& rotation = frameData.getCameraRotation();
    eq::Matrix4f positionM = eq::Matrix4f::IDENTITY;
    positionM.set_translation( frameData.getCameraPosition());

    const eq::Matrix4f model = getHeadTransform() * (positionM * rotation);
    std::cout<<model<<std::endl;

    const eq::Frustumf& frustum = getFrustum();
    eq::Vector4f pos;
    pos.set(0.0f, 0.0f, 0.0f, 1.0f);
    std::cout<<"position camera "<< model * pos<<std::endl;
    eq::Vector4f p1; p1.set(frustum.right(),frustum.bottom(),frustum.near_plane(),1.0f); p1 = model * p1; 
    eq::Vector4f p2; p2.set(frustum.right(),frustum.top(),frustum.near_plane(),1.0f);  p2 = model * p2;
    eq::Vector4f p3; p3.set(frustum.left(),frustum.top(),frustum.near_plane(),1.0f);  p3 = model * p3;
    eq::Vector4f p4; p4.set(frustum.left(),frustum.bottom(),frustum.near_plane(),1.0f);  p4 = model * p4;
    std::cout<<p1<<std::endl;
    std::cout<<p2<<std::endl;
    std::cout<<p3<<std::endl;
    std::cout<<p4<<std::endl;

    const eq::PixelViewport& pvp = getPixelViewport();
if(pvp.x <= 256 && pvp.y<=256)
{
    glColor3f(1.0f,1.0f,1.0f);
    glLineWidth(1); 
    glBegin(GL_QUADS);
	glVertex2f(p1.x(),p1.y()); 
	glVertex2f(p2.x(),p2.y()); 
	glVertex2f(p3.x(),p3.y()); 
	glVertex2f(p4.x(),p4.y()); 
    glEnd();
    glColor3f(1.0f,0.0f,0.0f);
    glBegin(GL_LINES);
	glVertex2f(pos.x(),pos.y()); 
	glVertex2f(p1.x(),p1.y()); 
    glEnd();
    glBegin(GL_LINES);
	glVertex2f(pos.x(),pos.y()); 
	glVertex2f(p2.x(),p2.y()); 
    glEnd();
    glBegin(GL_LINES);
	glVertex2f(pos.x(),pos.y()); 
	glVertex2f(p3.x(),p3.y()); 
    glEnd();
    glBegin(GL_LINES);
	glVertex2f(pos.x(),pos.y()); 
	glVertex2f(p4.x(),p4.y()); 
    glEnd();

    #if 0
    eq::Vector4f ray;
    eq::Vector4f up = p3 - p4;
    eq::Vector4f right = p1 - p4;
    float h = frustum.get_width()/pvp.h;
    float w = frustum.get_height()/pvp.w;

    for(int i=0; i<pvp.w; i++)
    {
    	for(int j=0; j<pvp.h; j++)
	{
	    ray = j*h*up + w*i*right;
	    glBegin(GL_LINES);
		glVertex2f(pos.x(),pos.y()); 
		glVertex2f(pos.x() + 10.0f*ray.x(),pos.y() + 10.0f*ray.y()); 
	    glEnd();
	    break;
	}
    }
    #endif

}
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
