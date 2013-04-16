/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "channel.h"

#include "initData.h"
#include "config.h"
#include "pipe.h"
#include "node.h"

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
    return eq::Channel::configExit();
}

void Channel::frameDraw( const eq::uint128_t& frameID )
{
    if( stopRendering( ))
        return;

    const Pipe* pipe = static_cast<const Pipe*>( getPipe( ));
    Node*       node = static_cast<Node*>( getNode( ));

    std::cout<<getName()<<" Device: "<<pipe->getDevice()<<std::endl;
    std::cout<<getName()<<" Port: "<<pipe->getPort()<<std::endl;
    
    // Check for CUDA RESOURCES
    if (_render.checkCudaResources())
    {
        if (!node->registerPipeResources(pipe->getDevice()))
        {
    	    LBERROR<<"Error creating pipe"<<std::endl;
    	    return;
        }
	_render.setCudaResources(node->getOctreePointer(pipe->getDevice()), node->getOctreeSizesPointer(pipe->getDevice()));
    }

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

	_render.resizeViewport(_lastViewport.w, _lastViewport.h, _pbo);
    }

    const FrameData& frameData = _getFrameData();

    // Compute cull matrix
    const eq::Matrix4f& rotation = frameData.getCameraRotation();
    eq::Matrix4f positionM = eq::Matrix4f::IDENTITY;
    positionM.set_translation( frameData.getCameraPosition());

    const eq::Matrix4f model = getHeadTransform() * (positionM * rotation);
    //std::cout<<"ModelView"<<std::endl<<model<<std::endl;

    const eq::Frustumf& frustum = getFrustum();
    eq::Vector4f pos;
    pos.set(0.0f, 0.0f, 0.0f, 1.0f);
    pos = model*pos;
    //std::cout<<"position camera "<< pos<<std::endl;
    eq::Vector4f p1; p1.set(frustum.right(),frustum.bottom(),frustum.near_plane(),1.0f); p1 = model * p1; 
    eq::Vector4f p2; p2.set(frustum.right(),frustum.top(),frustum.near_plane(),1.0f);  p2 = model * p2;
    eq::Vector4f p3; p3.set(frustum.left(),frustum.top(),frustum.near_plane(),1.0f);  p3 = model * p3;
    eq::Vector4f p4; p4.set(frustum.left(),frustum.bottom(),frustum.near_plane(),1.0f);  p4 = model * p4;
    //std::cout<<p1<<std::endl;
    //std::cout<<p2<<std::endl;
    //std::cout<<p3<<std::endl;
    //std::cout<<p4<<std::endl;
    /************************
     *********FRUSTUM********
     ****p3------------p2****
     *****|             |****
     *****|             |****
     ****p4------------p1****
     ************************
    */

    std::cout<<pvp<<std::endl;

    eq::Vector4f up = p3 - p4;
    eq::Vector4f right = p1 - p4;
    up.normalize();
    right.normalize();
    float w = frustum.get_width()/(float)pvp.w;
    float h = frustum.get_height()/(float)pvp.h;

    render_sphere(_pbo, pvp.w, pvp.h, pos.x(), pos.y(), pos.z(), p4.x(), p4.y(), p4.z(), up.x(), up.y(), up.z(), right.x(), right.y(), right.z(), w, h);

    _draw();
}

void Channel::_draw()
{
    glEnable( GL_TEXTURE_2D );

    glBindTexture( GL_TEXTURE_2D, _texture );
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _lastViewport.w, _lastViewport.h, GL_RGB, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    applyViewport();
    applyBuffer();
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
}
