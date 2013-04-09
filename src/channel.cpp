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


bool intersection(const eq::Vector3f& ray, const eq::Vector3f& posR, float * root)
{
       eq::Vector3f posEs;
       posEs.set(0.0f, 0.0f, 0.0f);
       eq::Vector3f omc = posEs - posR;
       double b = omc.dot(ray);
       double disc = b*b - omc.dot(omc) + 10.0f*10.0f;
       // If the discriminant is less than 0, then we totally miss the sphere.
       // This happens most of the time.
       if (disc > 0)
       {
         double d = sqrt(disc);
         double root2 = b + d;
         double root1 = b - d;
         // If root2 < 0, then root1 is also < 0, so they are both misses.
         if (root2 > 0)
         {
           // If root2 > 0, and root1 < 0, we are inside the sphere.
           if(root1 < 0)
           {
             *root=root2; return true;
           // If root2 > 0, and root1 > 0, we are hit the sphere.
           } 
	   else 
	   {
             *root = root1; return true;
           }
         }
       }
       return false;
}

void Channel::frameDraw( const eq::uint128_t& frameID )
{
    if( stopRendering( ))
        return;

    const FrameData& frameData = _getFrameData();
	
#if 0
    eq::Channel::frameDraw( frameID );
    const eq::Vector3f& position = frameData.getCameraPosition();
    glMultMatrixf( frameData.getCameraRotation().array );
    glTranslatef( position.x(), position.y(), position.z() );
#endif

    // Compute cull matrix
    const eq::Matrix4f& rotation = frameData.getCameraRotation();
    eq::Matrix4f positionM = eq::Matrix4f::IDENTITY;
    positionM.set_translation( frameData.getCameraPosition());

    const eq::Matrix4f model = getHeadTransform() * (positionM * rotation);
    std::cout<<"ModelView"<<std::endl<<model<<std::endl;

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
    std::cout<<pvp<<std::endl;

#if 0
    glColor3f(1.0f,1.0f,1.0f);
    glLineWidth(1); 
    glBegin(GL_QUADS);
	glVertex3f(p1.x(),p1.y(),p1.z()); 
	glVertex3f(p2.x(),p2.y(),p2.z()); 
	glVertex3f(p3.x(),p3.y(),p3.z()); 
	glVertex3f(p4.x(),p4.y(),p4.z()); 
    glEnd();
    glColor3f(1.0f,0.0f,0.0f);
    glBegin(GL_LINES);
	glVertex3f(pos.x(),pos.y(),pos.z()); 
	glVertex3f(p1.x(),p1.y(),p1.z()); 
    glEnd();
    glBegin(GL_LINES);
	glVertex3f(pos.x(),pos.y(),pos.z()); 
	glVertex3f(p2.x(),p2.y(),p2.z()); 
    glEnd();
    glBegin(GL_LINES);
	glVertex3f(pos.x(),pos.y(),pos.z()); 
	glVertex3f(p3.x(),p3.y(),p3.z()); 
    glEnd();
    glBegin(GL_LINES);
	glVertex3f(pos.x(),pos.y(),pos.z()); 
	glVertex3f(p4.x(),p4.y(),p4.z()); 
    glEnd();
#endif
    eq::Vector4f ray;
    eq::Vector4f up = p3 - p4;
    eq::Vector4f right = p1 - p4;
    up.normalize();
    right.normalize();
    float w = frustum.get_width()/(float)pvp.w;
    float h = frustum.get_height()/(float)pvp.h;
    glColor3f(0.0f,1.0f,0.0f);

float * data = new float [3*pvp.h*pvp.w];
    for(int i=0; i<pvp.w; i++)
    {
    	for(int j=0; j<pvp.h; j++)
	{
    	    ray = p4 - pos;
	    ray = ray + j*h*up + i*w*right;
	    ray.normalize();
if (i==0 && j==0)
std::cout<<ray<<std::endl;
	    float hit = 1000.f;
	    int p = 3*i + 3*j*pvp.w;
            if (intersection(ray, pos, &hit))
	    {
	    	data[p]=1.0f;
		data[p+1]=0.0f;
		data[p+2]=0.0f;
	    }
	    else
	    {
	    	data[p]=1.0f;
		data[p+1]=1.0f;
		data[p+2]=1.0f;
	    }
	}
    }

glDrawPixels(pvp.w, pvp.h, GL_RGB, GL_FLOAT, data);
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
