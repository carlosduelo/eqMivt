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

bool intersection(const eq::Vector4f& ray, const eq::Vector4f& posR, float r, float * t)
{
	eq::Vector4f posE; posE.set(0.0f, 0.0f, 0.0f, 1.0f);
	eq::Vector4f d = posR -posE;
	//Compute A, B and C coefficients
	float a = ray.dot(ray);
	float b = 2 * ray.dot(d);
	float c = d.dot(d) - (r * r);

	//Find discriminant
	float disc = b * b - 4 * a * c;

	// if discriminant is negative there are no real roots, so return 
	// false as ray misses sphere
	if (disc < 0)
		return false;

	// compute q as described above
	float distSqrt = sqrtf(disc);
	float q;
	if (b < 0)
		q = (-b - distSqrt);
	else
		q = (-b + distSqrt);

	// compute t0 and t1
	float t0 = q / (2.0f*a);
	float t1 = q / (2.0f*a);

	// make sure t0 is smaller than t1
	if (t0 > t1)
	{
		// if t0 is bigger than t1 swap them around
		float temp = t0;
		t0 = t1;
		t1 = temp;
	}

	// if t1 is less than zero, the object is in the ray's negative direction
	// and consequently the ray misses the sphere
	if (t1 < 0)
		return false;

	// if t0 is less than zero, the intersection point is at t1
	if (t0 < 0)
	{
		*t = t1;
		return true;
	}
	// else the intersection point is at t0
	else
	{
		*t = t0;
		return true;
	}
}

void Channel::frameDraw( const eq::uint128_t& frameID )
{
    if( stopRendering( ))
        return;

    const FrameData& frameData = _getFrameData();
	
#if 1
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
    pos = model*pos;
    std::cout<<"position camera "<< pos<<std::endl;
    eq::Vector4f p1; p1.set(frustum.right(),frustum.bottom(),frustum.near_plane(),1.0f); p1 = model * p1; 
    eq::Vector4f p2; p2.set(frustum.right(),frustum.top(),frustum.near_plane(),1.0f);  p2 = model * p2;
    eq::Vector4f p3; p3.set(frustum.left(),frustum.top(),frustum.near_plane(),1.0f);  p3 = model * p3;
    eq::Vector4f p4; p4.set(frustum.left(),frustum.bottom(),frustum.near_plane(),1.0f);  p4 = model * p4;
    std::cout<<p1<<std::endl;
    std::cout<<p2<<std::endl;
    std::cout<<p3<<std::endl;
    std::cout<<p4<<std::endl;
    /************************
     *********FRUSTUM********
     ****p3------------p2****
     *****|             |****
     *****|             |****
     ****p4------------p1****
     ************************
    */

    const eq::PixelViewport& pvp = getPixelViewport();
    std::cout<<pvp<<std::endl;

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
	    float hit = 1000.f;
	    int p = 3*i + 3*j*pvp.w;
            if (intersection(ray, pos, 1.0f, &hit))
	    {
	    	eq::Vector3f ph = pos + hit * ray;
	    	eq::Vector3f n = ph/1.0f;
		n.normalize();
		eq::Vector3f l; l.set(pos.x() - ph.x(), pos.y() - ph.y(), pos.z() - ph.z());
		l.normalize();
		eq::Vector3f k = n.cross(l);
	        // Compute normal 
	    	data[p]=k.x()*0.3f;
		data[p+1]=k.y()*0.5f;
		data[p+2]=k.z()*0.3f;
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
