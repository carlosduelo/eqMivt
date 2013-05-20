/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/



#ifndef EQ_MIVT_RAY_CASTING_SPHERE_H
#define EQ_MIVT_RAY_CASTING_SPHERE_H

#include <cuda_gl_interop.h> 

namespace eqMivt
{
	void render_sphere(GLuint pbo, int pvpW, int pvpH, float posx, float posy, float posz,  float LBx, float LBy, float LBz, float upx, float upy, float upz, float rightx, float righty, float rightz, float w, float h);
}

#endif /*EQ_MIVT_RAY_CASTING_SPHERE_H*/
