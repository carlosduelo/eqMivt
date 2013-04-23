/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_RENDER_H
#define EQ_MIVT_RENDER_H

#include "typedef.h"

#include "octree.h"
#include "cubeCache.h"
#include "rayCaster.h"

#include <cuda_gl_interop.h> 
#include "cuda_runtime.h"

namespace eqMivt
{

    class Render 
    {
	private:

		int _id;
	
	    bool _init;

		Octree		_octree;
		cubeCache * _cache;
		rayCaster	_raycaster;

	    int  _height;
	    int  _width;
	  
        visibleCube_t * _visibleCubesGPU;
        visibleCube_t * _visibleCubesCPU;

  	    struct cudaGraphicsResource * _cuda_pbo_resource;
    
   	    cudaStream_t _stream;

	    void _CreateVisibleCubes();
	    void _DestroyVisibleCubes();

	public:
	    Render();

	    ~Render();

	    void resizeViewport(int width, int height, GLuint pbo);

	    bool checkCudaResources();

	    void setCudaResources(OctreeContainer * oc, cubeCache * cc, int id);

		void frameDraw(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH);
    };

}
#endif /*EQ_MIVT_RENDER_H*/
