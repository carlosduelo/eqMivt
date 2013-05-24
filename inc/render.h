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

#include <lunchbox/clock.h>

#include <cuda_gl_interop.h> 
#include "cuda_runtime.h"

namespace eqMivt
{

    class Render 
    {
	private:

		int _id;
		std::string	_name;
	
	    bool _init;

		Octree		_octree;
		cubeCache * _cache;
		rayCaster	_raycaster;

	    int  _height;
	    int  _width;
	  
        visibleCube_t * _visibleCubesGPU;
        visibleCube_t * _visibleCubesCPU;
		int	*			_indexVisibleCubesGPU;
		int	*			_indexVisibleCubesCPU;


  	    struct cudaGraphicsResource * _cuda_pbo_resource;
    
   	    cudaStream_t _stream;
		struct  cudaDeviceProp _cudaProp;

	    void _CreateVisibleCubes();
	    void _DestroyVisibleCubes();

		// Statistics
		std::ofstream *		_outputFile;
		bool	_statistics;
		lunchbox::Clock		_resizeClock;
		int					_resizeTimes;
		double				_resizeAccum;
		lunchbox::Clock		_mapResourcesClock;
		int					_mapResourcesTimes;
		double				_mapResourcesAccum;
		lunchbox::Clock		_unmapResourcesClock;
		int					_unmapResourcesTimes;
		double				_unmapResourcesAccum;
		lunchbox::Clock		_octreeClock;
		int					_octreeTimes;
		double				_octreeAccum;
		lunchbox::Clock		_rayCastingClock;
		int					_rayCastingTimes;
		double				_rayCastingAccum;
		lunchbox::Clock		_cachePushClock;
		int					_cachePushTimes;
		double				_cachePushAccum;
		lunchbox::Clock		_cachePopClock;
		int					_cachePopTimes;
		double				_cachePopAccum;
		lunchbox::Clock		_frameDrawClock;
		int					_frameDrawTimes;
		double				_frameDrawAccum;

	public:
	    Render();

	    ~Render();

	    void resizeViewport(int width, int height, GLuint pbo);

	    bool checkCudaResources();

		void setCudaResources(OctreeContainer * oc, cubeCache * cc, int id, std::string name);

		void setStatistics(bool stat);

		void printCudaProperties();

		void frameDraw(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH);
    };

}
#endif /*EQ_MIVT_RENDER_H*/
