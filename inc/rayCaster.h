/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/
#ifndef _EQ_MIVT_RAY_CASTER_H_
#define _EQ_MIVT_RAY_CASTER_H_

#include "typedef.h"

#include <eq/eq.h>

#include "cuda_runtime.h"

namespace eqMivt
{

class rayCaster
{
	private:
		float 		_iso;
		float		_maxHeight;
		bool		_grid;

		vmml::vector<3, int> _offset;

		float * _r;
		float * _g;
		float * _b;

		// rayCasing Parameters
		float _step;
	public:
		rayCaster();

		~rayCaster();

		void setIsosurface(float isosurface);

		void setMaxHeight(float maxHeight);

		void setOffset(vmml::vector<3, int> offset);

		bool setColor(float * c);

		void setUseGrid(bool grid);

		void render(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, visibleCube_t * cube,int * indexCube, int3 cubeDim, int3 cubeInc, float * pixelBuffer, float * xGrid, float * yGrid, float * zGrid, vmml::vector<3, int> realDim, cudaStream_t stream);

		void renderCubes(eq::Vector4f origin, eq::Vector4f  LB, eq::Vector4f up, eq::Vector4f right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, visibleCube_t * cube,int * indexCube, int3 cubeDim, int3 cubeInc, float * pixelBuffer, float * xGrid, float * yGrid, float * zGrid, vmml::vector<3, int> realDim, cudaStream_t stream);
};
}

#endif
