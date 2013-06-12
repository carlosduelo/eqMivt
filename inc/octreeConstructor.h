/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_OCTREE_CONSTRUCTOR_H
#define EQ_MIVT_OCTREE_CONSTRUCTOR_H

#include <vector> 
#include <string>

#include "eq/eq.h"

namespace eqMivt
{

 bool createOctree(std::string type_file, std::vector<std::string> file_params, std::vector<int> maxLevel, std::vector< std::vector<float> > isosurfaceList, std::vector<int> numOctrees, std::vector< vmml::vector<3, int> > startCoordinates, std::vector< vmml::vector<3, int> > finishCoordinates, std::string octree_file, bool useCUDA);

}

#endif /* EQ_MIVT_OCTREE_CONSTRUCTOR_H */
