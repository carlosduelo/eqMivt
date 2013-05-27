/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_OCTREE_CONSTRUCTOR_H
#define EQ_MIVT_OCTREE_CONSTRUCTOR_H

#include <vector> 
#include <string>


namespace eqMivt
{

 bool createOctree(std::string type_file, std::vector<std::string> file_params, int maxLevel, std::vector<float> isosurfaceList, std::string octree_file);

}

#endif /* EQ_MIVT_OCTREE_CONSTRUCTOR_H */
