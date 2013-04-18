/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_TYPEDEF_H
#define EQ_MIVT_TYPEDEF_H

namespace eqMivt
{

/* indentifier type for octree's node */
typedef unsigned long long index_node_t;

typedef struct
{
	index_node_t 	id;
	float *			data;
	unsigned char   state;
	index_node_t	cubeID;
} visibleCube_t;

#define CUBE		(unsigned char)8
#define PAINTED 	(unsigned char)4
#define CACHED 		(unsigned char)2
#define NOCACHED 	(unsigned char)1
#define NOCUBE		(unsigned char)0

}
#endif /* EQ_MIVT_TYPEDEF_H */
