/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_TYPEDEF_H
#define EQ_MIVT_TYPEDEF_H

/* indentifier type for octree's node */
typedef unsigned long long index_node_t;

typedef struct
{
	index_node_t 	id;
	float * 	data;
	unsigned char   state;
	index_node_t	cubeID;
} visibleCube_t;

#endif /* EQ_MIVT_TYPEDEF_H */
