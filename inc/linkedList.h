/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_LINKED_LIST_H
#define EQ_MIVT_LINKED_LIST_H

#include <typedef.h>


namespace eqMivt
{
class NodeLinkedList
{
	public:
		NodeLinkedList * 	after;
		NodeLinkedList * 	before;
		unsigned int	 	element;
		index_node_t 	 	cubeID;
		int			references;
};

class LinkedList
{
	private:
		NodeLinkedList * 	list;
		NodeLinkedList * 	last;
		NodeLinkedList * 	memoryList;
		int			freePositions;
	public:
		LinkedList(int size);
		~LinkedList();

		/* pop_front and push_last */
		NodeLinkedList * 	getFirstFreePosition(index_node_t newIDcube, index_node_t * removedIDcube);

		NodeLinkedList * 	moveToLastPosition(NodeLinkedList * node);	

		void 			removeReference(NodeLinkedList * node);
		void 			addReference(NodeLinkedList * node);
};
}
#endif /*EQ_MIVT_LINKED_LIST_H*/
