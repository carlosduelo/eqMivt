/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <linkedList.h>


namespace eqMivt
{
LinkedList::LinkedList(int size)
{
	freePositions 	= size;
	memoryList 	= new NodeLinkedList[size];
	list 		= memoryList;
	last 		= &memoryList[size-1];

	for(int i=0; i<size; i++)
	{
		if (i==0)
		{
			memoryList[i].after 		= &memoryList[i+1];
			memoryList[i].before 		= 0;
			memoryList[i].element 		= i;
			memoryList[i].cubeID 		= 0;
			memoryList[i].references 	= 0;
		}
		else if (i==size-1)
		{
			memoryList[i].after 		= 0;
			memoryList[i].before 		= &memoryList[i-1];
			memoryList[i].element 		= i;
			memoryList[i].cubeID 		= 0;
			memoryList[i].references 	= 0;
		}
		else
		{
			memoryList[i].after 		= &memoryList[i+1];
			memoryList[i].before 		= &memoryList[i-1];
			memoryList[i].element 		= i;
			memoryList[i].cubeID 		= 0;
			memoryList[i].references 	= 0;
		}
	}
}

LinkedList::~LinkedList()
{
	delete[] memoryList;
}


NodeLinkedList * LinkedList::getFirstFreePosition(index_node_t newIDcube, index_node_t * removedIDcube)
{
	if (freePositions > 0)
	{
		NodeLinkedList * first = list;

		// Search first free position
		while(first->references != 0)
		{
			moveToLastPosition(first);
			first = list;
		}

		list = first->after;
		list->before = 0;
		
		first->after  = 0;
		first->before = last;
		
		last->after = first;
		
		last = first;
		*removedIDcube = last->cubeID;
		last->cubeID = newIDcube;

		return first;
	}

	return 0;
}

NodeLinkedList * LinkedList::moveToLastPosition(NodeLinkedList * node)
{
	if (node->before == 0)
	{
		NodeLinkedList * first = list;

		list = first->after;
		list->before = 0;
		
		first->after  = 0;
		first->before = last;
		
		last->after = first;
		
		last = first;

		return first;
	}
	else if (node->after == 0)
	{
		return node;
	}
	else
	{
		node->before->after = node->after;
		node->after->before = node->before;
		
		last->after = node;
		
		node->before = last;
		node->after  = 0;
		last = node;
		
		return node;
	}
}

void	LinkedList::removeReference(NodeLinkedList * node)
{
	if (node->references > 0)
	{
		node->references--;//&= ~(ref);

		if (node->references == 0)
			freePositions++;
	}
}

void 	LinkedList::addReference(NodeLinkedList * node)
{
	if (node->references == 0)
		freePositions--;

	node->references++;// |= ref;
}
}
