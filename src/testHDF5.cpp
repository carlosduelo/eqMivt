#include "fileFactory.h"
#include <lunchbox/clock.h>
#include <iostream>
#include <fstream>
#include "memoryCheck.h"
#include "mortonCodeUtil_CPU.h"

int main(int argc, char ** argv)
{


	std::string type_file = argv[1];
	std::vector<std::string> file_params;
	file_params.push_back(argv[2]);
	file_params.push_back(argv[3]);

	eqMivt::FileManager * file = eqMivt::CreateFileManage(type_file, file_params);


	vmml::vector<3, int> realDim = file->getRealDimension();
	/* Calcular dimension del árbol*/
	int dimension = fmax(realDim[0], fmax(realDim[1], realDim[2]));
	float aux = logf(dimension)/logf(2.0);
	float aux2 = aux - floorf(aux);
	int nLevels = aux2>0.0 ? aux+1 : aux;
	std::cout<<"Dimension "<<realDim<<" nLevels "<<nLevels<<std::endl;
	
	double memoryCPU = eqMivt::getMemorySize();
	int iL = 0;
	if (memoryCPU == 0)
	{
		std::cerr<<"Not possible, check memory aviable (the call failed due to OS limitations)"<<std::endl;
		iL = nLevels > 10 ? nLevels - 10 : 0;
	}
	else
	{
		for(int l=0; l <= nLevels; l++)
		{
			double bigCubeDim = pow(pow(2, nLevels - l), 3)*(float)sizeof(float);
			if ((bigCubeDim) < memoryCPU)
			{
				iL = l;
				break;
			}
		}
	}
	std::cout<<"Start in level "<< iL <<" dimension "<<pow(2, nLevels - iL)<<std::endl;
	std::cout<<std::endl;

	double max[nLevels-iL+1];
	double min[nLevels-iL+1];
	double med[nLevels-iL+1];

	int k=0;
	for(int l = iL; l<=nLevels; l++)
	{
		int d = pow(2, nLevels - iL);
		std::cout<<"Level "<< iL <<" dimension "<<pow(2, nLevels - iL)<<std::endl;
		eqMivt::index_node_t idStart = eqMivt::coordinateToIndex(vmml::vector<3, int>(0,0,0), l, nLevels);
		eqMivt::index_node_t idFinish = eqMivt::coordinateToIndex(vmml::vector<3, int>(dimension-1, dimension-1, dimension-1), l, nLevels);
		std::cout<<"Index "<<idStart<<" to "<<idFinish<<std::endl;

		float * data = new float[d*d*d];

		med[k] = 0;
		for(eqMivt::index_node_t id =idStart; id<=idFinish; id++)
		{
			lunchbox::Clock	time;
			time.reset();
			file->readCube(id, data, l, nLevels, vmml::vector<3, int>(d,d,d), vmml::vector<3, int>(0,0,0), vmml::vector<3, int>(d,d,d));
			double bd = (pow(d,3)/1024.0/1024.0)/(time.getTimed()/1000.0);
			if (bd > max[k])
				max[k] = bd;
			if (bd < min[k])
				min[k] = bd;
			med[k] += bd;
		}
		med[k] = med[k]/(double)(idFinish-idStart);

		k++;
		delete[] data;
	}

	k=0;
	for(int l = iL; l<=nLevels; l++)
	{
		std::cout<<"For level "<< iL <<" dimension "<<pow(2, nLevels - iL)<<std::endl; 
		std::cout<<"\t Max "<<max[k]<<" MB/s min "<<min[k]<<" MB/s average "<<med[k]<<" MB/s"<<std::endl;
		k++;
	}


	delete file;

	return 0;
}

