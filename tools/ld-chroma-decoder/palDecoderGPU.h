


#include <QDebug>
#include <QObject>
#include <QScopedPointer>
#include <QVector>
#include <QtMath>


#include "sourcefield.h"
#include "lddecodemetadata.h"

#include <CL/sycl.hpp>

//these two structs are used to store data about each line on the GPU
struct LineInfo {

	qint32 number;
	double bp, bq;
	double Vsw;
};


struct InInfo {

	const unsigned short* in0;
	const unsigned short* in1;
	const unsigned short* in2;
	const unsigned short* in3;
	const unsigned short* in4;
	const unsigned short* in5;
	const unsigned short* in6;

};



class DecodePAL {

public:
	DecodePAL(double sine[], double cosine[]);

	~DecodePAL();

void decodeFieldGPU(const SourceField &inputField, const SourceField &inputFieldTwo, const double *chromaData, double chromaGain, RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters, double sine[], double cosine[], double cfilt[][4], double yfilt[][2]);


private:

	cl::sycl::queue myQueue;

	//buffer storing info about the lien
	cl::sycl::buffer<LineInfo> bufLineInfo{cl::sycl::range<1>(576)};
	//buffer storing pointers to areas around a line
	cl::sycl::buffer<InInfo> bufInInfo{cl::sycl::range<1>(576)};

	//M and N buffers used to store chroma data
	cl::sycl::buffer<double, 3> bufM{cl::sycl::range<3>(4, 576, 1135)};
	cl::sycl::buffer<double, 3> bufN{cl::sycl::range<3>(4, 576, 1135)};

	//buffer to represent a black line
	cl::sycl::buffer<unsigned short> bufBlackLine{cl::sycl::range<1>(1135)};

	//buffers to store sine and cosine data
	cl::sycl::buffer<double> bufSine{cl::sycl::range<1>(1135)};
	cl::sycl::buffer<double> bufCosine{cl::sycl::range<1>(1135)};

	//buffers that were used for test in parelizing the analysing of the colour burst of each line
	//cl::sycl::buffer<double, 2> bufBurstPrecalcbp{cl::sycl::range<2>(576, 40)};
	//cl::sycl::buffer<double, 2> bufBurstPrecalcbq{cl::sycl::range<2>(576, 40)};
	//cl::sycl::buffer<double, 2> bufBurstPrecalcbpo{cl::sycl::range<2>(576, 40)};
	//cl::sycl::buffer<double, 2> bufBurstPrecalcbqo{cl::sycl::range<2>(576, 40)};

};



