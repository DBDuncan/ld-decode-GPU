


#include <QDebug>
#include <QObject>
#include <QScopedPointer>
#include <QVector>
#include <QtMath>


#include "sourcefield.h"
#include "lddecodemetadata.h"

#include <CL/sycl.hpp>


struct LineInfo {
	//explicit LineInfo(qint32 number);

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




//void decodeFieldGPU(const SourceField &inputField, const double *chromaData, double chromaGain, RGBFrame &outputFrame);

class DecodePAL {

public:
	DecodePAL();

	~DecodePAL();



void decodeFieldGPU(const SourceField &inputField, const SourceField &inputFieldTwo, const double *chromaData, double chromaGain, RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters, double sine[], double cosine[], double cfilt[][4], double yfilt[][2]);


private:

	cl::sycl::buffer<LineInfo> bufLineInfo{cl::sycl::range<1>(576)};

	cl::sycl::buffer<InInfo> bufInInfo{cl::sycl::range<1>(576)};

	cl::sycl::buffer<double, 3> bufM{cl::sycl::range<3>(4, 1135, 576)};
	cl::sycl::buffer<double, 3> bufN{cl::sycl::range<3>(4, 1135, 576)};

	cl::sycl::buffer<unsigned short> bufBlackLine{cl::sycl::range<1>(1135)};


};



