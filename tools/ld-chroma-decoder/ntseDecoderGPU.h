
#include <QDebug>
#include <QObject>
#include <QScopedPointer>
#include <QVector>
#include <QtMath>


#include "sourcefield.h"
#include "lddecodemetadata.h"

#include <CL/sycl.hpp>

     struct Configuration {
         double chromaGain = 1.0;
         bool colorlpf = false;
         bool colorlpf_hq = true;
         bool whitePoint75 = false;
         qint32 dimensions = 2;
         bool adaptive = true;
         bool showMap = false;

         double cNRLevel = 0.0;
         double yNRLevel = 1.0;

         qint32 getLookBehind() const;
         qint32 getLookAhead() const;
     };


class DecodeNTSC {

public:

	DecodeNTSC();

	~DecodeNTSC();


	void decodeFrameGPU(const SourceField &inputFieldOne, const SourceField &inputFieldTwo, RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters, double yNRLevel, double irescale, double chromaGain, bool whitePoint75);

//void decodeFrameGPU(RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters);

private:

	cl::sycl::buffer<double, 2> bufClpBuffer2D{cl::sycl::range<2>(525, 910)};


	cl::sycl::queue myQueue;

};


