
#include <QDebug>
#include <QObject>
#include <QScopedPointer>
#include <QVector>
#include <QtMath>


#include "sourcefield.h"
#include "lddecodemetadata.h"

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

void decodeFrameGPU(const SourceField &inputFieldOne, const SourceField &inputFieldTwo, RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters, QVector<quint16> rawbuffer, double yNRLevel, double irescale, double chromaGain, bool whitePoint75);

void decodeFrameGPU(RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters);
