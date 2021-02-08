


#include <QDebug>
#include <QObject>
#include <QScopedPointer>
#include <QVector>
#include <QtMath>


#include "sourcefield.h"
#include "lddecodemetadata.h"



void decodeFieldGPU(const SourceField &inputField, const double *chromaData, double chromaGain, RGBFrame &outputFrame);

void decodeFieldGPU(const SourceField &inputField, const double *chromaData, double chromaGain, RGBFrame &outputFrame, const LdDecodeMetaData::VideoParameters &videoParameters, double sine[], double cosine[]);
