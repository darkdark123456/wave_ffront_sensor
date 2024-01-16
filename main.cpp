#include "WaveFrontSensor.h"
#include <QtWidgets/QApplication>





int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    WaveFrontSensor w;
    w.show();
    return a.exec();
}
