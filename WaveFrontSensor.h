#pragma once

#include    <QtWidgets/QWidget>
#include    "ui_WaveFrontSensor.h"
#include    <opencv.hpp>
#include    <qfile>
#include    <qfiledialog.h>
#include    <qtoolbutton.h>
#include    <qgridlayout.h>
#include    <qlabel.h>
#include    <qdialog.h>
#include    <qplaintextedit.h>
#include    <qtextstream.h>
#include    <future>
#include    <qmessagebox.h>
#include    "Configuration.h"
#include    "CustomException.h"
#include    "Algorithm.h"
#include    "TimeConsuming.h"


QT_BEGIN_NAMESPACE
namespace Ui { class WaveFrontSensorClass; };
QT_END_NAMESPACE

class WaveFrontSensor : public QWidget
{
    Q_OBJECT
public:
    typedef QVector<QVector<float>> QVector2D_;
    typedef QVector<QVector<QVector<float>>> QVector3D_;
    typedef QVector<QVector<QVector<QVector<float>>>> QVector4D_;
public:
    WaveFrontSensor(QWidget *parent = nullptr);
    ~WaveFrontSensor();
public:
    void initDefaultDisplay();
    void free();
public:
    void processHartmanngram(Configuration&, cv::Mat&, cv::Mat&, QString thresholding_model = "adaptive",
        int img_int_thr = 0, int area_thr = 1, float ration_thr = 0.05,
        int min_order_u = 0, int max_order_u = 0,
        int min_order_v = 0, int max_order_v = 0,
        int edge_exclusion = 1);
    void          readDataFromDisk(QString filename,QVector<float>& intensity);
    QVector2D_    readWaveFrontSensorImageDat(QString filename, float nu_detector, float nv_detector, int upsampleing);
    QVector2D_    addNoNoise(QVector2D_& intensity_map);
    void          CAT_DATA_INFO(const QVector4D_&,const QVector3D_&);
public:
    void TestFunction();


private:
    Ui::WaveFrontSensorClass *ui;
private:
    QSharedPointer<QToolButton>  select_data_tool_button_;
    QGridLayout* grid_layout_;
    QString                       filename_;
    QPlainTextEdit* infoText;
    QDialog* dialog;
};
