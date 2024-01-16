#pragma once

/**
 *这是一个仿制软件所有的代码参考 https://github.com/huanglei0114/hartmann-analyzer.
 */
#include    <QtWidgets/QWidget>
#include    "ui_WaveFrontSensor.h"
#include    <opencv.hpp>
#include    <opencv2/opencv.hpp>
#include    <opencv2/core/utils/logger.hpp>
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
#include    <qgraphicsview.h>
#include    <qgraphicsscene.h>
#include    <qimage.h>
#include    <qpixmap.h>
#include    <qmath.h>
#include    <qmetatype.h>
#include    <qvariant.h>
#include    <qmap.h>
#include    <condition_variable>
#include    <cmath>
#include    <limits>

#include    "Configuration.h"
#include    "CustomException.h"
#include    "Algorithm.h"
#include    "phase.h"
#include    "ModalMethodCalculateModel.h"
#include    "Zernike.h"
#include    "Legendre.h"
#include    "OMDA_Algorithm.h"
#include    "matplotlibcpp.h"

namespace plt = matplotlibcpp;


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
    void processHartmanngram(    Configuration&, cv::Mat&, cv::Mat, HIOlAB_CXX_14::ThresholdMode thresholding_mode=HIOlAB_CXX_14::ThresholdMode::Adaptive,
                                 int img_int_thr = 0, int area_thr = 1, float ration_thr = 0.05,
                                 int* min_order_u = nullptr, int* max_order_u = nullptr,
                                 int* min_order_v = nullptr, int* max_order_v = nullptr,
                                 int edge_exclusion = 1);
public:
    void                         readDataFromDisk(QString filename,QVector<float>& intensity);
    QVector2D_                   readWaveFrontSensorImageDat(QString filename, float nu_detector, float nv_detector, int upsampleing);
public:
    void                         initInfoDialog();
    QVector<QVector<quint16>>    addNoNoise(QVector2D_&);
    void                         InitLoadInfo();
    void                         writreGaryscalePNG(const cv::Mat& );
    void                         setConfHartmanngramFilename(Configuration& );
public :
    void analyzeHartmanngram(    cv::Mat& hartmanngram, 
                                 cv::Mat&, cv::Mat&,cv::Mat&,cv::Mat& ,
                                 cv::Mat& ,cv::Mat&,cv::Mat&,cv::Mat&, 
                                 float dist_mask_to_detector, float pixel_size = 1.48e-6,
                                 float grid_period = 20e-6, HIOlAB_CXX_14::ThresholdMode thresholding_mode=HIOlAB_CXX_14::ThresholdMode::Adaptive,
                                 unsigned char img_int_thr=0,int block_size=31,int area_thr=1,
                                 int min_fringle_number=8,cv::Mat starting_pixel=cv::Mat(),
                                 float ration_thr=0.05,
                                 float centroid_power=1.7,
                                 int* min_order_u=nullptr,int* max_order_u=nullptr,
                                 int* min_order_v=nullptr,int* max_order_v=nullptr,
                                 int edge_exclusion=1,
                                 bool is_show=false);
public:
    void analyzeHartmannSlopes(  cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, float wave_length, cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&,
                                 QString str_method = QString("zonal"),ModalMethodCalculateModel str_model=ModalMethodCalculateModel::Legendre,
                                 int num_of_terms = 100,
                                 bool is_show = true);
public:
    void calculateWrappedPhasesUV(  cv::Mat& hartmanngram, cv::Mat& u2d_image, cv::Mat& v2d_image,
                                    int min_fringe_number,
                                    cv::Mat& u_wrapped_phase,cv::Mat& u_phase,
                                    cv::Mat& v_wrapped_phase,cv::Mat& v_phase);
public:

    void calculateWrappedPhase(     cv::Mat& spectrum,cv::Mat& u2d_img,cv::Mat& v2d_img,
                                    int min_fringe_number,cv::Mat& wrapped_phase,cv::Mat& amplitude,
                                    bool is_u = true);

    std::pair<double,double> calculateCentroid(cv::Mat&, cv::Mat&, cv::Mat&);

public:
    void cropImageWithFringeOrders( cv::Mat& img,
                                    cv::Mat& u2d_image,cv::Mat& v2d_image,
                                    cv::Mat& fringe_orders_u,cv::Mat& fringe_orders_v,
                                    cv::Mat& mask1,cv::Mat& mask2,cv::Mat& order_mask,
                                    int& order_u,int& order_v,
                                    cv::Mat& sub_img,
                                    int& min_u,int& min_v,
                                    cv::Mat& u2d_sub_img,cv::Mat& v2d_sub_img,
                                    double& avg_int_sub_img);


    std::tuple<double,double,double> calculateAberrationRMSInWavelength(cv::Mat&, double wavelength);
signals:
    void loadMessage(const QString& load_message);
private slots:
    void loadMessageToDialog(const QString& message);
public:
    static bool FFT2(cv::Mat&,cv::Mat&);
    static bool FFT1(cv::Mat&, cv::Mat&);
    static bool FFT_Shift(cv::Mat& ,cv::Mat&);
    static bool IFFT(cv::Mat&, cv::Mat&);
    static void IFFT_Shift(cv::Mat&);
public:
    void fffff();
    void ggggg();
private:
    Ui::WaveFrontSensorClass*     ui;
    QSharedPointer<QToolButton>   select_data_tool_button_;
    QSharedPointer<QToolButton>   parms_config_button_;
    QGridLayout*                  grid_layout_;
    QString                       filename_;
    QPlainTextEdit*               infoText;
    QDialog*                      dialog;
    QString                       hartmanngram_filename = "ex21_res_int_pr_se.dat";
    QString                       hartmanngram_png_filename = "./data_example_21/ex21_res_int_pr_se.dat.png";
    QGraphicsView*                graphic_view;
    QGraphicsScene*               graphio_scene;
};
