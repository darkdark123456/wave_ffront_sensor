#include "WaveFrontSensor.h"

WaveFrontSensor::WaveFrontSensor(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::WaveFrontSensorClass())
{
    ui->setupUi(this);
    initDefaultDisplay();

    dialog = new QDialog();
    connect(select_data_tool_button_.data(), &QToolButton::clicked, this, [&]()->void {
        filename_ = QFileDialog::getOpenFileName(this, " ", " ", "*.dat");
        if (filename_.isEmpty() || filename_.isNull()) {

            QMessageBox::warning(this, "警告", u8"没有获得选择的数据，请重新选择数据");
            return;

        }

        try {
            Configuration conf;
            cv::Mat image1 = cv::Mat::zeros(cv::Size(2, 2), CV_8UC1);
            cv::Mat image2 = cv::Mat::zeros(cv::Size(2, 2), CV_8UC1);
            processHartmanngram(conf, image1, image2);
        }
        catch (const std::exception& RUN_TIME_EXCEPTION) {

            qDebug() << RUN_TIME_EXCEPTION.what();
        }

        });
}

WaveFrontSensor::~WaveFrontSensor()
{
    free();
    delete ui;

}


void WaveFrontSensor::initDefaultDisplay() {

    grid_layout_ = new QGridLayout(this);
    select_data_tool_button_ = QSharedPointer<QToolButton>(new QToolButton());
    select_data_tool_button_->setText(u8"选择数据");
    grid_layout_->addWidget(select_data_tool_button_.data(), 0, 0);

}

/**
 * .
 * \param configuration      配置文件
 * \param hartmanngram       待处理的Hartmanngram 图像
 * \param starting_pixel     解包
 * \param thresholding_model 二值化模式 默认使用自适应二值化
 * \param img_int_thr        用来过滤掉信噪比低的像素点                默认0
 * \param area_thr           面积阈值 过滤掉面积较小的连通域           默认1
 * \param ration_thr         强度体积比阈值 用来过滤宽高比小的连通域   默认0.05
 * \param min_order_u        水平方向上拟合多项式的最小阶数            默认2
 * \param max_order_u        水平方向上的拟合多项式的最大阶数          默认4
 * \param min_order_v        垂直方向上的拟合多项式的最小阶数          默认2
 * \param max_order_v        垂直方向上的拟合多项式的最大阶数          默认4
 * \param edge_exclusion     排除靠近图像边缘的连通域的边缘宽度        默认1
 */
void WaveFrontSensor::processHartmanngram(Configuration& configuration, cv::Mat& hartmanngram, cv::Mat& starting_pixel,
    QString thresholding_model,
    int img_int_thr, int area_thr, float ration_thr,
    int min_order_u, int max_order_u,
    int min_order_v, int max_order_v,
    int edge_exclusion) {
    /** 1 获得用户默认的参数*/
    double grid_period = configuration["grid_period"];
    double dist_mask_to_detector = configuration["dist_mask_to_detector"];
    double min_fingre_number = configuration["lowest_fringe_order"];
    double centroid_power = configuration["centroid_power"];
    double detector_pixel_size = configuration["detector_pixel_size"];
    double wavelength = configuration["wavelength"];
    readWaveFrontSensorImageDat(filename_, configuration["nu_detector"], configuration["nv_detector"], configuration["upsampling"]);
}


/**
 * 默认的值
 * \param filename   "data_exmaple_21//ex21_res_int_pr_se.dat"
 * \param nu_detector 512
 * \param nv_detector 512
 * \param upsampleing  2
 * \return  Array 2D
 */


QVector<float> WaveFrontSensor::readDataFromDisk(QString filename) {

    QVector<float> intensity;
    QFile file(filename);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {

        QTextStream outputStream(&file);
        int lineToSkip = 10;
        for (int i = 0; i < lineToSkip; i++) {

            outputStream.readLine();
        }

        while (!outputStream.atEnd()) {

            QString line = outputStream.readLine();

            intensity.append(line.toFloat());
        }
        file.close();
    }
    else {

        throw FileOpenException();
    }

    return intensity;
}



void WaveFrontSensor::readWaveFrontSensorImageDat(QString filename, float nu_detector, float nv_detector, int upsampleing) {

    QVector<float> intensity;
    QSize size(static_cast<int>(upsampleing * nv_detector), static_cast<int>(upsampleing * nu_detector));

    try {
        std::future<QVector<float>> fut = std::async(std::launch::async, &WaveFrontSensor::readDataFromDisk, this, filename);
        intensity = fut.get();
    }
    catch (const std::exception& fileOpenException) {

        HIOlAB_CXX_14::ExceptionInfoPrint(fileOpenException.what());
    }


    QVector2D_ Intensity2D;
    Intensity2D.resize(size.height());

    for (int i = 0; i < Intensity2D.size(); i++) {

        Intensity2D[i].resize(size.width());
    }

                                                                                                                                                                                                                                                                                               
    HIOlAB_CXX_14::reshape(intensity, Intensity2D, size.height(), size.width());
    HIOlAB_CXX_14::flipud(Intensity2D);

    
    int dim[] = { static_cast<int>(nv_detector),upsampleing,static_cast<int>(nu_detector),upsampleing };

    QVector4D_ Array4D(dim[0], QVector<QVector<QVector<float>>>(dim[1], QVector<QVector<float>>(dim[2], QVector<float>(dim[3], 0))));

    try {

        HIOlAB_CXX_14::convertArray2D_TO_Array_4D(Intensity2D, dim[0], dim[1], dim[2], dim[3], Array4D);
    }
    catch (const std::exception& dimConvertException) {

        HIOlAB_CXX_14::ExceptionInfoPrint(dimConvertException.what());
    }


    QVector3D_  Array3D(dim[0],QVector<QVector<float>>(dim[1],QVector<float>(dim[2],0)));
   
   
    HIOlAB_CXX_14::Mean_Axis_3(Array4D, dim, Array3D);

    dialog->setWindowTitle(u8"加载信息窗口");
    dialog->setFixedSize(400, 500);
    infoText = new QPlainTextEdit(dialog);
    infoText->setReadOnly(true);
    infoText->setFixedSize(dialog->size());
    infoText->appendPlainText(u8"加载的数据如下 ");
    infoText->appendPlainText(QString::number(Array4D.size()));
    infoText->appendPlainText(QString::number(Array4D[0].size()));
    infoText->appendPlainText(QString::number(Array4D[0][0].size()));
    infoText->appendPlainText(QString::number(Array4D[0][0][0].size()));
    infoText->appendPlainText(u8"最后两个元素 ");
    infoText->appendPlainText(QString::number(Array4D[511][1][511][0]));
    infoText->appendPlainText(QString::number(Array4D[511][1][511][1]));
    
    infoText->appendPlainText(u8"沿着坐标轴3的均值矩阵shape为 ");
    infoText->appendPlainText(QString::number(Array3D.size()));
    infoText->appendPlainText(QString::number(Array3D[0].size()));
    infoText->appendPlainText(QString::number(Array3D[0][0].size()));
    infoText->appendPlainText(u8"沿着坐标轴3的均值矩阵最后的两个值  ");

    infoText->appendPlainText(QString::number(Array3D[511][1][510]));
    infoText->appendPlainText(QString::number(Array3D[511][1][511]));


    dialog->show();

}



cv::Mat WaveFrontSensor::addNoNoise(cv::Mat& intensity_map) {

    return cv::Mat::zeros(cv::Size(1, 1), CV_8UC1);
}

void WaveFrontSensor::free() {


    if (grid_layout_ != nullptr) {

        delete grid_layout_;
    }

    if (infoText != nullptr) {

        delete infoText;
    }

    if (dialog != nullptr) {

        delete dialog;
    }

}