#include "WaveFrontSensor.h"

WaveFrontSensor::WaveFrontSensor(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::WaveFrontSensorClass())
{
    ui->setupUi(this);
    initDefaultDisplay();

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
    QVector2D_ waveFrontImageData=readWaveFrontSensorImageDat(filename_, configuration["nu_detector"], configuration["nv_detector"], configuration["upsampling"]);
    TestFunction();
    addNoNoise(waveFrontImageData);
}


/**
 * 默认的值
 * \param filename   "data_exmaple_21//ex21_res_int_pr_se.dat"
 * \param nu_detector 512
 * \param nv_detector 512
 * \param upsampleing  2
 * \return  Array 2D
 */


void WaveFrontSensor::readDataFromDisk(QString filename, QVector<float>& intensity) {

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
}



void WaveFrontSensor::CAT_DATA_INFO(const QVector4D_& array_4d , const QVector3D_& array_3d) {

    dialog = new QDialog();
    dialog->setWindowTitle(u8"加载信息窗口");
    dialog->setFixedSize(400, 500);
    infoText = new QPlainTextEdit(dialog);
    infoText->setReadOnly(true);
    infoText->setFixedSize(dialog->size());
    infoText->appendPlainText(u8"加载的数据如下 ");
    infoText->appendPlainText(QString::number(array_4d.size()));
    infoText->appendPlainText(QString::number(array_4d[0].size()));
    infoText->appendPlainText(QString::number(array_4d[0][0].size()));
    infoText->appendPlainText(QString::number(array_4d[0][0][0].size()));
    infoText->appendPlainText(u8"最后两个元素 ");
    infoText->appendPlainText(QString::number(array_4d[511][1][511][0]));
    infoText->appendPlainText(QString::number(array_4d[511][1][511][1]));

    infoText->appendPlainText(u8"沿着坐标轴3的均值矩阵shape为 ");
    infoText->appendPlainText(QString::number(array_3d.size()));
    infoText->appendPlainText(QString::number(array_3d[0].size()));
    infoText->appendPlainText(QString::number(array_3d[0][0].size()));
    infoText->appendPlainText(u8"沿着坐标轴3的均值矩阵前的两个值  ");

    infoText->appendPlainText(QString::number(array_3d[0][0][0]));
    infoText->appendPlainText(QString::number(array_3d[0][0][1]));
    infoText->appendPlainText(u8"沿着坐标轴3的均值矩阵最后的两个值  ");

    infoText->appendPlainText(QString::number(array_3d[511][1][510]));
    infoText->appendPlainText(QString::number(array_3d[511][1][511]));
    dialog->show();
}



/**
 * .
 * \param filename      采集到的数据
 * \param nu_detector   水平方向的检测
 * \param nv_detector   竖直方向上的检测
 * \param upsampleing   采样大小
 * 存在问题 如果数据类型维单精度float 我只想开辟一块4维空间(512,2,512,2)8MB 一块3维空间(512,2,512)4MB 一块2维空间(1024,1024)8MB 1块一维空间1024*1024  8MB，total=28MB 但是单线程读取磁盘是否太慢了？？？ 经过实验大约需要3s
 */
QVector<QVector<float>> WaveFrontSensor::readWaveFrontSensorImageDat(QString filename, float nu_detector, float nv_detector, int upsampleing) {
    
    QVector<float> intensity;
    QSize size(static_cast<int>(upsampleing * nv_detector), static_cast<int>(upsampleing * nu_detector));

    try {

        TimeConsuming timer;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        timer.setStartTime(start);
        readDataFromDisk(filename, intensity);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        timer.setEndTime(end);
        std::chrono::duration<double> duration=timer.calculateDuration();
        qDebug() <<u8"从磁盘读取文件耗时 " << duration.count() / (1000000) << " s";
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
    QVector2D_ waveFrontImage(dim[0], QVector<float>(dim[2], 0));
    HIOlAB_CXX_14::Mean_Axis_1(Array3D,dim[0],dim[1],dim[2], waveFrontImage);
    CAT_DATA_INFO(Array4D, Array3D);

    infoText->appendPlainText(QString(u8"二维均值矩阵的最前面两个元素 ") + QString::number(waveFrontImage[0][0]) + QString::number(waveFrontImage[0][1]));
    infoText->appendPlainText(QString(u8"二维均值矩阵的最后面两个元素 ") + QString::number(waveFrontImage[511][510]) + QString::number(waveFrontImage[511][511]));

    return waveFrontImage;
}




QVector<QVector<quint16>> WaveFrontSensor::addNoNoise(QVector2D_& intensity_map) {
    QVector<QVector<quint16>> detector_image;
    char pixel_depth = 14;

    TimeConsuming timer;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    auto min_value = HIOlAB_CXX_14::minValue(intensity_map);
    auto max_vaue  = HIOlAB_CXX_14::maxValue(intensity_map);
    HIOlAB_CXX_14::operator-(intensity_map, min_value); /**0.0098s*/
    HIOlAB_CXX_14::operator/(intensity_map, max_vaue - min_value);/**0.0072*/
    HIOlAB_CXX_14::operator*(intensity_map, std::pow(2, pixel_depth) - 1);
    HIOlAB_CXX_14::floor(intensity_map);
    HIOlAB_CXX_14::thresholdProcessing(intensity_map, std::pow(2, pixel_depth) - 1);
    HIOlAB_CXX_14::astype(intensity_map, detector_image);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
    intensity_map.clear();
    return  detector_image;
}




void WaveFrontSensor::TestFunction() {

    //int dim[4] = { 4,2,4,2 };
    //QVector2D_ array_2d(8, QVector<float>(8, 0));
    //QVector2D_ array_2d_(4, QVector<float>(4, 0));

    //QVector4D_ array_4d(dim[0], QVector<QVector<QVector<float>>>(dim[1], QVector<QVector<float>>(dim[2], QVector<float>(dim[3], 0))));
    //QVector3D_ array_3d(dim[0],QVector<QVector<float>>(dim[1],QVector<float>(dim[2],0)));
    //int index = 0;
    //for (int i = 0; i < 8; i++)
    //{
    //    for (int j = 0; j < 8; j++)
    //    {
    //        array_2d[i][j] = index++;
    //    }

    //}

    //try{

    //    HIOlAB_CXX_14::convertArray2D_TO_Array_4D(array_2d, 4, 2, 4, 2, array_4d);
    //}
    //catch (const std::exception& dimConvertException){

    //    HIOlAB_CXX_14::ExceptionInfoPrint(dimConvertException.what());

    //}


    //HIOlAB_CXX_14::Mean_Axis_3(array_4d, dim, array_3d);
    //CAT_DATA_INFO(array_4d, array_3d);

    //HIOlAB_CXX_14::Mean_Axis_1(array_3d, dim[0],dim[1],dim[2],array_2d_);


    //try {
    //    
    //    auto min_value = HIOlAB_CXX_14::minValue(array_2d);
    //    auto max_value = HIOlAB_CXX_14::maxValue(array_2d);

    //    qDebug() << "测试函数中的最小值 " << min_value << "\n";
    //    qDebug() << "测试函数中的最大值 " << max_value << "\n";

    //}
    //catch (const std::exception& vectorNullException){


    //}

    //测试addNoNorise函数
    qDebug() << "测试addNoNorise函数\n";
    QVector<float> array{ 2,1,2,1,2,4,3,2,3 };
    int index = 0;
    QVector2D_ matrix(3, QVector<float>(3, 0));
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++) {

            matrix[i][j] = array[index++];
        }

    }
    auto min_value = HIOlAB_CXX_14::minValue(matrix);
    qDebug() << " min " << min_value << "\n";
    auto max_v = HIOlAB_CXX_14::maxValue(matrix);
    qDebug() << "max " << max_v << "\n";
    auto ptp = HIOlAB_CXX_14::ptp(matrix);
    qDebug() << " ptp " << ptp << "\n";

    HIOlAB_CXX_14::operator-(matrix, min_value);

    qDebug() << u8"将一个矩阵减去最小值后的 矩阵的值 \n";
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++) {

            std::cout << matrix[i][j] << " ";
        }

        std::cout << std::endl;
    }

    HIOlAB_CXX_14::operator/(matrix, max_v - min_value);
    qDebug() << u8"将一个矩阵除以峰值后矩阵的值 \n";
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++) {

            std::cout << matrix[i][j] << " ";
        }

        std::cout << std::endl;
    }

    HIOlAB_CXX_14::operator*(matrix, pow(2,2));
    qDebug() << u8"将一个矩阵乘以后矩阵的值 \n";
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++) {

            std::cout << matrix[i][j] << " ";
        }

        std::cout << std::endl;
    }


    HIOlAB_CXX_14::floor(matrix);
    qDebug() << u8"将一个矩阵取整数后矩阵的值 \n";
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++) {

            std::cout << matrix[i][j] << " ";
        }

        std::cout << std::endl;
    }

    HIOlAB_CXX_14::thresholdProcessing(matrix, 2);
    qDebug() << u8"阈值处理后矩阵的值 \n";
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++) {

            std::cout << matrix[i][j] << " ";
        }

        std::cout << std::endl;
    }
    qDebug() << "每个元素占用的字节 " << sizeof(matrix[0][0]) << "\n";

    QVector<QVector<quint16>> vector2D;
    HIOlAB_CXX_14::astype(matrix, vector2D);
    qDebug() << u8"类型转换后矩阵的值 \n";
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++) {

            std::cout << matrix[i][j] << " ";
        }

        std::cout << std::endl;
    }
    qDebug() << u8"类型转换后每个元素的类型\n";

    qDebug() << typeid(decltype(vector2D[0][0])).name() << "\n";
    qDebug() << "每个元素占用的字节 " << sizeof(vector2D[0][0]) << "\n";


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