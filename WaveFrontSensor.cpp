#include "WaveFrontSensor.h"

WaveFrontSensor::WaveFrontSensor(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::WaveFrontSensorClass())
{
    ui->setupUi(this);
    initInfoDialog();
    initDefaultDisplay();
    
    connect(this, &WaveFrontSensor::loadMessage, this, &WaveFrontSensor::loadMessageToDialog);
    connect(parms_config_button_.data(), &QToolButton::clicked, this, [&]() -> void {

        });
    connect(select_data_tool_button_.data(), &QToolButton::clicked, this, [&]()->void {
        filename_ = QFileDialog::getOpenFileName(this, " ", " ", "*.dat");
        if (filename_.isEmpty() || filename_.isNull()) {

            QMessageBox::warning(this, "警告", u8"没有获得选择的数据，请重新选择数据");
            return;

        }

        try {


            Configuration configuration;
            QVector2D_ waveFrontImageData = readWaveFrontSensorImageDat(filename_, configuration["nu_detector"], configuration["nv_detector"], configuration["upsampling"]);
            QVector<QVector<quint16>> qt_hartmanngram_png = addNoNoise(waveFrontImageData);
            QString png_filename = "./data_example_21/ex21_res_int_pr_se.dat.png";
            cv::Mat hartmanngramm_png(qt_hartmanngram_png.size(), qt_hartmanngram_png[0].size(),CV_16U);
            HIOlAB_CXX_14::array2D_Convert_CV_Mat<quint16,unsigned short>(qt_hartmanngram_png,hartmanngramm_png);
            writreGaryscalePNG(hartmanngramm_png);
            cv::Mat starting_pixel=(cv::Mat_<int>(1,2)<< static_cast<int>(configuration["nu_detector"] / 2),static_cast<int>(configuration["nv_detector"] / 2));
            processHartmanngram(configuration, hartmanngramm_png, starting_pixel,HIOlAB_CXX_14::ThresholdMode::OTSU);
        }
        catch (const std::exception& RUN_TIME_EXCEPTION) {

            HIOlAB_CXX_14::ExceptionInfoPrint(RUN_TIME_EXCEPTION.what());
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
    select_data_tool_button_->setText(u8"开始");
    grid_layout_->addWidget(select_data_tool_button_.data(), 0, 0);

    parms_config_button_ = QSharedPointer<QToolButton>(new QToolButton());
    parms_config_button_->setText(u8"参数配置");
    
    select_data_tool_button_->setStyleSheet("QToolButton {"
        "color: blue;"
        "border-radius: 8px;"
        "border: 2px solid #999;"
        "}");
    parms_config_button_->setStyleSheet("QToolButton {"
        "color: blue;"
        "border-radius: 8px;"
        "border: 2px solid #999;"
        "}");
    
    grid_layout_->addWidget(parms_config_button_.data(), 0, 1);

    graphic_view = new QGraphicsView();
    graphio_scene = new QGraphicsScene();
    graphic_view->setScene(graphio_scene);

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




void WaveFrontSensor::initInfoDialog() {
    
    dialog = new QDialog(this);
    dialog->setWindowTitle(u8"加载信息窗口");
    dialog->setFixedSize(400, 500);
    infoText = new QPlainTextEdit(dialog);
    infoText->setReadOnly(true);
    infoText->setFixedSize(dialog->size());
    infoText->appendPlainText(u8">>> 程序初始化......... ");
    infoText->appendPlainText(u8">>> 程序初始化成功 ");
    dialog->setUpdatesEnabled(true);
    dialog->show();
    dialog->setUpdatesEnabled(true);
}

void WaveFrontSensor::InitLoadInfo() {



    dialog->show();
}



void WaveFrontSensor::loadMessageToDialog(const QString& message) {

    infoText->appendPlainText(message);
   
}


/**
 * .
 * \param filename      采集到的数据
 * \param nu_detector   水平方向的检测
 * \param nv_detector   竖直方向上的检测
 * \param upsampleing   采样大小
 * 存在问题 如果数据类型维单精度float 我只想开辟一块4维空间(512,2,512,2)8MB 一块3维空间(512,2,512)4MB 一块2维空间(1024,1024)8MB 1块一维空间1024*1024  8MB，total=28MB 但是单线程读取磁盘是否太慢了经过实验大约需要3s
 */
QVector<QVector<float>> WaveFrontSensor::readWaveFrontSensorImageDat(QString filename, float nu_detector, float nv_detector, int upsampleing) {
   
    
    QVector<float> intensity;
    QSize size(static_cast<int>(upsampleing * nv_detector), static_cast<int>(upsampleing * nu_detector));
    emit loadMessageToDialog("load message succ ");
    try {

       
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        readDataFromDisk(filename, intensity);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration=end-start;
        qDebug() <<u8"从磁盘读取文件耗时 " << duration.count() / (1000000) << " s";
        infoText->appendPlainText(u8">>> 加载数据成功 ");
        QCoreApplication::processEvents();
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
   
 
    return waveFrontImage;
}




QVector<QVector<quint16>> WaveFrontSensor::addNoNoise(QVector2D_& intensity_map) {
    QVector<QVector<quint16>> detector_image;
    char pixel_depth = 14;


    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    auto min_value = HIOlAB_CXX_14::minValue(intensity_map);
    auto max_vaue  = HIOlAB_CXX_14::maxValue(intensity_map);
    HIOlAB_CXX_14::operator-(intensity_map, min_value); 
    HIOlAB_CXX_14::operator/(intensity_map, max_vaue - min_value);
    HIOlAB_CXX_14::operator*(intensity_map, std::pow(2, pixel_depth) - 1);
    HIOlAB_CXX_14::floor(intensity_map);
    HIOlAB_CXX_14::thresholdProcessing(intensity_map, std::pow(2, pixel_depth) - 1);
    HIOlAB_CXX_14::astype(intensity_map, detector_image);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
    intensity_map.clear();
    return  detector_image;
}


void WaveFrontSensor::writreGaryscalePNG(const cv::Mat& hartmanngram_png) {
    
    QString png_filename = "./data_example_21/ex21_res_int_pr_se.dat.png";
    cv::imwrite(png_filename.toStdString(), hartmanngram_png);
    infoText->appendPlainText(u8">>> 保存数据图像成功 ");
    
}



void WaveFrontSensor::setConfHartmanngramFilename(Configuration& configuration) {

}




static void PYTHON_SUB_IMAGE_SHOW(float* image,int row,int cols,int order_u,int order_v,double u0_cen_in_sub_image, double v0_cen_in_sub_image) {
    
    std::map<std::string, std::string> keywords({ { "interpolation","nearest" }, {"cmap","hot"} });
    QString title = QString("(order_u,order_v)=(%1,%2)").arg(order_u).arg(order_v);
    plt::figure();
    plt::imshow(image, 14, 13, 1, keywords);
    std::map<std::string, std::string> keywords_ = { {"color", "m"}, {"marker", "o"} };
    keywords_["markersize"] = "5";
    plt::plot(std::vector<double>{u0_cen_in_sub_image}, std::vector<double>{v0_cen_in_sub_image}, keywords_);
    plt::title(title.toStdString());
    plt::xlabel("Horizontal Pixel");
    plt::ylabel("Vertical   Pixel");
  //  plt::show(); //这会阻塞主线程 还出现的问题是 当多个线程启动画图时 会出现访问冲突的问题 但是我不想加锁 也不想用条件变量 这会使得主线程变得极其慢
    delete[] image;
}


static void PYTHON_HARTMANNGRAM_SHOW() {

    std::vector<double> x{ 1,2,3,4 };
    std::vector<double> y{ 2,3,6,8 };
    
    plt::figure_size(200,400);
    plt::title("x1");
    plt::plot(x, y);

    plt::show();
}



static void Function() {
    Py_IsInitialized() ? HIOlAB_CXX_14::MessagePrint(u8">>>初始化Python环境成功") : Py_Initialize();
    std::vector<double> x{ 1,2,3,4 };
    std::vector<double> y{ 2,3,6,8 };
    plt::figure_size(200, 400);
    plt::plot(x, y);
    plt::show();
    Py_Finalize();
}

void WaveFrontSensor::fffff() {

    std::vector<double> x{ 1,2,3,4 };
    std::vector<double> y{ 2,3,6,8 };

    plt::figure_size(200, 400);
    plt::plot(x, y);

    plt::show();
}



void WaveFrontSensor::ggggg() {

    Py_IsInitialized() ? HIOlAB_CXX_14::MessagePrint(u8">>>初始化Python环境成功") : Py_Initialize();

    //! do
    std::vector<double> x{ 1,2,3,4 };
    std::vector<double> y{ 2,3,6,8 };

    plt::figure_size(200, 400);
    plt::plot(x, y);

    plt::show();
    Py_Finalize();
}

void WaveFrontSensor::analyzeHartmanngram(  cv::Mat& hartmanngram, 
                                            cv::Mat&x2d_wfr, cv::Mat&y2d_wfr, cv::Mat&sx2d_wfr, cv::Mat&sy2d_wfr, 
                                            cv::Mat&uld_centroid, cv::Mat&vld_centroid, cv::Mat&u2d_centroid,cv::Mat&v2d_centroid,
                                            float dist_mask_to_detector,float pixel_size, float grid_period,HIOlAB_CXX_14::ThresholdMode thresholding_mode,
                                            unsigned char img_int_thr, int block_size,   int area_thr, int min_fringle_number,
                                            cv::Mat starting_pixel,    float ration_thr, float centroid_power,
                                            int*   min_order_u, int* max_order_u,
                                            int*   min_order_v, int* max_order_v,
                                            int    edge_exclusion,
                                            bool   is_show)
{
    infoText->appendPlainText("\n");
    infoText->appendPlainText(u8"***************************");
    infoText->appendPlainText(u8"开始计算 hartmanngram 的坡度 ");


    cv::Mat x2d_image_um(hartmanngram.size().height, hartmanngram.size().width, CV_32S);
    cv::Mat y2d_image_um(hartmanngram.size().height, hartmanngram.size().width, CV_32S);

    cv::Mat u2d_image = y2d_image_um.clone();
    cv::Mat v2d_image = x2d_image_um.clone();

    
    HIOlAB_CXX_14::meshgrid<int, int>(u2d_image, v2d_image, 0, hartmanngram.cols, 0, hartmanngram.rows);

    x2d_image_um = x2d_image_um * pixel_size * 1e6;
    y2d_image_um = y2d_image_um * pixel_size * 1e6;

    hartmanngram.convertTo(hartmanngram, CV_64F);
    cv::Mat OutputArray = hartmanngram.clone();
    double mat_min_value, mat_max_value;
    cv::Point min_location, max_location;

    cv::minMaxLoc(OutputArray, &mat_min_value, &mat_max_value, &min_location, &max_location);
    OutputArray = (OutputArray / mat_max_value) * (pow(2, 8) - 1);
    HIOlAB_CXX_14::floor<double>(OutputArray);;
    OutputArray.convertTo(OutputArray, CV_8U);

    cv::GaussianBlur(OutputArray, OutputArray, cv::Size(5, 5), 0);

    switch (thresholding_mode) {

        case HIOlAB_CXX_14::ThresholdMode::Adaptive:
            cv::adaptiveThreshold(OutputArray, OutputArray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, -1);
            break;
        case HIOlAB_CXX_14::ThresholdMode::OTSU:
            cv::threshold(OutputArray, OutputArray, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
            break;
        case HIOlAB_CXX_14::ThresholdMode::IMG_INT_THR:
            cv::threshold(OutputArray, OutputArray, img_int_thr, 255, cv::THRESH_BINARY);
            break;
        default:
            cv::adaptiveThreshold(OutputArray, OutputArray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, -1);
            break;
    }

    cv::Mat labels, stats, centroids;
    int num_of_rois = cv::connectedComponentsWithStats(OutputArray, labels, stats, centroids);
    infoText->appendPlainText(QString(u8"连通域的数量") + QString::number(num_of_rois));
    infoText->appendPlainText(QString("label shape height width ") + QString::number(labels.size().height) + QString(" ") + QString::number(labels.size().width));
    infoText->appendPlainText(QString("stats shape height width ") + QString::number(stats.size().height) + QString(" ") + QString::number(stats.size().width));
    infoText->appendPlainText(QString("centroids shape height width ") + QString::number(centroids.size().height) + QString(" ") + QString::number(centroids.size().width));
    cv::Mat area = stats.col(cv::CC_STAT_AREA);
    cv::transpose(area, area);
    infoText->appendPlainText(QString("area shape height width ") + QString::number(area.size().height) + QString(" ") + QString::number(area.size().width));
    int num_of_valid_rois = cv::countNonZero(area >= area_thr) - 1;
    cv::Mat roi_idx_of_valid_rois = cv::Mat::zeros(1, num_of_valid_rois, area.type());
    num_of_valid_rois = 0;


    for (int roi_idx = 1; roi_idx < num_of_rois; roi_idx++) {

        if (area.at<int>(0, roi_idx) >= area_thr) {

            roi_idx_of_valid_rois.at<int>(0, num_of_valid_rois) = roi_idx;
            ++num_of_valid_rois;
        }
        else {

            HIOlAB_CXX_14::thresholdProcessing<int>(roi_idx_of_valid_rois, roi_idx);
        }
    }


    infoText->appendPlainText(u8"Calculating the wrapped phase ");
    cv::Mat u_wrapped_phase(hartmanngram.rows, hartmanngram.cols, CV_64F);
    cv::Mat u_amplitude(hartmanngram.rows, hartmanngram.cols, CV_64F);
    cv::Mat v_wrapped_phase(hartmanngram.rows, hartmanngram.cols, CV_64F);
    cv::Mat v_amplitude(hartmanngram.rows, hartmanngram.cols, CV_64F);
    calculateWrappedPhasesUV(hartmanngram, u2d_image, v2d_image, min_fringle_number, u_wrapped_phase, u_amplitude, v_wrapped_phase, v_amplitude);


    cv::Mat quality_image = (u_amplitude + v_amplitude) / 2;
    cv::Mat quality_map = quality_image.clone();
    cv::minMaxLoc(quality_image, &mat_min_value, &mat_max_value, &min_location, &max_location);
    quality_image = 255 * (quality_image - mat_min_value) / (mat_max_value - mat_min_value);
    HIOlAB_CXX_14::floor<double>(quality_image);
    quality_image.convertTo(quality_image, CV_8U);

    cv::threshold(quality_image, quality_image, 0, 1, cv::THRESH_OTSU);
    cv::Mat searching_mask = (quality_image >= 0) / 255;


    //！3.3 如果用户没有提供指定的strarting_pixel 那么应该通过计算得到一个 
    if (starting_pixel.empty()) {

        cv::Mat starting_pixel_copy(1, 2, CV_32S);
        infoText->appendPlainText(QString(">>> starting_pixel is empty  ,create it "));
        double centroid_u = 0;
        double centroid_v = 0;

        try {

            std::pair<double, double> centroid = calculateCentroid(quality_image, u2d_image, v2d_image);
            centroid_u = centroid.first;
            centroid_v = centroid.second;

        }
        catch (const std::exception& matNullException) {

            HIOlAB_CXX_14::ExceptionInfoPrint(matNullException.what());
        }

        auto  area = cv::sum(quality_image * 1)[0];
        float r_thr = sqrt(area / M_PI) / 2;
        cv::Mat r_2d(u2d_image.rows, u2d_image.cols, CV_64F);


        cv::Mat temp_mat_1 = u2d_image - centroid_u;
        cv::Mat temp_mat_2 = v2d_image - centroid_v;

        cv::pow(temp_mat_1, 2, temp_mat_1);
        cv::pow(temp_mat_2, 2, temp_mat_2);
        cv::sqrt(temp_mat_1 + temp_mat_2, r_2d);

        searching_mask = (r_2d < r_thr) / 255;
        cv::Mat result(quality_map.rows, quality_map.cols, quality_map.type());

        try {

            std::vector<double> index_vec=HIOlAB_CXX_14::booleanArrayIndexing<double>(quality_map, searching_mask);
            mat_min_value = index_vec.at(0);
            std::pair<std::vector<int>, std::vector<int>> index_pair_vector = HIOlAB_CXX_14::where<double>(quality_map, mat_min_value);

            starting_pixel_copy.at<int>(0, 0) = index_pair_vector.first.at(0);
            starting_pixel_copy.at<int>(0, 1) = index_pair_vector.second.at(0);

            starting_pixel = starting_pixel_copy;  //! shape [2,1]
        }

        catch (const std::exception& matNullException) {

            HIOlAB_CXX_14::ExceptionInfoPrint(matNullException.what());
        }
    }



    HIOlAB_CXX_14::MessagePrint(u8">>> 确定展开的相位和条纹阶数");
    HIOlAB_CXX_14::MessagePrint(u8">>> 相位角展开");
    cv::Mat uwpu(u_wrapped_phase.rows, u_wrapped_phase.cols, u_wrapped_phase.type());
    cv::Mat uwpv(v_wrapped_phase.rows, v_wrapped_phase.cols, v_wrapped_phase.type());


    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    QGPU2SC_<double>(u_wrapped_phase, quality_map, starting_pixel, uwpu);
    QGPU2SC_<double>(v_wrapped_phase, quality_map, starting_pixel, uwpv);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << ">>> QGPU2SC Function Runtime " << duration.count() << "\n";

    cv::Mat fringe_orders_u = (uwpu-u_wrapped_phase) / (2 * M_PI);
    cv::Mat fringe_orders_v = (uwpv-v_wrapped_phase) / (2 * M_PI);

    fringe_orders_u.convertTo(fringe_orders_u, CV_32S);
    fringe_orders_v.convertTo(fringe_orders_v, CV_32S);


  
    //! 4. check the finge orders in which the labelled regions belong to
    HIOlAB_CXX_14::MessagePrint(">>> Checking fringe..... ");
    cv::Mat orders_of_labbed_rois = cv::Mat::zeros(num_of_valid_rois, 2, CV_32S); //! 存放的复数矩阵 
    cv::Mat mask = cv::Mat::zeros(fringe_orders_u.rows, fringe_orders_u.cols, CV_8U);
    int median = 0;

    std::vector<int> index_vec1;
    std::vector<int> index_vec2;
    start = std::chrono::steady_clock::now();

   

    //！ 在我去除加锁操作后 执行时间来到了1.32S 但是我还是不满意 先凑活用
    for (int idx_of_valid_roi = 0; idx_of_valid_roi < num_of_valid_rois; idx_of_valid_roi++) {
        mask = 0;
        HIOlAB_CXX_14::getMask<int>(labels, roi_idx_of_valid_rois.at<int>(0, idx_of_valid_roi), mask);

        index_vec1 = HIOlAB_CXX_14::booleanArrayIndexing<int>(fringe_orders_u, mask);
  

        try{
           
            HIOlAB_CXX_14::median<int>(index_vec1, median);
            index_vec1.clear();
            orders_of_labbed_rois.at<int>(idx_of_valid_roi, 0) = median;
        }
        catch (const std::exception& outOfRangeException){

            HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());

        }

        index_vec2 = HIOlAB_CXX_14::booleanArrayIndexing<int>(fringe_orders_v, mask);

        try{

            HIOlAB_CXX_14::median<int>(index_vec2, median);
            index_vec2.clear();
            orders_of_labbed_rois.at<int>(idx_of_valid_roi, 1) = median;

        }
        catch (const std::exception& outOfRangeException){

            HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
        }
    }

    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << ">>> Loop Runtime " << duration.count() <<"S\n";
    HIOlAB_CXX_14::unique<int>(orders_of_labbed_rois,orders_of_labbed_rois);


    //! 5.calculate centroids and slopes 
    HIOlAB_CXX_14::MessagePrint(">>> calculateing centroid and slopes... \n");
    int example_order_u = 2;
    int example_order_v = 3;

    cv::Mat sub_image_example;
    cv::Mat sub_image;
    int min_u_in_order_mask;
    int min_v_in_order_mask;
    cv::Mat u2d_sub_image;
    cv::Mat v2d_sub_image;
    double avg_int_sub_image;

    cv::Mat u2d_image_(hartmanngram.rows, hartmanngram.cols, CV_32S);
    cv::Mat v2d_image_(hartmanngram.rows, hartmanngram.cols, CV_32S);
    HIOlAB_CXX_14::meshgrid<int, int>(u2d_image_, v2d_image_, 0, hartmanngram.cols, 0, hartmanngram.rows);
    
    cv::Mat mask1 = cv::Mat::zeros(fringe_orders_u.rows, fringe_orders_u.cols, CV_8U);
    cv::Mat mask2 = cv::Mat::zeros(fringe_orders_v.rows, fringe_orders_v.cols, CV_8U);
    cv::Mat order_mask = mask2.clone();
    cropImageWithFringeOrders(hartmanngram,u2d_image_,v2d_image_, 
                              fringe_orders_u, fringe_orders_v,
                              mask1,mask2,order_mask,
                              example_order_u, example_order_v,
                              sub_image_example, min_u_in_order_mask, min_v_in_order_mask,
                              u2d_sub_image, v2d_sub_image, avg_int_sub_image);


    double  sum_of_sub_image_example = cv::sum(sub_image_example)[0];

    //! 5.2calculate the centroid and slopes in vectors  ,example shape [1,618]
    

    constexpr double nan_d_value = std::numeric_limits<double>::quiet_NaN();

    /**&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
    cv::Mat sxld(1, orders_of_labbed_rois.rows, CV_64F);
    sxld = nan_d_value;

    cv::Mat syld(1, orders_of_labbed_rois.rows, CV_64F);
    syld = nan_d_value;

    uld_centroid = cv::Mat::zeros(1, orders_of_labbed_rois.rows, CV_64F);
    uld_centroid = nan_d_value;

    vld_centroid = cv::Mat::zeros(1, orders_of_labbed_rois.rows, CV_64F);
    vld_centroid = nan_d_value;

    cv::Mat uld_aperture_center(1, orders_of_labbed_rois.rows, CV_64F);
    uld_aperture_center = nan_d_value;

    cv::Mat vld_aperture_center(1, orders_of_labbed_rois.rows, CV_64F);
    vld_aperture_center = nan_d_value;

    cv::Mat alidade_sub_image(1, orders_of_labbed_rois.rows, CV_64F);
    alidade_sub_image = nan_d_value;
    /**&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
    
    
    int order_u = 0;
    int order_v = 0;
    


   // double u0d_aperture_center = 0;
   // double v0d_aperture_center = 0;
    std::pair<double, double> centroid({ 0,0 });
    start = std::chrono::steady_clock::now();
    //! 注意 这里仍然可以使用多线程进行矩阵的分块运算 暂时未用 目前2.25s左右 先凑活用吧 但是还是感觉太慢了 还可以优化到1s以内
    for (int nidx = 0; nidx < orders_of_labbed_rois.rows; nidx++) {
        
        order_u = orders_of_labbed_rois.at<int>(nidx, 0);
        order_v = orders_of_labbed_rois.at<int>(nidx, 1);

        if ((order_u!=0 ||  order_v!=0)) {

            //! crop image
            cropImageWithFringeOrders(hartmanngram, u2d_image_,v2d_image_,
                fringe_orders_u, fringe_orders_v,
                mask1,mask2,order_mask,
                order_u, order_v,
                sub_image, min_u_in_order_mask, min_v_in_order_mask,
                u2d_sub_image, v2d_sub_image, avg_int_sub_image);


            //! 5.2.2 计算质心和斜率 
            if ((cv::sum(sub_image)[0] / sum_of_sub_image_example) >= ration_thr) {
                //! 图像增强
                cv::Mat power_sub_image(sub_image.rows, sub_image.cols, sub_image.type());
                cv::pow(sub_image, centroid_power, power_sub_image);

                centroid = calculateCentroid(power_sub_image, u2d_sub_image, v2d_sub_image);

                double u0d_centroid = centroid.first + min_u_in_order_mask;
                double v0d_centroid = centroid.second + min_v_in_order_mask;

                //! 计算参考的孔径中心
                double u0d_aperture_center = static_cast<double>(hartmanngram.cols) / 2. - 0.5 + static_cast<double>(grid_period / pixel_size) * order_u;
                double v0d_aperture_center = static_cast<double>(hartmanngram.rows) / 2. - 0.5 + static_cast<double>(grid_period / pixel_size) * order_v;

                //! 计算点位置的变换
                double du0d = u0d_centroid - u0d_aperture_center;
                double dv0d = v0d_centroid - v0d_aperture_center;

                //! 计算slopes
                sxld.at<double>(0, nidx) = du0d * pixel_size / dist_mask_to_detector;
                syld.at<double>(0, nidx) = dv0d * pixel_size / dist_mask_to_detector;
                uld_centroid.at<double>(0, nidx) = u0d_centroid;
                vld_centroid.at<double>(0, nidx) = v0d_centroid;
                uld_aperture_center.at<double>(0, nidx) = u0d_aperture_center;
                vld_aperture_center.at<double>(0, nidx) = v0d_aperture_center;
                alidade_sub_image.at<double>(0, nidx) = avg_int_sub_image;



            }
            else {

                if (is_show) {

                    printf("Spot (%d,%d) with Ration=%.3f<%0.3f is excluded", order_u, order_v, cv::sum(sub_image)[0] / sum_of_sub_image_example, ration_thr);
                }

            }

            //! 5.2.3 Show one of the sub_images 遍历复数矩阵 如果实部和给定的实部 虚部和给定的虚部相等 将结果展示出来
            if (order_u == example_order_u && order_v == example_order_v) {
                if (is_show) {
                    std::cout << ">>> find u,v " << order_u << " " << order_v << std::endl;
                    std::map<std::string,std::string> keywords = { { "interpolation", "nearest" }, { "cmap", "hot" } };
                    sub_image.convertTo(sub_image, CV_32F);
                    int num = sub_image.rows * sub_image.cols;

                    float* image= new float[num]();
                    for (int i = 0; i < num; i++) {

                        image[i] = sub_image.at<float>(i);
                    }
                    
                    //std::thread syncThreadFirst(PYTHON_SUB_IMAGE_SHOW,std::move(image),sub_image.rows,sub_image.cols,order_u,order_v,centroid.first,centroid.second);
                    //syncThreadFirst.detach();
                    
                }
            }
        }
        else {

            if (is_show) {

               printf("The order (u,v)=(%d,%d) does not contain a valid spot", order_u, order_v);
            }
        }

    }

    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << ">>> Loop X1  RunTime " << duration.count() << " s\n";


    //! 6. 将质心和斜率取集合
    //! 6.1 如果序列不是由用户提供的 ，会自动计算生成一个orders，在这种情况下，我们认为边缘处的波前收到衍射的影响，因此这些边缘处的序列将被去除
    cv::Mat unique_orders_u = orders_of_labbed_rois.col(0);
    cv::Mat unique_orders_v = orders_of_labbed_rois.col(1);
    cv::Point mat_max_value_pos, mat_min_value_pos;
    

    if (min_order_u == nullptr) {
        cv::minMaxLoc(unique_orders_u, &mat_min_value, &mat_max_value, &mat_min_value_pos, &mat_max_value_pos);
        min_order_u = new int(static_cast<int>(mat_min_value) + edge_exclusion);
    }

    if (max_order_u == nullptr) {
        cv::minMaxLoc(unique_orders_u, &mat_min_value, &mat_max_value, &mat_min_value_pos, &mat_max_value_pos);
        max_order_u = new int(static_cast<int>(mat_max_value) - edge_exclusion);
    }

    if (min_order_v == nullptr) {
        cv::minMaxLoc(unique_orders_v, &mat_min_value, &mat_max_value, &mat_min_value_pos, &mat_max_value_pos);
        min_order_v= new int(static_cast<int>(mat_min_value) + edge_exclusion);
    }

    if (max_order_v == nullptr) {

        cv::minMaxLoc(unique_orders_v, &mat_min_value, &mat_max_value, &mat_min_value_pos, &mat_max_value_pos);
        max_order_v = new int(static_cast<int>(mat_max_value) - edge_exclusion);
    }


    //! 6.2集合测量
    int  nu_wfr = (*max_order_u) - (*min_order_u) + 1;
    int  nv_wfr = (*max_order_v) - (*min_order_v) + 1;

    /** &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& 元旦前 新增Nan值修改*/
    sx2d_wfr = cv::Mat::zeros(nv_wfr, nu_wfr, CV_64F);//! parm 
   // sx2d_wfr.setTo(cv::Scalar(nan_d_value));
    sx2d_wfr = nan_d_value;
    sy2d_wfr = sx2d_wfr.clone();
    u2d_centroid = sx2d_wfr.clone();
    v2d_centroid = sx2d_wfr.clone();
    cv::Mat u2d_aperture_center = sx2d_wfr.clone();
    cv::Mat v2d_aperture_center = sx2d_wfr.clone();
    cv::Mat ai2d_sub_image = sx2d_wfr.clone();
    mask1 = cv::Mat::zeros(unique_orders_u.rows, unique_orders_u.cols, CV_8U);
    mask2 = cv::Mat::zeros(unique_orders_v.rows, unique_orders_v.cols, CV_8U);
    int m, n,column;

    /** &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& */
    start = std::chrono::steady_clock::now();
    for (int order_v = *min_order_v; order_v < *max_order_v + 1;order_v++) {
        for (int order_u = *min_order_u; order_u < *max_order_u+1; order_u++){

            m = order_v - (*min_order_v);
            n = order_u - (*min_order_u);
            mask1 = 0;
            mask2 = 0;
            HIOlAB_CXX_14::getMask(unique_orders_u, order_u, mask1);
            HIOlAB_CXX_14::getMask(unique_orders_v, order_v, mask2);
            cv::bitwise_and(mask1, mask2, mask1);
            cv::Mat nonZerosPostion;
            cv::findNonZero(mask1, nonZerosPostion);
           
            if (!nonZerosPostion.empty()) {

                column = nonZerosPostion.at<cv::Point>(0).y;
                sx2d_wfr.at<double>(m, n) = sxld.at<double>(0, column);
                sy2d_wfr.at<double>(m, n) = syld.at<double>(0, column);
                u2d_centroid.at<double>(m, n) = uld_centroid.at<double>(0, column);
                v2d_centroid.at<double>(m, n) = vld_centroid.at<double>(0, column);
                u2d_aperture_center.at<double>(m, n) = uld_aperture_center.at<double>(0, column);
                v2d_aperture_center.at<double>(m, n) = vld_aperture_center.at<double>(0, column);
                ai2d_sub_image.at<double>(m, n) = alidade_sub_image.at<double>(0, column);
            }
        }
    }

    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << ">>> Loop X2 RunTime " << duration.count() << " s\n";



    //! 7.定义x和y坐标以及x-y-z坐标中的斜率

    x2d_wfr = cv::Mat::zeros(nv_wfr, nu_wfr, CV_32S);
    y2d_wfr = cv::Mat::zeros(nv_wfr, nu_wfr, CV_32S);

    HIOlAB_CXX_14::meshgrid<int,int>(x2d_wfr, y2d_wfr, *min_order_u, *min_order_u + nu_wfr, *min_order_v, *min_order_v + nv_wfr);

    x2d_wfr.convertTo(x2d_wfr, CV_64F); //! new update  12-28
    y2d_wfr.convertTo(y2d_wfr, CV_64F);

    x2d_wfr *= grid_period;
    y2d_wfr *= grid_period;

    //! 8. 展示
    cv::Mat x2d_wfr_um = x2d_wfr * 1e6;
    cv::Mat y2d_wfr_um = y2d_wfr * 1e6;

    cv::Mat sx2d_wfr_urad = sx2d_wfr * 1e6;
    cv::Mat sy2d_wfr_urad = sy2d_wfr * 1e6;


    if (is_show) {


    }




    if (max_order_u != nullptr) {
        delete max_order_u;
    }

    if (max_order_v != nullptr) {

        delete max_order_v;
    }

    if (min_order_u != nullptr) {

        delete min_order_u;
    }

    if (min_order_v != nullptr) {

        delete min_order_v;
    }



}




/**
 * .
 * \param x2d           波前重建的x坐标
 * \param y2d           波前重建的y坐标
 * \param sx2d          波前重建的x斜率
 * \param sy2d          波前重建的y斜率
 * \param wave_length   波长
 * \param wfr           重建的波前
 * \param abr           重建的像差
 * \param wfr_coefs_wl  波长中波前的系数
 * \param abr_coes_wl   波长中像差的系数
 * \param str_method    重建的方式
 * \param str_model     模态法和系数计算中使用的模型 这里使用勒让德多项式和zernike多项式
 * \param num_of_terms  模型中的条件数
 * \param is_show       是否展示
 */
void WaveFrontSensor::analyzeHartmannSlopes(cv::Mat& x2d, cv::Mat& y2d, cv::Mat& sx2d, cv::Mat& sy2d, float wave_length, 
                                            cv::Mat& wfr, cv::Mat& abr, cv::Mat& wfr_coefs_wl, cv::Mat& abr_coes_wl, 
                                            QString str_method,ModalMethodCalculateModel str_model, int num_of_terms,
                                            bool is_show) {
    cv::Mat z2d_wfr;
    cv::Mat zx;
    cv::Mat zy;
    cv::Mat wfr_coefs;
    std::vector<cv::Mat> zxm3d;
    std::vector<cv::Mat> zym3d;
    cv::Mat xn2d;
    cv::Mat yn2d;
    cv::Mat xy_norm;
    cv::Mat jld(1, num_of_terms, CV_32S);
    wave_length = static_cast<double>(wave_length);
    constexpr double nan_double_value = std::numeric_limits<double>::quiet_NaN();


    for (int i = 1; i <=num_of_terms; i++){

        jld.at<int>(0, i-1) = i;
    }

   
    switch (str_model){
        case ModalMethodCalculateModel::Legendre:
            std::cout << "Using legendre  recontruction " << std::endl;
            Legendre::intergrate(sx2d, sy2d, x2d, y2d, jld,z2d_wfr,zx,zy,wfr_coefs,zxm3d,zym3d,xn2d,yn2d,xy_norm);
            break;
        case ModalMethodCalculateModel::Zernike:
            Zernike::intergrate(sx2d, sy2d, x2d, y2d, jld, z2d_wfr, zx, zy, wfr_coefs, zxm3d, zym3d, xn2d, yn2d, xy_norm);
            break;
        default:
            break;
    }


    if (str_method=="zonal") {

        z2d_wfr = 0;
        OMDA_Algorithm::calculate2DHeightFromSlope(sx2d, sy2d, x2d, y2d, z2d_wfr);

    }


    wfr_coefs_wl = wfr_coefs / wave_length;
    cv::Mat mask_sx=cv::Mat::zeros(sx2d.rows, sx2d.cols, CV_8U);
    cv::Mat mask_sy=cv::Mat::zeros(sy2d.rows, sy2d.cols, CV_8U);
    
    try{
       
        HIOlAB_CXX_14::isnan<double>(sx2d, mask_sx);
        HIOlAB_CXX_14::isnan<double>(sy2d, mask_sy);
    }
    catch (const VectorNullException& matrixNullException){

        HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
    }


    cv::bitwise_or(mask_sx, mask_sy,mask_sx);
    
    try{
       
        HIOlAB_CXX_14::getMaskedMatrix<double>(z2d_wfr, mask_sx, nan_double_value);
    }
    catch (const VectorNullException& matrixNullException){

        HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
    }
    
    catch (const ShapeException& shapeException) {

        HIOlAB_CXX_14::MessagePrint(shapeException.what());
    }
 
    

    cv::Mat wfr_wl = z2d_wfr / wave_length;
    cv::Mat z2d_res;
    OMDA_Algorithm::remove2DSphere(x2d,y2d,z2d_wfr,z2d_res);
    OMDA_Algorithm::remove2DTilt(x2d, y2d, z2d_res,z2d_res);

    for (int i = 0; i < 10; i++) {

        OMDA_Algorithm::remove2DSphere(x2d,y2d,z2d_res,z2d_res);
        OMDA_Algorithm::remove2DTilt(x2d,y2d,z2d_res,z2d_res);
    }


    num_of_terms = wfr_coefs.rows * wfr_coefs.cols;
    cv::Mat jld_(1, num_of_terms, CV_32S);
    for (int i = 1; i <= num_of_terms; i++) {

        jld_.at<int>(0, i - 1) = i;
    }

 

    cv::Mat abr_models, abr_coefs, xy_norm_;
    switch (str_model) {

    case ModalMethodCalculateModel::Legendre:
        Legendre::decompose(z2d_res, xn2d, yn2d, jld_, xy_norm_,abr_models,abr_coefs);
        for (int i = 0; i < 3; i++) {

            abr_coefs.at < double >(0,i)= 0;
        }
        break;
    case ModalMethodCalculateModel::Zernike:

        //! 未完善的接口
        break;
    default:

        //! 未完善的接口
        break;
    }

     cv::Mat abr_coefs_wl = abr_coefs / wave_length;
     HIOlAB_CXX_14::getMaskedMatrix<double>(z2d_res, mask_sx, nan_double_value);
     OMDA_Algorithm::remove2DTilt(x2d, y2d, z2d_res, z2d_res);
     cv::Mat abr_wl = z2d_res / wave_length;
     HIOlAB_CXX_14::getMaskedMatrix<double>(abr_models, mask_sx, nan_double_value);
     cv::Mat abr_models_wl = abr_models / wave_length;

     calculateAberrationRMSInWavelength(z2d_res, wave_length);

}



void WaveFrontSensor::cropImageWithFringeOrders(cv::Mat& img,cv::Mat& u2d_image,cv::Mat& v2d_image, 
                                                cv::Mat& fringe_orders_u, cv::Mat& fringe_orders_v,
                                                cv::Mat& mask1,cv::Mat& mask2,cv::Mat& order_mask,
                                                int& order_u, int& order_v,
                                                cv::Mat& sub_image,
                                                int& min_u_in_order_mask, int& min_v_in_order_mask,
                                                cv::Mat& u2d_sub_image, cv::Mat& v2d_sub_image,
                                                double& avg_int_sub_img) {

    mask1 = 0;
    mask2 = 0;
    order_mask = 0;

    HIOlAB_CXX_14::getMask<int>(fringe_orders_u, order_u, mask1);
    HIOlAB_CXX_14::getMask<int>(fringe_orders_v, order_v, mask2);
    cv::bitwise_and(mask1, mask2, order_mask);

    std::vector<int> index_vec_1 = HIOlAB_CXX_14::booleanArrayIndexing<int>(u2d_image, order_mask);
    std::vector<int> index_vec_2 = HIOlAB_CXX_14::booleanArrayIndexing<int>(v2d_image, order_mask);

    min_u_in_order_mask = index_vec_1.at(0);
    int max_u_in_order_mask = index_vec_1.at(index_vec_1.size() - 1);

    min_v_in_order_mask = index_vec_2.at(0);
    int max_v_in_order_mask = index_vec_2.at(index_vec_2.size() - 1);

    cv::Rect roi(min_u_in_order_mask, min_v_in_order_mask, max_u_in_order_mask - min_u_in_order_mask + 1, max_v_in_order_mask - min_v_in_order_mask + 1);

    sub_image = cv::Mat::zeros(max_v_in_order_mask - min_v_in_order_mask + 1, max_u_in_order_mask - min_u_in_order_mask + 1, CV_64F);
    sub_image = img(roi).clone();

    cv::Mat sub_mask = cv::Mat::zeros(max_v_in_order_mask - min_v_in_order_mask + 1, max_u_in_order_mask - min_u_in_order_mask + 1, CV_8U);
    sub_mask = order_mask(roi).clone();

    u2d_sub_image = cv::Mat::zeros(sub_image.rows, sub_image.cols, CV_32S);
    v2d_sub_image = cv::Mat::zeros(sub_image.rows, sub_image.cols, CV_32S);
    HIOlAB_CXX_14::meshgrid<int, int>(u2d_sub_image, v2d_sub_image, 0, sub_image.cols, 0, sub_image.rows);

    sub_mask.convertTo(sub_mask, sub_image.type());
    cv::Mat mid_result = sub_image.mul(sub_mask);
    double mat_min_value, mat_max_value;
    cv::Point min_postion, max_postion;
    cv::minMaxLoc(mid_result, &mat_min_value, &mat_max_value, &min_postion, &max_postion);

    //! first 代表了sub_image_max_v : ROW  second 代表了sub_image_max_u : COL
    std::pair<std::vector<int>, std::vector<int>> index_vec_pair = HIOlAB_CXX_14::where<double>(sub_image, mat_max_value);

    double r_thr = std::sqrt(static_cast<double>(cv::sum(order_mask)[0]) / M_PI);
    cv::Mat r_2d_sub_image(u2d_sub_image.rows, u2d_sub_image.cols, CV_32S);

    mid_result.release();
    try {

        mid_result = cv::Mat::zeros(u2d_sub_image.rows, u2d_sub_image.cols, CV_32S);
        mid_result = u2d_sub_image - index_vec_pair.second.at(0);
        cv::pow(mid_result, 2, mid_result);
        r_2d_sub_image = v2d_sub_image - index_vec_pair.first.at(0);
        cv::pow(r_2d_sub_image, 2, r_2d_sub_image);
        r_2d_sub_image += mid_result;
        r_2d_sub_image.convertTo(r_2d_sub_image, CV_64F);
        cv::sqrt(r_2d_sub_image, r_2d_sub_image);
    }

    catch (const std::exception& outOfRangeException) {

        HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
    }

    cv::Mat r_mask = cv::Mat::zeros(r_2d_sub_image.rows, r_2d_sub_image.cols, CV_8U);
    r_mask = (r_2d_sub_image < r_thr);
    cv::Mat r_mask_copy = r_mask.clone();
    cv::bitwise_not(r_mask, r_mask);
    HIOlAB_CXX_14::getMaskedMatrix<double>(sub_image, r_mask);
    avg_int_sub_img = cv::sum(sub_image)[0] / cv::countNonZero(r_mask_copy);
}



std::tuple<double,double,double> WaveFrontSensor::calculateAberrationRMSInWavelength(cv::Mat& abr, double wavelength) {
    
    cv::Mat abr_wl = abr / wavelength;
    cv::Mat mask = cv::Mat::zeros(abr_wl.rows, abr_wl.cols, CV_8U);
    HIOlAB_CXX_14::isnan<double>(abr_wl, mask);
    HIOlAB_CXX_14::getMaskedMatrix<double>(abr_wl, mask, 0);
    cv::Scalar mean, std_dev;
    cv::meanStdDev(abr_wl, mean, std_dev);
    double denominator = 1.0 / std_dev.val[0];
    printf("Aberration = λ/ %.1lf\n",denominator);
    double strehl_ratio = std::exp(-(std::pow(2 * M_PI * std_dev.val[0], 2)));
    printf("Strehl ratio (Mahajan\'s approximation) = %.4lf\n", strehl_ratio);

    return std::make_tuple(std_dev.val[0],denominator,strehl_ratio);
}




bool  WaveFrontSensor::FFT_Shift(cv::Mat& source,cv::Mat& CV_Shift_Mat) {
    
    int cx = source.cols / 2;
    int cy = source.rows / 2;

    cv::Mat Block_0(source, cv::Rect(0, 0, cx, cy));
    cv::Mat Block_1(source, cv::Rect(cx, 0, cx, cy));
    cv::Mat Block_2(source, cv::Rect(0, cy, cx, cy));
    cv::Mat Block_3(source, cv::Rect(cx, cy, cx, cy));
    
    Block_0.copyTo(CV_Shift_Mat);
    Block_3.copyTo(Block_0);
    CV_Shift_Mat.copyTo(Block_3);

    Block_1.copyTo(CV_Shift_Mat);
    Block_2.copyTo(Block_1);
    CV_Shift_Mat.copyTo(Block_2);

    CV_Shift_Mat = source;
    return CV_Shift_Mat.empty() ? throw FftException("FFT Shift Exception") : true;
}




bool  WaveFrontSensor::FFT1(cv::Mat& InputArray, cv::Mat& OutputArray) {

    cv::dft(InputArray, OutputArray,cv::DFT_COMPLEX_OUTPUT);
    FFT_Shift(OutputArray,OutputArray);
    return OutputArray.empty() ? throw FftException("FFT1 Calculate Exception ") : true;
}



bool WaveFrontSensor::FFT2(cv::Mat& InputArray ,cv::Mat&  OutputArray) {

    int rows    = InputArray.rows;
    int columns = InputArray.cols;
    int row_padded = cv::getOptimalDFTSize(rows);
    int col_padded = cv::getOptimalDFTSize(columns);
    cv::Mat padded_mat;
    cv::copyMakeBorder(InputArray, padded_mat, 0, row_padded - InputArray.rows, 0, col_padded - InputArray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<double>(padded_mat),cv::Mat::zeros(padded_mat.size(),CV_64F) };
    cv::Mat     complexl;
    cv::merge(planes, 2, complexl);
    cv::dft(complexl, complexl,cv::DFT_COMPLEX_OUTPUT);

    try{
        FFT_Shift(complexl, OutputArray);
    }
    catch (const std::exception& fftException){

        HIOlAB_CXX_14::ExceptionInfoPrint(fftException.what());
    }

    return OutputArray.empty() ? throw FftException("FFT2 Calculate Exception ") : true;
}


bool WaveFrontSensor::IFFT(cv::Mat& InputArray, cv::Mat& OutputArray) {

    cv::idft(InputArray, OutputArray, cv::DFT_COMPLEX_INPUT | cv::DFT_SCALE);
    return OutputArray.empty() ? throw FftException("IFFT Calculate Exception ") : true;
}





void WaveFrontSensor::IFFT_Shift(cv::Mat& source) {

    int cx = source.cols / 2;
    int cy = source.rows / 2;

    cv::Mat q0(source, cv::Rect(0, 0, cx, cy)); 
    cv::Mat q1(source, cv::Rect(cx, 0, cx, cy)); 
    cv::Mat q2(source, cv::Rect(0, cy, cx, cy)); 
    cv::Mat q3(source, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp; 

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}



/** 存在问题 定义的变量mat太多了 解决 能不能复用？ */

void WaveFrontSensor::calculateWrappedPhase(cv::Mat& spectrum, cv::Mat& u2d_img, cv::Mat& v2d_img, int min_fringe_number, cv::Mat& wrapped_phase, cv::Mat& amplitude, bool is_u) {

        cv::Point2f center(static_cast<double>(spectrum.cols) / 2, static_cast<double>(spectrum.rows) / 2);
        u2d_img.convertTo(u2d_img, CV_64F);
        v2d_img.convertTo(v2d_img, CV_64F);
        cv::Mat un2d = u2d_img - center.x;
        cv::Mat vn2d = v2d_img - center.y;
        cv::Mat planes[2];
        cv::split(spectrum, planes);

        cv::magnitude(planes[0], planes[1], planes[0]);
        cv::Mat amp = planes[0];

        cv::Mat_<double> (rn2d);
        cv::sqrt(un2d.mul(un2d) + vn2d.mul(vn2d), rn2d);
  
        cv::Mat_<bool>(r_mask);
        r_mask = (rn2d > min_fringe_number)/255;
        
        cv::Mat_<bool>(uv_mask);
        is_u==true ? (uv_mask = ((vn2d < un2d)/255) & ((vn2d > -un2d))/255) : (uv_mask = ((vn2d > un2d)/255) & ((vn2d > -un2d)/255));

        cv::Mat ruv_mask = r_mask & uv_mask;
        ruv_mask.convertTo(ruv_mask, CV_64F);
        cv::Mat filtered_amp = amp.mul(ruv_mask);

        cv::Point maxLoc;
        cv::minMaxLoc(filtered_amp, nullptr, nullptr, nullptr, &maxLoc);

        cv::Point2i uc, vc;
        uc = maxLoc;
        vc = maxLoc;

        int uc_half = static_cast<int>(std::floor((uc.x - center.x) / 2 + center.x));
        int vc_half = static_cast<int>(std::floor((vc.y - center.y) / 2 + center.y));

        if (filtered_amp.at<double>(vc.y, uc.x) < 3 * filtered_amp.at<double>(vc_half, uc_half)) {
            uc.x = uc_half;
            vc.y = vc_half;
        }

        cv::Mat unc = cv::Mat(1, 1, CV_64F, static_cast<double>(uc.x) - center.x);
        cv::Mat vnc = cv::Mat(1, 1, CV_64F, static_cast<double>(vc.y) - center.y);

        double r_thr = rn2d.at<double>(vc.y, uc.x) / 4;
        cv::pow(un2d - unc, 2, un2d);
        cv::pow(vn2d - vnc, 2, vn2d);
        cv::sqrt(un2d+vn2d,un2d);
        cv::Mat rc2d = un2d;
        cv::Mat mask= (rc2d < r_thr) / 255;
        mask.convertTo(mask, CV_64F);
        cv::Mat filtered_complex_img(spectrum.rows, spectrum.cols,spectrum.type());
        std::vector<cv::Mat> channels;
        cv::split(spectrum, channels);
        cv::Mat real = channels[0].mul(mask);
        cv::Mat imag = channels[1].mul(mask);
        cv::Mat plans[] = { real,imag };

        cv::merge(plans,2,filtered_complex_img);

        cv::Mat shift_filter_complex_img=filtered_complex_img.clone();
        cv::Mat ifft_result_mat(shift_filter_complex_img.rows,shift_filter_complex_img.cols,shift_filter_complex_img.type());

        try{
            IFFT_Shift(shift_filter_complex_img);
            IFFT(shift_filter_complex_img, ifft_result_mat);
        }
        catch (const std::exception& fftException){

            HIOlAB_CXX_14::ExceptionInfoPrint(fftException.what());
        }


        cv::split(ifft_result_mat,plans);
        HIOlAB_CXX_14::angle<double>(plans[0], plans[1], wrapped_phase);

        cv::pow(plans[0], 2, plans[0]);
        cv::pow(plans[1], 2, plans[1]);
        cv::sqrt(plans[0] + plans[1], amplitude);
}




void WaveFrontSensor::calculateWrappedPhasesUV(  cv::Mat&hartmanngram, cv::Mat&u2d_image, cv::Mat& v2d_image, 
                                                 int min_fringe_number,
                                                 cv::Mat& u_wrapped_phase,cv::Mat& u_amplitude,cv::Mat& v_wrapped_phase,cv::Mat& v_amplitude ) {
    cv::Mat  spectrum;

    try{

        FFT2(hartmanngram, spectrum);
    }
    catch (const std::exception& fftException){

        HIOlAB_CXX_14::ExceptionInfoPrint(fftException.what());

    }

    calculateWrappedPhase(spectrum,u2d_image,v2d_image,min_fringe_number,u_wrapped_phase,u_amplitude,true);
    calculateWrappedPhase(spectrum, u2d_image, v2d_image, min_fringe_number, v_wrapped_phase, v_amplitude, false);

}


std::pair<double,double> WaveFrontSensor::calculateCentroid(cv::Mat& sub_image, cv::Mat& u2d_image, cv::Mat&v2d_image) {
    
    if (u2d_image.empty() || v2d_image.empty()) {
        HIOlAB_CXX_14::meshgrid<int, int>(u2d_image, v2d_image, 0, sub_image.cols, 0, sub_image.rows);

        if (u2d_image.empty() || v2d_image.empty()) {

            throw VectorNullException("Throw u2d_image or v2d_image is null exception");
        }


    }

    if (sub_image.type() != u2d_image.type()) {

        u2d_image.convertTo(u2d_image,sub_image.type());

    }

    if (sub_image.type() != v2d_image.type()) {

        v2d_image.convertTo(v2d_image, sub_image.type());
    }

 
    double  image_sum = cv::sum(sub_image)[0];
    double  u2d_image_sum = cv::sum(sub_image.mul(u2d_image))[0];
    double  v2d_image_sum = cv::sum(sub_image.mul(v2d_image))[0];
    double  centroid_u = u2d_image_sum / image_sum;
    double  centroid_v = v2d_image_sum / image_sum;
    return std::make_pair(centroid_u,centroid_v);
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
void WaveFrontSensor::processHartmanngram(Configuration& configuration, cv::Mat& hartmanngram, cv::Mat starting_pixel,
    HIOlAB_CXX_14::ThresholdMode thresholding_mode,
    int img_int_thr, int area_thr, float ration_thr,
    int* min_order_u, int* max_order_u,
    int* min_order_v, int* max_order_v,
    int edge_exclusion) {

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    /** 1 获得用户默认的参数*/
    double grid_period = configuration["grid_period"];
    double dist_mask_to_detector = configuration["dist_mask_to_detector"];
    double min_fingre_number = configuration["lowest_fringe_order"];
    double centroid_power = configuration["centroid_power"];
    double detector_pixel_size = configuration["detector_pixel_size"];
    double wavelength = configuration["wavelength"];

    /** 2 分析Hartmann 图像*/
    cv::Mat x2d, y2d, sx2d, sy2d;
    cv::Mat uld_centroid, vld_centroid;
    cv::Mat u2d_centroid, v2d_centroid;
    analyzeHartmanngram(hartmanngram,
                        x2d,y2d,sx2d,sy2d,
                        uld_centroid,vld_centroid,u2d_centroid,v2d_centroid, 
                        dist_mask_to_detector, detector_pixel_size, grid_period, 
                        HIOlAB_CXX_14::ThresholdMode::OTSU,
                        img_int_thr, 31, area_thr, min_fingre_number, starting_pixel, ration_thr, centroid_power,
                        min_order_u, max_order_u, min_order_v, max_order_v, edge_exclusion, true);


    //! 目前所有的数据的都没有问题 开始进行最后的多项式拟合重建操作
    
    std::cout << u8">>> Start reconstructing....\n";

    //! 3.分析图像的形状
    cv::Mat wfr, abr, coefs_wfr_wl, coefs_abr_wl;
    analyzeHartmannSlopes(x2d, y2d, sx2d, sy2d, wavelength, wfr, abr, coefs_wfr_wl, coefs_abr_wl);



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


    if(graphio_scene!=nullptr){
        
        delete graphio_scene;
    }

    if (graphic_view != nullptr) {

        delete graphic_view;
    }

}