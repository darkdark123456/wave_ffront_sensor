#pragma once
#pragma warning(disable:26451)

#include    <qvector.h>
#include    <algorithm>
#include    <queue>
#include    <mutex>
#include    <iostream>
#include    <complex>
#include    <functional>
#include    <map>
#include    <forward_list>
#include    <QtConcurrent/QtConcurrent>
#include    <qdebug.h>
#include    <Eigen/Dense>
#include    <Eigen/core>
#include	<Eigen/Sparse>


namespace HIOlAB_CXX_14 {

    enum class ThresholdMode {
        
        Adaptive = 0,
        OTSU = 1,
        IMG_INT_THR = 2
    };

   
    enum class MaxSupportUseThreads {

        Default_Level_Thread_Configurator = 1,
        Sec_Level_Thread_Configurator = 2,
        Thd_Level_Thread_Configurator = 4,
        Fo_Level_Thread_Configurator  = 6,
        Fi_Level_Thread_Configurator  = 8,
        Si_level_Thread_Configurator  = 10,
        Sev_level_Thread_Configurator = 12,
        Ei_level_Thread_Configurator  = 14,
        Max_level_Thread_Configurator = 16
    };


    static void ExceptionInfoPrint(const char* const _Message) {

        qDebug() << "Exception throw:  " << _Message << "\n";
    }


    static void MessagePrint(const char* const _Message) {

        qDebug() << _Message << "\n";
    }



    template<typename T>
    static void reshape(QVector<T>& Array_1D, QVector<QVector<T>>& Array_2D, int height, int width) {

        int index = 0;
        int instensityLength = Array_1D.length();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {

                index < instensityLength ? Array_2D[i][j] = Array_1D[index++] : Array_2D[i][j] = 0;
            }

        }
    }


    template<typename T>
    static void swap(QVector<T>& Vector_1, QVector<T>& Vector_2) {
        int low = 0;
        int high = MIN(Vector_1.length(), Vector_2.length());
        for (; low < high; low++) {

            std::swap(Vector_1[low], Vector_2[low]);
        }
    }


    template<typename T>
    static void flipud(QVector<QVector<T>>& source_array) {
        int low = 0;
        int high = source_array.length() - 1;
        while (low < high) {

            HIOlAB_CXX_14::swap(source_array[low++], source_array[high--]);
        }
    }


    template<typename T, typename Dim = int>
    static void convertArray2D_TO_Array_4D(const QVector<QVector<T>>& Array_2D, Dim dim1, Dim dim2, Dim dim3, Dim dim4,
        QVector<QVector<QVector<QVector<T>>>>& Array4D) {
        int rows = Array_2D.size();
        int columns = Array_2D[0].size();

        if ((dim1 * dim2 * dim3 * dim4) > (rows * columns)) {

            throw DimConvertException();
        }

        int index = 0;
        for (auto d1 = 0; d1 < dim1; d1++) {
            for (auto d2 = 0; d2 < dim2; d2++) {
                for (auto d3 = 0; d3 < dim3; d3++) {
                    for (auto d4 = 0; d4 < dim4; d4++) {

                        Array4D[d1][d2][d3][d4] = Array_2D[d1 * 2 + d2][d3 * 2 + d4];
                    }
                }

            }
        }

    }


    template<typename T>
    static void  Mean_Axis_3(const QVector<QVector<QVector<QVector<T>>>>& Array4D,int* dim,QVector<QVector<QVector<T>>>& Mean_Array3D) {
        for (auto d1 = 0; d1 < dim[0]; d1++) {
            for (auto d2 = 0; d2 < dim[1]; d2++) {
                for (auto d3 = 0; d3 < dim[2]; d3++) {

                    double sum = 0.;
                    for (auto d4 = 0; d4 < dim[3]; d4++) {
                        
                        sum += Array4D[d1][d2][d3][d4];
                    }
                    Mean_Array3D[d1][d2][d3] = sum / dim[3];
                }

            }
        }
    }



    template<typename T>
    static void Mean_Axis_1(const QVector<QVector<QVector<T>>>& Array3D,const int& dim1,const int& dim2,const int& dim3, QVector<QVector<T>>& Mean_Array2D) {
        double sum = 0;
        for (auto i = 0; i < dim1; i++) {
            for (auto j= 0; j < dim3; j++) {
                sum = 0.;
                for (auto k = 0; k < dim2; k++) {

                        sum += Array3D[i][k][j];       
                }
                Mean_Array2D[i][j] = sum / dim2;
            }
        }
    }



    template<typename T>
    static auto minValue(const QVector<QVector<T>>& Array2D) -> typename std::remove_reference<decltype(Array2D[0][0])>::type{
 
        if (Array2D.isEmpty()) {

            throw VectorNullException();
        }
        auto min_value = FLT_MAX;
        auto row_count = Array2D.size();
        for (auto i = 0; i < row_count; i++){
            auto iterator = std::min_element(Array2D[i].constBegin(), Array2D[i].constEnd());
            if (*iterator < min_value ) {
                
                min_value = *iterator;
            }

        }
        return min_value;
    }



    template<typename T>
    static auto maxValue(const QVector<QVector<T>>& Array2D) -> typename std::remove_reference<decltype(Array2D[0][0])>::type {
        if (Array2D.isEmpty()) {

            throw VectorNullException();
        }
        auto max_value = FLT_MIN;
        auto row_count = Array2D.size();
        for (auto i = 0; i < row_count; i++){

            auto iterator = std::max_element(Array2D[i].constBegin(), Array2D[i].constEnd());
            if (*iterator > max_value) {

                max_value = *iterator;
            }
        }
        return max_value;
    } 





    static int SystemCurrentCanUseThreads() {

        int current_thread_num = QThread::idealThreadCount() - 1;
        
        
        //! 采用单线程处理
        if (current_thread_num <= 0) {

            return 0;
        }


        //! return 16 threads
        if (current_thread_num >= static_cast<int>(MaxSupportUseThreads::Max_level_Thread_Configurator)) {

            
            return static_cast<int>(MaxSupportUseThreads::Max_level_Thread_Configurator);
        }

        //! return 14 threads
        else if(current_thread_num >= static_cast<int>(MaxSupportUseThreads::Ei_level_Thread_Configurator)){

            
            return static_cast<int>(MaxSupportUseThreads::Ei_level_Thread_Configurator);

        }

        //! return 12 threads
        else if (current_thread_num >= static_cast<int>(MaxSupportUseThreads::Sev_level_Thread_Configurator)) {


            return static_cast<int>(MaxSupportUseThreads::Sev_level_Thread_Configurator);
        }

        //! return 10 threads
        else if (current_thread_num >= static_cast<int>(MaxSupportUseThreads::Si_level_Thread_Configurator)) {


            return static_cast<int>(MaxSupportUseThreads::Si_level_Thread_Configurator);

        }

        //! return 8 threads
        else if (current_thread_num >= static_cast<int>(MaxSupportUseThreads::Fi_Level_Thread_Configurator)) {

            
            return static_cast<int>(MaxSupportUseThreads::Fi_Level_Thread_Configurator);
        }

        //! return 6 threads
        else if (current_thread_num >= static_cast<int>(MaxSupportUseThreads::Fo_Level_Thread_Configurator)) {

            return static_cast<int>(MaxSupportUseThreads::Fo_Level_Thread_Configurator);


        }

        //! return 4 threads
        else if (current_thread_num >= static_cast<int>(MaxSupportUseThreads::Thd_Level_Thread_Configurator)) {

            return static_cast<int>(MaxSupportUseThreads::Thd_Level_Thread_Configurator);


        }

        //! return 2 threads
        else if (current_thread_num >= static_cast<int>(MaxSupportUseThreads::Sec_Level_Thread_Configurator)) {

            return static_cast<int>(MaxSupportUseThreads::Sec_Level_Thread_Configurator);


        }

        else{
  
  
        }

        return 0;
    }


        




    template<typename T>
    static auto ptp(const QVector<QVector<T>>& Array2D){

        T ptp = HIOlAB_CXX_14::maxValue(Array2D) - HIOlAB_CXX_14::minValue(Array2D);
        return ptp;
    }


    template <typename T,typename U>
    static void operator-(QVector<QVector<T>>& Array2D, U value) {
        auto rows    = Array2D.size();
        auto columns = Array2D[0].size();
        int  os_remain_thread_num = QThread::idealThreadCount()-2;
        int  actual_thread_num     = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        if (block_num==0){

            actual_thread_num = 1;
        }

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num-1; i++){
            start_row = i * block_num;
            end_row =   (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Array2D, start_row, end_row, columns, value]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try{

                            Array2D[i][j] -= value;
                        }
                        catch (const std::exception& outOfRangeException){

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for(QFuture<void>& future : futures){

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++){
                try {

                    Array2D[i][j] -= value;
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }



    template<typename T,typename U>
    static void operator/(QVector<QVector<T>>& Array2D,const U& numerator) {
        if (numerator == 0) {

            throw  DenominatorZeroException();
        }

        auto rows = Array2D.size();
        auto columns = Array2D[0].size();
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        if (block_num == 0) {

            actual_thread_num = 1;
        }

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Array2D, start_row, end_row, columns, numerator]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            Array2D[i][j] /= numerator;
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    Array2D[i][j] /= numerator;
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }



    template<typename T,typename U>
    static void operator*(QVector<QVector<T>>& Array2D,const U& value) {
        auto rows = Array2D.size();
        auto columns = Array2D[0].size();
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        block_num < 128 ? actual_thread_num = 1 : void(0);

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Array2D, start_row, end_row, columns, value]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            Array2D[i][j] *= value;
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    Array2D[i][j] *= value;
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }


    
    template<typename T>
    static void VectorMul(std::vector<cv::Mat>& InputArray1, std::vector<cv::Mat>& InputArray2,std::vector<cv::Mat>& OutputArray) {

        if (InputArray1.empty() || InputArray2.empty()) {

            return;
        }

        if (InputArray1.size() != InputArray2.size()) {

            throw MatrixMultDimException("InputArray1 size != InputArray2 size");
        }

        if (InputArray1.at(0).type() != InputArray2.at(0).type()) {

            throw TypeException();
        }

        OutputArray.resize(InputArray1.size());
        auto row = 0;
        auto end_row=0;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = InputArray2.size() / actual_thread_num;
        block_num < 25 ? actual_thread_num = 1 : void(0);

        QVector<QFuture<void>> futures;
        futures.resize(actual_thread_num - 1);
        for (int i = 0; i < actual_thread_num-1; i++) {
            row = i*block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future_ = QtConcurrent::run([&InputArray1,&InputArray2,&OutputArray,row,end_row]() -> void {
                for (int j = row; j < end_row;j++) {
                    try{

                      OutputArray.at(j) = InputArray1.at(j).mul(InputArray2.at(j));

        
                    }
                    catch (const std::exception& outOfRangeException){

                        HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
                    }

                }
                
                });
            futures.append(future_);
        }

        for (auto & future : futures) {
                
            future.waitForFinished();
        }

        for (int i = end_row; i <InputArray2.size() ; i++){

            try {

               OutputArray.at(i) = InputArray1.at(i).mul(InputArray2.at(i));
 
            }
            catch (const std::exception& outOfRangeException) {

                HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
            }
        }
    }



    template<typename T>
    static void VectorMul(std::vector<cv::Mat>& InputArray1,cv::Mat& InputArray2, std::vector<cv::Mat>& OutputArray) {

        if (InputArray1.empty() || InputArray2.empty()) {

            throw VectorNullException();
        }

        if (InputArray1.size() != static_cast<size_t>(InputArray2.rows * InputArray2.cols)) {

            throw MatrixMultDimException("InputArray1 size != InputArray2 size exception in VectorMul function");
        }

        if (InputArray2.rows > 1) {

            throw ShapeException("InputArray2 row > 1 exception ");
        }

        OutputArray.resize(InputArray1.size());
        auto row = 0;
        auto end_row = 0;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = InputArray1.size() / actual_thread_num;
        block_num < 25 ? actual_thread_num = 1 : void(0);

        QVector<QFuture<void>> futures(actual_thread_num - 1);
 
        for (int i = 0; i < actual_thread_num - 1; i++) {
            row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future_ = QtConcurrent::run([&InputArray1, &InputArray2, &OutputArray, row, end_row]() -> void {
                for (int j = row; j < end_row; j++) {
                    try {

                        OutputArray.at(j)=InputArray1.at(j) * InputArray2.at<T>(0, j);
                    }
                    catch (const std::exception& outOfRangeException) {

                        HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
                    }
                }
                });
            futures.append(future_);
        }

        for (auto & future_ : futures) {

            future_.waitForFinished();
        }

        for (int i = end_row; i < InputArray1.size(); i++) {
            try {

                OutputArray.at(i)=InputArray1.at(i) * InputArray2.at<T>(0, i);
            }
            catch (const std::exception& outOfRangeException) {

                HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
            }
        }
    }


    //! 将高维d*m*n的矩阵按指定axis进行相加 模仿的numpy.sum  按照并行加法器的思想
    template<typename T>
    static void sum(std::vector<cv::Mat>& InputArray, cv::Mat& OutputArray , int axis = 0) {
        
        if (InputArray.empty()) {

            throw VectorNullException();
        }

        int row = 0;
        int end_row = 0;
        int os_remain_thread_num = QThread::idealThreadCount();
        int actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int block_num = InputArray.size() / actual_thread_num;
        block_num < 25 ? actual_thread_num = 1 : void(0);
        int end_index = actual_thread_num -1;
        std::vector<cv::Mat> mid_result(actual_thread_num);
        OutputArray = cv::Mat::zeros(InputArray[0].rows, InputArray[0].cols, InputArray[0].type());
        QVector<QFuture<void>> futures(actual_thread_num - 1);

        for (int i = 0; i < actual_thread_num - 1; i++) {
            row = i * block_num;
            end_row = (i + 1) * block_num;
            mid_result[i] = cv::Mat::zeros(InputArray[0].rows, InputArray[0].cols, InputArray[0].type());
            QFuture<void> future_=QtConcurrent::run([&InputArray,&mid_result,i,axis,row,end_row]() -> void {
                switch (axis) {
                    case 0:
                        for (int j = row; j < end_row; j++) {
                            try{

                                mid_result[i] += InputArray[j];
                            }
                            catch (const std::exception& outRangeException){

                                HIOlAB_CXX_14::MessagePrint(outRangeException.what());
                            }
                        }
                        break;
                    case 1:
                        break;
                    case 2:
                        break;
                    default:
                        break;
                }

                });
            futures.append(future_);
        }


        for (auto & future : futures) {

            future.waitForFinished();
        }

        mid_result[end_index] = cv::Mat::zeros(InputArray[0].rows, InputArray[0].cols, InputArray[0].type());
        for (int i = end_row; i < InputArray.size(); i++) {
            try{
                
                mid_result[end_index] += InputArray[i];
            }
            catch (const std::exception& outRangeException){

                HIOlAB_CXX_14::MessagePrint(outRangeException.what());
            }
        }

        for (int i = 0; i < mid_result.size(); i++) {

            OutputArray += mid_result[i];
        }
    }





    template<typename T>
    static void DimensionReduce(std::vector<cv::Mat>& InputArray,cv::Mat& outputArray) {
        if (InputArray.empty()) {
            
            throw VectorNullException();
        }

        int start_row = 0;
        int end_row = 0;
        int os_remain_thread_num = QThread::idealThreadCount();
        int actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int block_num = InputArray.size() / actual_thread_num;
        int tmp_row = InputArray.at(0).rows;
        int tmp_col = InputArray.at(0).cols;
        int expand_dim = tmp_row*tmp_col;
        block_num < 25 ? actual_thread_num = 1 : void(0);
        outputArray = cv::Mat::zeros(  InputArray.size(), tmp_row * tmp_col, InputArray.at(0).type());
        std::vector<cv::Mat> vec_mat(actual_thread_num);
        int end_index = actual_thread_num - 1;

        QVector<QFuture<void>> futures(actual_thread_num - 1);
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&InputArray, &outputArray, &vec_mat,start_row, end_row,expand_dim,i]() -> void {
                for (int m = start_row; m < end_row; m++) {
                    try{

                        vec_mat[i] = InputArray.at(m).reshape(1,expand_dim ).t();
                        vec_mat[i].copyTo(outputArray.row(m));
                    }
                    catch (const std::exception& outOfRangeException){

                        HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
                    }
                }

                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        //! 剩余的任务交给主线程chuli 
        for (int m = end_row; m < InputArray.size(); m++) {

            try {

                vec_mat[end_index] = InputArray.at(m).reshape(1, expand_dim).t();
                vec_mat[end_index].copyTo(outputArray.row(m));
            }
            catch (const std::exception& outOfRangeException){

                HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
            }

        }

        cv::transpose(outputArray, outputArray);
    }


    
    template<typename T>
    static void floor(QVector<QVector<T>>& Array2D) {
        auto rows = Array2D.size();
        auto columns = Array2D[0].size();
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        if (block_num == 0) {

            actual_thread_num = 1;
        }

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Array2D, start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            Array2D[i][j] = std::floor(Array2D[i][j]) ;
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    Array2D[i][j] = std::floor(Array2D[i][j]) ;
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }



    template<typename T>
    static void floor(cv::Mat& CV_Mat) {
        auto rows = CV_Mat.rows;
        auto columns = CV_Mat.cols;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        if (block_num == 0) {

            actual_thread_num = 1;
        }

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&CV_Mat, start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            CV_Mat.at<T>(i, j) = std::floor(CV_Mat.at<T>(i, j));
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    CV_Mat.at<T>(i, j) = std::floor(CV_Mat.at<T>(i, j));
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }


    template<typename T>
    static void ceild(cv::Mat& InputArray) {

     
        auto rows = InputArray.rows;
        auto columns = InputArray.cols;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        block_num < 128 ? actual_thread_num = 1 : void(0);
        QVector<QFuture<void>> futures;
        for (int i_ = 0; i_ < actual_thread_num - 1; i_++) {
            start_row = i_ * block_num;
            end_row = (i_ + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&InputArray, start_row, end_row, columns]() -> void {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            InputArray.at<T>(i, j) = std::ceil(InputArray.at<T>(i, j));
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    InputArray.at<T>(i, j) = std::ceil(InputArray.at<T>(i, j));
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }

    }




    //! note 所有的参数都必须是整数
    static void Mod(cv::Mat& InputArray, cv::Mat& OutputArray,int value) {

        if (InputArray.empty()) {

            throw VectorNullException("InputArray is empty exception");
        }


        //! 取模运算仅针对于整数矩阵
        InputArray.convertTo(InputArray, CV_32S);

        auto rows = InputArray.rows;
        auto columns = InputArray.cols;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;
        OutputArray = cv::Mat::zeros(InputArray.rows, InputArray.cols, InputArray.type());
        block_num < 128 ? actual_thread_num = 1 : void(0);

        QVector<QFuture<void>> futures;
        for (int i_ = 0; i_< actual_thread_num - 1; i_++) {
            start_row = i_ * block_num;
            end_row = (i_ + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&InputArray,&OutputArray, value,start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            OutputArray.at<int>(i, j) = InputArray.at<int>(i, j) % value;
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    OutputArray.at<int>(i, j) = InputArray.at<int>(i, j) % value;
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }



    }







    template <typename T,typename U>
    static void thresholdProcessing(QVector<QVector<T>>& Array2D, const U& threshold) {
        auto rows = Array2D.size();
        auto columns = Array2D[0].size();
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        if (block_num == 0) {

            actual_thread_num = 1;
        }

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Array2D, start_row, end_row, columns, threshold]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            Array2D[i][j] > threshold ? Array2D[i][j] = threshold : void(0);

                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    Array2D[i][j] > threshold ? Array2D[i][j] = threshold : void(0);
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }



    template<typename T,typename U = qint16>
    static void astype(QVector<QVector<T>>& Array2D_T, QVector<QVector<U>>& Array2D_U) {
       
        if (Array2D_T.isEmpty()) {

            throw VectorNullException("The converted matrix is empty exception ");
        }

        auto rows = Array2D_T.size();
        auto columns = Array2D_T[0].size();
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        if (block_num == 0) {

            actual_thread_num = 1;
        }

        Array2D_U.resize(rows);
        for (QVector<U>& Array1D_U : Array2D_U) {

            Array1D_U.resize(columns);

        }

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Array2D_T,&Array2D_U, start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            Array2D_U[i][j] = static_cast<U>(Array2D_T[i][j]);

                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    Array2D_U[i][j] = static_cast<U>(Array2D_T[i][j]);
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }



    }


    template<typename T,typename U=unsigned short>
    static void array2D_Convert_CV_Mat(const QVector<QVector<T>>& Array2D,cv::Mat& CV_Mat) {
        
        if (Array2D.isEmpty()) {

            throw VectorNullException("The converted matrix is empty exception ");
        }

        auto rows = Array2D.size();
        auto columns = Array2D[0].size();
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        if (block_num == 0) {

            actual_thread_num = 1;
        }

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Array2D, &CV_Mat,start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                           
                            CV_Mat.at<U>(i, j) = Array2D[i][j];
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    CV_Mat.at<U>(i, j) = Array2D[i][j];
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }



    template <typename T,typename U>
    static void CV_Mat_convert__Array2D(QVector<QVector<T>>& Array2D, const cv::Mat& CV_Mat) {
        auto rows = CV_Mat.size().height;
        auto columns =CV_Mat.size().width;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        Array2D.resize(rows);

        for (QVector<T>& Array1D : Array2D) {

            Array1D.resize(columns);
        }

        block_num < 128 ? actual_thread_num = 1 : void(0);
        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Array2D, &CV_Mat, start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            Array2D[i][j] = CV_Mat.at<U>(i, j);
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    Array2D[i][j] = CV_Mat.at<U>(i, j);
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }





    template<typename T,typename U>
    static void meshgrid(cv::Mat& OutputArray_1, cv::Mat& OutputArray_2,  T min_v_1, T max_v_1, T min_v_2, T max_v_2) {

        int rows    = static_cast<int>(max_v_2 - min_v_2);
        int columns = static_cast<int>(max_v_1 - min_v_1);

        if (rows == 0 || columns == 0) {

            throw VectorNullException("Exception throw , rows or cols is zero  ");
        }

        int size_1 = max_v_1 - min_v_1;
        int size_2 = max_v_2 - min_v_2;
        QVector<T> tmpVec1;
        QVector<T> tmpVec2;
        tmpVec1.resize(size_1);
        tmpVec2.resize(size_2);

        for (size_t i = 0; i < size_1; i++){
            try {

                tmpVec1[i] = min_v_1++;
            }
            catch (const std::exception& outOfRangeException) {

                HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
            }


        }

        for (size_t i = 0; i < size_2; i++) {
            try {

                tmpVec2[i] = min_v_2++;
            }
            catch (const std::exception& outOfRangeException) {

                HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
            }
        }

        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        block_num < 128 ? actual_thread_num = 1 : void(0);
        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&OutputArray_1,&OutputArray_2, &tmpVec1,&tmpVec2, start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            OutputArray_1.at<U>(i, j) = tmpVec1[j];
                            OutputArray_2.at<U>(i, j) = tmpVec2[i];
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });

            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    OutputArray_1.at<U>(i, j) = tmpVec1[j];
                    OutputArray_2.at<U>(i, j) = tmpVec2[i];
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }

    }
    




   /* template<typename T>
    static bool isNan(cv::Mat& Inputarray) {
        if (Inputarray.empty) {

            throw VectorNullException();
        }

        auto rows = Inputarray.rows;
        auto columns = Inputarray.cols;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        block_num < 128 ? actual_thread_num = 1 : void(0);

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Inputarray, start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            std::isnan(Inputarray.at<T>(i, j)) ? is_nan.store(true), return true : void(0);

                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    std::isnan(Inputarray.at<T>(i, j)) ? is_nan.store(true), return true : void(0);
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }

        return false;
    }*/





    template <typename T, typename U>
    static void thresholdProcessing(cv::Mat& Array2D, const U& threshold) {

        if (Array2D.empty()) {

            throw VectorNullException("InputArray is Null Exception ");
        }


        auto rows = Array2D.rows;
        auto columns = Array2D.cols;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;
        
        block_num < 128 ? actual_thread_num = 1 : void(0);
        
        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&Array2D, start_row, end_row, columns, threshold]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {
                            
                            Array2D.at<T>(i, j) == threshold ? Array2D.at<T>(i, j) = 0 : void(0);

                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    Array2D.at<T>(i, j) == threshold ? Array2D.at<T>(i, j) = 0 : void(0);
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }



    //! 将一个复数矩阵进行求相位角操作 返回的相位角规定在 [-Π,Π]
    template<typename T>
    static void angle(const cv::Mat& InputArray_Real,const cv::Mat& InputArray_Imag, cv::Mat& OutputArray) {
        
        if (InputArray_Imag.empty() && InputArray_Real.empty() || InputArray_Real.size()!=InputArray_Imag.size() ) {

            throw VectorNullException("The inputArray is empty when calculating phase angle");

        }

        auto rows = InputArray_Real.size().height;
        auto columns = InputArray_Real.size().width;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        block_num < 128 ? actual_thread_num = 1 : void(0);
        
        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&InputArray_Real,&InputArray_Imag,&OutputArray, start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            OutputArray.at<T>(i, j) = std::atan2(InputArray_Imag.at<T>(i, j), InputArray_Real.at<T>(i, j));
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {
                    
                    OutputArray.at<T>(i, j) = std::atan2(InputArray_Imag.at<T>(i, j), InputArray_Real.at<T>(i, j));
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }




    template<typename T>
    static  std::vector<T> booleanArrayIndexing(cv::Mat& InputArray, cv::Mat& mask) {

        if (InputArray.empty() || mask.empty()) {

            throw VectorNullException("InputArray is Null Exception ");
        }

        typedef           std::vector<T>::iterator iterator;
        iterator          postion;
        auto              rows = InputArray.size().height;
        auto              columns = InputArray.size().width;
        int               os_remain_thread_num = QThread::idealThreadCount() - 2;
        int               actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int               block_num = rows / actual_thread_num;
        int               start_row = 0;
        int               end_row = 0;
        int               end_index = actual_thread_num - 1;
        std::vector<T>    result;
        int               result_size = 0;
        std::vector<std::vector<T>> index_vector(actual_thread_num);
        
        block_num < 128 ? actual_thread_num = 1 : void(0);

        QVector<QFuture<void>> futures;
        for (int i_ = 0; i_ < actual_thread_num - 1; i_++) {
            start_row = i_ * block_num;
            end_row = (i_ + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&InputArray, &mask,i_,&index_vector, start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            //mask.at<bool>(i, j) == 1 ? index_vector[i_].emplace_back(InputArray.at<T>(i, j)) : void(0);
                            mask.at<bool>(i, j) == 1 ? index_vector[i_].push_back(InputArray.at<T>(i, j)) : void(0);
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    //mask.at<bool>(i, j) == 1 ? index_vector[end_index].emplace_back(InputArray.at<T>(i, j)) : void(0);
                    mask.at<bool>(i, j) == 1 ? index_vector[end_index].push_back(InputArray.at<T>(i, j)) : void(0);
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }

        for (int i = 0; i < index_vector.size(); i++) {

            result_size += index_vector[i].size();
        }

        result.resize(result_size);
        postion = result.begin();

        for (int i = 0; i < actual_thread_num; i++) {
                if (index_vector[i].size()==0) {

                    continue;
                }

                postion=std::copy(index_vector[i].begin(), index_vector[i].end(), postion);
            }

        std::sort(result.begin(), result.end());
        return result;
    }


    template<typename T>
    static void reshape(Eigen::VectorX<T>& vector, int row, int col, Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& result) {
        if (vector.rows() == 0 || vector.cols() == 0) {

            throw VectorNullException("Reshape func input is null exception");
        }

        int index = 0;
        for (int m = 0; m < row; m++) {
            for (int n = 0; n < col; n++) {
                try{
                    result.coeffRef(m, n) = vector[index++];
                }
                catch (const OutOfRangeException& outOfRangeException) {

                    HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
                }
            }
        }
    }




    template <typename T>
    static std::pair<std::vector<int>, std::vector<int>> where(const cv::Mat& InputArray, const T& value) {
        if (InputArray.empty()) {

            throw VectorNullException("InputArray is Null Exception ");
        }

        std::pair<std::vector<int>,std::vector<int>> indexVecPair;
        std::mutex mutex_;

        auto rows = InputArray.size().height;
        auto columns = InputArray.size().width;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;
        
        block_num < 128 ? actual_thread_num = 1 : void(0);
        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&InputArray, &value,&indexVecPair,&mutex_,start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    for (int j = 0; j < columns; j++) {
                        try {

                            InputArray.at<T>(i, j) == value ? indexVecPair.first.push_back(i),indexVecPair.second.push_back(j) : void(0) ;
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    InputArray.at<T>(i, j) == value ? indexVecPair.first.push_back(i), indexVecPair.second.push_back(j) : void(0);
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }

        return indexVecPair;
    }


    template<typename T>
    static std::vector<T> createHanningWindow(int window_size) {
        std::vector<T> window(window_size);
        for (int i = 0; i < window_size; i++) {

            window[i] = 0.5 * (1 - std::cos(2 * M_PI * i / (static_cast<double>(window_size) - 1)));


        }
        return window;
    }



    template<typename T>
    static void getMask(cv::Mat& InputArray, const T& value ,cv::Mat& mask) {
        
        if (InputArray.empty() || mask.empty()) {

            throw VectorNullException("InputArray is Null Exception ");
        }

        auto rows = InputArray.size().height;
        auto columns = InputArray.size().width;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        block_num < 128 ? actual_thread_num = 1 : void(0);

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&InputArray,&mask,value,start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                            InputArray.at<T>(i, j) == value ? mask.at<uchar>(i,j) = 1 : void(0);
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                    InputArray.at<T>(i, j) == value ? mask.at<uchar>(i,j) = 1 : void(0);
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }


    //! 1-8新增 这里有一个默认值为0
    template<typename T>
    static void getMaskedMatrix(cv::Mat& InputArray, cv::Mat& mask,T value=0) {
        if (InputArray.empty() || mask.empty()) {

            throw VectorNullException("InputArray is Null Exception ");
        }

        if (InputArray.size() != mask.size()) {

            throw ShapeException("Inputarray shape  not  equal mask shape ,can not mask ");
        }


        auto rows    = InputArray.size().height;
        auto columns = InputArray.size().width;
        int  os_remain_thread_num = QThread::idealThreadCount() - 2;
        int  actual_thread_num = os_remain_thread_num <= 0 ? 1 : 4;
        int  block_num = rows / actual_thread_num;
        int  start_row = 0;
        int  end_row = 0;

        block_num < 128 ? actual_thread_num = 1 : void(0);

        QVector<QFuture<void>> futures;
        for (int i = 0; i < actual_thread_num - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            QFuture<void> future = QtConcurrent::run([&InputArray, &mask,value, start_row, end_row, columns]() {
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < columns; j++) {
                        try {

                           mask.at<unsigned char>(i,j) == 255 ? InputArray.at<T>(i, j) = value : void(0);
                        }
                        catch (const std::exception& outOfRangeException) {

                            HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                        }
                    }
                }
                });
            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }

        for (int i = end_row; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                try {

                   mask.at<unsigned char>(i,j) == 255 ? InputArray.at<T>(i, j) = value : void(0);
                }
                catch (const std::exception& outOfRangeException) {

                    HIOlAB_CXX_14::ExceptionInfoPrint(outOfRangeException.what());
                }
            }
        }
    }


    //! non sync use 1*n  note: use o3 optimize, do not use emplace_back member function and emplace function 
    template<typename T>
    static void Update(const cv::Mat& Inputarray, const cv::Mat& mask, std::vector<T>& result,bool flag) {
        if (Inputarray.empty() || mask.empty()) {

            throw VectorNullException();
        }

        if (Inputarray.cols != mask.cols) {

            throw ShapeException();
        }

        result.resize(mask.cols- cv::countNonZero(mask));
        int index = 0;
        for (int i = 0; i < Inputarray.cols; i++) {

            if (mask.at<uchar>(0, i) == 0) {

                result[index++]= Inputarray.at<T>(0, i);
            }
        }
    }


    //! unused 
    template<typename T>
    static void  Update(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& InputArray, const Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic>& mask, 
                        std::vector<T>& result){

        if (InputArray.rows() == 0 || InputArray.cols() == 0 || mask.rows() == 0 || mask.cols() == 0) {

            throw VectorNullException("InputArray or mask is empty in the Update(T Args&&)...\n");
        }


        if (InputArray.rows() != mask.rows() || InputArray.cols() != mask.cols()) {

            throw ShapeException("InpputArray shape != mask shape in the Update(T Args&&)....");
        }



        typedef std::vector<std::vector<T>>::iterator iterator;
        int current_use_threads = HIOlAB_CXX_14::SystemCurrentCanUseThreads();
        int block_num = InputArray.rows() / current_use_threads;
        int rows = InputArray.rows();
        int cols = InputArray.cols();
        int start_row = 0;
        int end_row = 0;
        int not_nan_value_count = (mask.array()==0).count();


        block_num <= 5 ? current_use_threads = 1 : void(0);

        std::vector<std::vector<T>> mid_result(current_use_threads);
        
        result.resize(not_nan_value_count);


        QVector<QFuture<void>> futures(current_use_threads - 1);

        for (int i = 0; i < current_use_threads - 1; i++) {
            start_row = i * block_num;
            end_row = (i + 1) * block_num;
            int count = ((mask(Eigen::seqN(start_row, end_row), Eigen::seqN(0, cols)).array() )== 0).count();
            QFuture<void> future = QtConcurrent::run([&InputArray, &mask, &mid_result, i, start_row, end_row, cols, count]() ->void {
                mid_result[i].resize(count);
                for (int m = start_row; m < end_row; m++) {
                    for (int n = 0; n < cols; n++) {
                        try {

                            mask.coeffRef(m, n) == 0 ? mid_result[i].emplac_back(InputArray.coeffRef(m, n)) : void(0);
                        }
                        catch (const OutOfRangeException& outOfRangeException) {

                            HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
                        }
                    }
                }
                });

            futures.append(future);
        }

        for (QFuture<void>& future : futures) {

            future.waitForFinished();
        }


        int count = ((mask(Eigen::seqN(end_row, rows), Eigen::seqN(0, cols))).array()==0).count();
        mid_result[current_use_threads - 1].resize(count);
        for (int m = end_row; m < rows; m++) {
            for (int n = 0; n < cols; n++) {
                try {
                    mask.coeffRef(m, n) == 0 ? mid_result[current_use_threads - 1].emplace_back(InputArray.coeffRef(m, n)) : void(0);
                }
                catch (const OutOfRangeException& outRangeException) {

                    HIOlAB_CXX_14::MessagePrint(outRangeException.what());

                }


            }
        }

        postion = result.begin();

        for (int i = 0; i < current_use_threads; i++) {
            if (mid_result[i].size() == 0) {

                continue;
            }

            postion = std::copy(mid_result[i].begin(), mid_result[i].end(), postion);
        }
    }


    template<typename T>
    static void median(std::vector<T>& vec,int &median) {

        if (vec.empty()) {

            throw VectorNullException();
        }
       
        if (vec.size() % 2 == 1) {

            median = vec[vec.size() / 2];
            return;
        }

        median = ( vec[vec.size() / 2 - 1] + vec[vec.size() / 2] )/ 2;
    }



    template<typename T>
    struct MapCompare {
        bool operator()(const T& lhs, const T& rhs) const  {

            return lhs< rhs;
        }

    };


    template<typename T>
    static void unique(cv::Mat& InputComplexArray,cv::Mat& OutputComplexArray) {
        
        if (InputComplexArray.empty()) {

            throw VectorNullException("Input array Null exceprion ");

        }

        std::multimap<T, T, HIOlAB_CXX_14::MapCompare<T>> complex_mult_map;
        cv::Mat real_col = InputComplexArray.col(0);
        cv::Mat imag_col = InputComplexArray.col(1);
        for (int  i = 0;i < InputComplexArray.rows; i++) {
           
            complex_mult_map.insert({ real_col.at<T>(i, 0),imag_col.at<T>(i, 0) });
        }
       
        OutputComplexArray = cv::Mat::zeros(complex_mult_map.size(),2,InputComplexArray.type());
        int i = 0;
        for (const auto& complex_ : complex_mult_map) {

            OutputComplexArray.at<T>(i, 0) = complex_.first;
            OutputComplexArray.at<T>(i, 1) = complex_.second;
            ++i;
        }
    }





    //! 不能用于整数 在c中使用的IEEE754来表示的nan值 不是并行版 可以用列分快优化 对于nan和inf都应该返回255 与python的版本的逻辑稍有不同
    template<typename T>
    static void isfinite(cv::Mat& InputArray, cv::Mat& mask) {
        if (InputArray.empty()) {

            throw VectorNullException();
        }

        for (int i = 0; i < InputArray.rows; i++){
            for (int j = 0; j < InputArray.cols; j++){

                if (std::isnan(InputArray.at<T>(i, j)) || InputArray.at<T>(i, j) == DBL_MAX || InputArray.at<T>(i, j) == DBL_MIN) {

                    mask.at<unsigned char>(i, j) = 255;
                }
            }
        }
    }



    //! like np.isnan(array) 没有列分块优化 应该使用一个分块计算的函数
    template<typename T>
    static void isnan(cv::Mat& InputArray, cv::Mat& mask) {
        if (InputArray.empty()) {

            throw VectorNullException();
        }

        for (int i = 0; i < InputArray.rows; i++) {
            for (int j = 0; j < InputArray.cols; j++) {

                if (std::isnan(InputArray.at<T>(i, j))) {

                    mask.at<unsigned char>(i, j) = 255;
                }
            }
        }
    }



    //! output a vector ,contain no nan value in inputArray, outputAray  is 1D
    template<typename T>
    static void updateMatrix(cv::Mat& InputArray,cv::Mat& OutputArray,int count) {
        if(InputArray.empty()) {

            throw VectorNullException();
        }
        OutputArray = cv::Mat::zeros(1, InputArray.cols - count,InputArray.type());
        int j = 0;
       
        for (int i = 0; i < InputArray.cols; i++){

            try {

                !std::isnan(InputArray.at<T>(0, i)) ? OutputArray.at<T>(0, j) = InputArray.at<T>(0, i), ++j : void(0);
            }
            catch (const std::exception& outOfRangeException){

                HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
            }
        }
    }


    //! remove nan value row ouputArray is 2D
    static void removeNanValue(cv::Mat& matrix, cv::Mat& mask, cv::Mat& outputArray) {

        if (matrix.empty() || mask.empty()) {

            throw VectorNullException("InputArray or mask is empty in the removeNanValue(T Args&&)...\n");
        }

        //if (mask.rows != matrix.rows  || mask.cols != matrix.cols) {
        //    
        //        throw ShapeException("InpputArray shape != mask shape in the  removeNanValue(T Args&&)....");
        //}
        int index = 0;
        for (int i = 0; i < matrix.rows; i++) {
            if (cv::countNonZero(mask.row(i)) == 0) {
                matrix.row(i).copyTo(outputArray.row(index));
                ++index;
            }
        }
        
    }


    //! Remove nan row in the InputArray by mask. The mask is [1,,,,1]
    template<typename T>
    static void removeNanValue(cv::Mat& InputArray, cv::Mat& mask, cv::Mat& OutputArray,bool flag) {

        if (InputArray.empty() || mask.empty()) {

            throw VectorNullException("InputArray or mask is empty in the removeNanValue(T Args&&)...\n");
        }

        if ( mask.rows > 1 || OutputArray.rows != (mask.cols-cv::countNonZero(mask)) || InputArray.cols != OutputArray.cols) {

            throw ShapeException("InpputArray shape != mask shape in the  removeNanValue(T Args&&)....");
        }
        int index = 0;
        for (int i = 0; i < mask.cols; i++) {
            if (mask.at<uchar>(0,i)==0) {
                InputArray.row(i).copyTo(OutputArray.row(index));
                ++index;
            }
        }
    }
   




    template<typename T>
    static void dot(cv::Mat& firstInputArray, cv::Mat& secondIputArray, cv::Mat& outputArray) {
        if (firstInputArray.empty() || secondIputArray.empty()) {

            throw VectorNullException();
        }
        
        if (firstInputArray.cols != secondIputArray.cols) {

            throw ShapeException();
        }
        
        T value = 0;
        outputArray = cv::Mat::zeros(1, firstInputArray.rows, firstInputArray.type());

        for (int i = 0; i < firstInputArray.rows; i++) {
            
            value = firstInputArray.row(i).dot(secondIputArray);
            outputArray.at < T>(0,i)= value;
        }
    }





    template <typename U>
    static void Print_U_Char_Matrix(cv::Mat& Mat) {

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 10; j++) {

                std::cout << static_cast<int>(Mat.at<U>(i, j)) << " ";
            }

            std::cout << "\n";
            std::cout << "***********" << "\n";
        }

        for (int i = Mat.rows - 1; i < Mat.rows; i++) {
            for (int j = 0; j < Mat.cols; j++) {

                std::cout << static_cast<int>(Mat.at<U>(i, j)) << " ";
            }

            std::cout << "\n";
            std::cout << "***********" << "\n";
        }
    }



    template <typename U>
    static void Print_Matrix(cv::Mat& Mat) {

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < Mat.cols; j++) {

                std::cout << Mat.at<U>(i, j) << " ";
            }

            std::cout << "\n";
            std::cout << "***********" << "\n";
        }

    }


    template<typename U>
    static void Print_Complex_Matrix(cv::Mat& Complex_Mat) {
   
        for (int i = 0; i < 1; ++i) {
            for (int j = 0; j < 10; ++j) {
                std::complex<U> value = Complex_Mat.at<std::complex<U>>(i, j);
                std::cout << "(" << value.real() << "," << value.imag() << "j )" << " ";

            }
            std::cout << "\n";
            std::cout << "***********" << "\n";
    }

    }





};
