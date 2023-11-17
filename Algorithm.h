#pragma once
#include <qvector.h>
#include <algorithm>
#include <qdebug.h>
#include <QtConcurrent/QtConcurrent>



namespace HIOlAB_CXX_14 {



    static void ExceptionInfoPrint(const char* const _Message) {

        qDebug() << "Exception throw:  " << _Message << "\n";
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

        if (block_num == 0) {

            actual_thread_num = 1;
        }

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
                            if (Array2D[i][j] > threshold) {

                                Array2D[i][j] = threshold;
                            }

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

                    if (Array2D[i][j] > threshold) {

                        Array2D[i][j] = threshold;
                    }
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




    
};
