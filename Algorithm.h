#pragma once
#include <qvector.h>
#include <algorithm>
#include <qdebug.h>

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
    void  Mean_Axis_3(const QVector<QVector<QVector<QVector<T>>>>& Array4D,int* dim,QVector<QVector<QVector<T>>>& Mean_Array3D) {
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


};
