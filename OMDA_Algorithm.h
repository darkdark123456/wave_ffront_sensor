#pragma once
#pragma warning(disable:26451)

/** Note:
 *  All code do not use emplace and emplace_back in std::vector and other container when use o3 optimilize,
 */

#include	<opencv.hpp>
#include	<limits>
#include	<fstream>
#include	<vector>
#include	"CustomException.h"
#include	"Algorithm.h"





namespace OMDA_Algorithm {


	static void remove2DTilt(cv::Mat& x2d,cv::Mat& y2d,cv::Mat& z2d,cv::Mat& z2d_res) {

		cv::Mat mask = cv::Mat::zeros(z2d.rows, z2d.cols, CV_8U);


		try{
			
			HIOlAB_CXX_14::isfinite<double>(z2d, mask);
		}
		catch (const VectorNullException& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}

		
		cv::Mat mask_flatten = mask.reshape(1, mask.rows * mask.cols).t();
		int nan_value_count = mask.rows * mask.cols - cv::countNonZero(mask);
		std::vector<double> z1d(nan_value_count);
		std::vector<double> x1d(nan_value_count);
		std::vector<double> y1d(nan_value_count);


		try{
			
			HIOlAB_CXX_14::Update<double>(z2d.reshape(1, z2d.rows * z2d.cols).t(), mask_flatten, z1d, 0);
			HIOlAB_CXX_14::Update<double>(x2d.reshape(1, x2d.rows * x2d.cols).t(), mask_flatten, x1d, 0);
			HIOlAB_CXX_14::Update<double>(y2d.reshape(1, y2d.rows * y2d.cols).t(), mask_flatten, y1d, 0);
		}

		catch (const VectorNullException& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}
		catch(const ShapeException& shapeException){

			HIOlAB_CXX_14::MessagePrint(shapeException.what());
		}

		cv::Mat matrix_h;
		cv::vconcat(cv::Mat::ones(1, x1d.size(), CV_64F), cv::Mat(x1d).t(), matrix_h);
		cv::vconcat(matrix_h, cv::Mat(y1d).t(), matrix_h);
		cv::transpose(matrix_h, matrix_h);

		Eigen::MatrixXd A(matrix_h.rows, matrix_h.cols);
		cv::cv2eigen(matrix_h, A);
		Eigen::Map<Eigen::VectorXd> B(z1d.data(), z1d.size());
		Eigen::JacobiSVD<Eigen::MatrixXd> SVD(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::VectorXd result = SVD.solve(B);

		if (z1d.size() == 0) {
			for (int i = 0; i < result.size(); i++) {

				result[i] = std::numeric_limits<double>::quiet_NaN();
			}
		}
		z2d_res = z2d - (result[1] * x2d + result[2] * y2d + result[0] );
	}



	static void remove2DSphere(cv::Mat& x2d, cv::Mat& y2d, cv::Mat& z2d,cv::Mat& z2d_res) {
		cv::Mat mask = cv::Mat::zeros(z2d.rows, z2d.cols, CV_8U);
		try{

			HIOlAB_CXX_14::isfinite<double>(z2d, mask);
		}
		catch (const VectorNullException& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}

		

		cv::Mat mask_flatten = mask.reshape(1, mask.rows * mask.cols).t();
		int nan_value_count = mask.rows * mask.cols - cv::countNonZero(mask);
		std::vector<double> z1d(nan_value_count);
		std::vector<double> x1d(nan_value_count);
		std::vector<double> y1d(nan_value_count);


		try{
			
			HIOlAB_CXX_14::Update<double>(z2d.reshape(1, z2d.rows * z2d.cols).t(), mask_flatten, z1d, 0);
			HIOlAB_CXX_14::Update<double>(x2d.reshape(1, x2d.rows * x2d.cols).t(), mask_flatten, x1d, 0);
			HIOlAB_CXX_14::Update<double>(y2d.reshape(1, y2d.rows * y2d.cols).t(), mask_flatten, y1d, 0);
		}

		catch (const VectorNullException& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}

		catch (const ShapeException& shapeException) {

			HIOlAB_CXX_14::MessagePrint(shapeException.what());
		}

		cv::Mat matrix_h;
		cv::Mat r1, r2;
		cv::pow(cv::Mat(x1d).t(), 2, r1);
		cv::pow(cv::Mat(y1d).t(), 2, r2);
		cv::vconcat(cv::Mat::ones(1, x1d.size(), CV_64F), cv::Mat(x1d).t(), matrix_h);
		cv::vconcat(matrix_h, cv::Mat(y1d).t(), matrix_h);
		cv::vconcat(matrix_h, r1 + r2, matrix_h);
		cv::transpose(matrix_h, matrix_h);

		Eigen::MatrixXd A(matrix_h.rows, matrix_h.cols);
		cv::cv2eigen(matrix_h, A);
		Eigen::Map<Eigen::VectorXd> B(z1d.data(), z1d.size());
		Eigen::JacobiSVD<Eigen::MatrixXd> SVD(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::VectorXd result = SVD.solve(B);

		if (z1d.size() == 0) {
			for (int i = 0; i < result.size(); i++) {

				result[i] = std::numeric_limits<double>::quiet_NaN();
			}
		}
		r1.release();
		r2.release();
		cv::pow(x2d, 2, r1);
		cv::pow(y2d, 2, r2);
		z2d_res = z2d - (result[1] * x2d + result[2] * y2d + result[0]+result[3]*(r1+r2));
	}

	

	static void calculate2DHeightFromSlope(cv::Mat& sx2d, cv::Mat& sy2d, cv::Mat& x2d, cv::Mat& y2d, cv::Mat& z2d_wfr) {

		int nrows = sx2d.rows;
		int ncols = sx2d.cols;
		constexpr double nan_double_value = std::numeric_limits<double>::quiet_NaN();

		cv::Mat mask;
		cv::Mat mask_sx2d = cv::Mat::zeros(sx2d.rows, sx2d.cols, CV_8U);
		cv::Mat mask_sy2d = cv::Mat::zeros(sy2d.rows, sy2d.cols, CV_8U);

		try{
			
			HIOlAB_CXX_14::isfinite<double>(sx2d, mask_sx2d);
			HIOlAB_CXX_14::isfinite<double>(sy2d, mask_sy2d);
		}

		catch (const std::exception& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}
		cv::bitwise_and(mask_sx2d, mask_sy2d, mask);



		//! expand in x direction		
		cv::Mat temp_nan_1 = cv::Mat::zeros(nrows, 1, CV_64F);
		cv::Mat temp_nan_2 = cv::Mat::zeros(nrows, 2, CV_64F);

		temp_nan_1 = nan_double_value;
		temp_nan_2 = nan_double_value;


		cv::Mat sx2d_expand;
		cv::hconcat(temp_nan_1, sx2d, sx2d_expand);
		cv::hconcat(sx2d_expand, temp_nan_2, sx2d_expand);

		cv::Mat x2d_expand;
		cv::hconcat(temp_nan_1, x2d, x2d_expand);
		cv::hconcat(x2d_expand, temp_nan_2, x2d_expand);

		cv::Mat expand_mask = cv::Mat::zeros(sx2d_expand.rows, sx2d_expand.cols, CV_8U);

		try{
			
			HIOlAB_CXX_14::isnan<double>(sx2d_expand, expand_mask);
		}
		catch (const std::exception& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}

		cv::Mat kernel = (cv::Mat_<uchar>(1, 5) << 1, 1, 0, 1, 0);

		cv::Mat dilated_expand_mask;

		//! 这里在膨胀计算前必须先镜像 据观察结果 scipy 和 opencv中 读取kernel的数据的顺序不同  
		cv::flip(kernel, kernel,1);
		std::cout << kernel << std::endl;
		cv::dilate(expand_mask, dilated_expand_mask, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
		cv::Mat mask_x = dilated_expand_mask.colRange(1, dilated_expand_mask.cols - 2) & (~expand_mask.colRange(1, expand_mask.cols - 2));

		//!  Expand in y - direction
		temp_nan_1.release();
		temp_nan_1 = cv::Mat::zeros(1, ncols, CV_64F);
		temp_nan_2.release();
		temp_nan_2 = cv::Mat::zeros(2, ncols, CV_64F);

		temp_nan_1 = nan_double_value;
		temp_nan_2 = nan_double_value;


		cv::Mat sy2d_expand;
		cv::vconcat(temp_nan_1, sy2d, sy2d_expand);
		cv::vconcat(sy2d_expand, temp_nan_2, sy2d_expand);

		cv::Mat y2d_expand;
		cv::vconcat(temp_nan_1, y2d, y2d_expand);
		cv::vconcat(y2d_expand, temp_nan_2, y2d_expand);

		temp_nan_1.release();
		temp_nan_2.release();


		expand_mask = cv::Mat::zeros(sy2d_expand.rows, sy2d_expand.cols, CV_8U);

		try{

			HIOlAB_CXX_14::isnan<double>(sy2d_expand, expand_mask);
		}
		catch (const std::exception& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}

		cv::transpose(kernel, kernel);

		dilated_expand_mask.release();
		cv::Mat dilated_mask_;
		cv::dilate(expand_mask, dilated_mask_, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
		cv::Mat mask_y = dilated_mask_.rowRange(1, dilated_mask_.rows - 2) & (~expand_mask.rowRange(1, expand_mask.rows - 2));

		int num = nrows * ncols;
		Eigen::SparseMatrix<int> matrix_dx(num, num);
		Eigen::SparseMatrix<int> matrix_dy(num, num);

		for (int i = 1; i < num; i++) {

			matrix_dx.coeffRef(i - 1, i - 1) = -1;
			matrix_dx.coeffRef(i - 1, i) = 1;
		}
		matrix_dx.coeffRef(num - 1, num - 1) = -1;


		for (int i = 1; i < num; i++) {

			matrix_dy.coeffRef(i - 1, i - 1) = -1;
		}
		matrix_dy.coeffRef(num - 1, num - 1) = -1;


		for (int i = ncols; i < num; i++) {

			matrix_dy.coeffRef(i - ncols, i) = 1;
		}

		
		matrix_dx.makeCompressed();
		matrix_dy.makeCompressed();



		cv::Mat matrix_gx3 = (sx2d_expand.colRange(1, sx2d_expand.cols - 2) + sx2d_expand.colRange(2, sx2d_expand.cols - 1));
		matrix_gx3 = matrix_gx3.mul((0.5 * (x2d_expand.colRange(2, x2d_expand.cols - 1)- x2d_expand.colRange(1, x2d_expand.cols - 2))));


		cv::Mat matrix_gy3 = (sy2d_expand.rowRange(1, sy2d_expand.rows - 2) + sy2d_expand.rowRange(2, sy2d_expand.rows - 1));
		matrix_gy3 = matrix_gy3.mul(0.5 * (y2d_expand.rowRange(2, y2d_expand.rows - 1) - y2d_expand.rowRange(1, y2d_expand.rows - 2)));


		constexpr double cof1 = 13.0 / 24.0;
		constexpr double cof2 = 1.0 / 13.0;

		cv::Mat mid_result_1 = (-cof2) * sx2d_expand.colRange(0, sx2d_expand.cols - 3) + sx2d_expand.colRange(1, sx2d_expand.cols - 2) + sx2d_expand.colRange(2, sx2d_expand.cols - 1)
							 - cof2 * sx2d_expand.colRange(3, sx2d_expand.cols);
		cv::Mat mid_result_2 =(x2d_expand.colRange(2, x2d_expand.cols - 1) - x2d_expand.colRange(1, x2d_expand.cols - 2))*cof1;
		cv::Mat matrix_gx5   = mid_result_1.mul(mid_result_2);
		mid_result_1.release();
		mid_result_2.release();

		mid_result_1 = (-cof2) * sy2d_expand.rowRange(0, sy2d_expand.rows - 3) + sy2d_expand.rowRange(1, sy2d_expand.rows - 2) + sy2d_expand.rowRange(2, sy2d_expand.rows - 1)
			- cof2 * sy2d_expand.rowRange(3, sy2d_expand.rows);
		mid_result_2= (y2d_expand.rowRange(2, y2d_expand.rows - 1) - y2d_expand.rowRange(1, y2d_expand.rows - 2))*cof1;
		cv::Mat matrix_gy5 = mid_result_1.mul(mid_result_2);

		mid_result_1.release();
		mid_result_2.release();
	


		matrix_gx3.copyTo(matrix_gx5, mask_x);
		matrix_gy3.copyTo(matrix_gy5, mask_y);


		sx2d_expand.release();
		sy2d_expand.release();
		x2d_expand.release();
		y2d_expand.release();
		matrix_gx3.release();
		matrix_gy3.release();
		

		mask_x.release();
		mask_y.release();
		mask_x = cv::Mat::zeros(matrix_gx5.rows, matrix_gx5.cols, CV_8U);
		mask_y = cv::Mat::zeros(matrix_gy5.rows, matrix_gy5.cols, CV_8U);


		try{
			
			HIOlAB_CXX_14::isfinite<double>(matrix_gx5, mask_x);
			HIOlAB_CXX_14::isfinite<double>(matrix_gy5, mask_y);
		}
		catch (const std::exception& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}


		cv::Mat mask_x_flatten = mask_x.reshape(1, mask_x.rows*mask_x.cols).t();
		cv::Mat mask_y_flatten = mask_y.reshape(1, mask_y.rows*mask_y.cols).t();
		

		Eigen::SparseMatrix<int> nan_matrix_dx;
		int row_ = mask_x_flatten.cols-cv::countNonZero(mask_x_flatten);
		std::vector<int> row_index;
		row_index.resize(row_);
		int count = 0;

		for (int i = 0; i < mask_x_flatten.cols; i++) {
			if (mask_x_flatten.at<uchar>(0, i) == 0) {

				row_index[count++] = i;
			}
		}

		nan_matrix_dx.resize(row_, matrix_dx.cols());
		nan_matrix_dx.reserve(row_ * matrix_dx.cols());
	
		matrix_dx = matrix_dx.transpose();
		for (int i = 0; i < row_index.size(); i++) {
			int row = row_index[i];
				for (typename Eigen::SparseMatrix<int>::InnerIterator iter(matrix_dx, row); iter; ++iter) {

					nan_matrix_dx.coeffRef(i, iter.row()) = iter.value();
				}
		}






		//! Dual calculate sparse matrix y
		Eigen::SparseMatrix<int> nan_matrix_dy;
		row_ = mask_y_flatten.cols - cv::countNonZero(mask_y_flatten);
		row_index.clear();
		row_index.resize(row_);
		count = 0;
		
		for (int i = 0; i < mask_y_flatten.cols; i++) {
			if (mask_y_flatten.at<uchar>(0, i) == 0) {

				row_index[count++] = i;
			}
		}



		nan_matrix_dy.resize(row_, matrix_dy.cols());
		nan_matrix_dy.reserve(row_* matrix_dy.cols());
		matrix_dy = matrix_dy.transpose();

		for (int i = 0; i < row_index.size(); i++) {
			int row = row_index[i];
			for (typename Eigen::SparseMatrix<int>::InnerIterator iter(matrix_dy, row); iter; ++iter) {

				nan_matrix_dy.coeffRef(i, iter.row()) = iter.value();
			}
		}


		//! 再将B1 B2表示出来 not async 
		std::vector<double> mgx5_nan_value_vec;
		std::vector<double> mgy5_nan_value_vec;

		try{
			
			HIOlAB_CXX_14::Update<double>(matrix_gx5.reshape(1, matrix_gx5.rows * matrix_gx5.cols).t(), mask_x_flatten, mgx5_nan_value_vec, 0);
			HIOlAB_CXX_14::Update<double>(matrix_gy5.reshape(1, matrix_gy5.rows * matrix_gy5.cols).t(), mask_y_flatten, mgy5_nan_value_vec, 0);
		}
		catch (const  VectorNullException& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}

		catch (const ShapeException& shapeException) {

			HIOlAB_CXX_14::MessagePrint(shapeException.what());
		}


		Eigen::SparseMatrix<int> matrix_d(nan_matrix_dx.rows() + nan_matrix_dy.rows(), std::max(nan_matrix_dx.cols(),nan_matrix_dy.cols()));
		matrix_d.reserve(nan_matrix_dx.size()+nan_matrix_dy.size());


		std::vector<Eigen::Triplet<int>> matrix_dx_triple_list;
		std::vector<Eigen::Triplet<int>> matrix_dy_triple_list;
		std::vector<Eigen::Triplet<int>> matrix_d_triple_list;

		//! Do not use emplace memeber func when use o3 optimilize  
		for (int k = 0; k < nan_matrix_dx.outerSize(); k++) {
			for (typename Eigen::SparseMatrix<int>::InnerIterator iterator(nan_matrix_dx, k); iterator; ++iterator) {

				matrix_dx_triple_list.push_back(Eigen::Triplet<int>(iterator.row(), iterator.col(), iterator.value()));
			}
		}


		for (int k = 0; k < nan_matrix_dy.outerSize(); k++) {
			for (typename Eigen::SparseMatrix<int>::InnerIterator iterator(nan_matrix_dy, k); iterator; ++iterator) {

				matrix_dy_triple_list.push_back(Eigen::Triplet<int>(iterator.row()+nan_matrix_dx.rows(), iterator.col(), iterator.value()));
			}
		}
		
		matrix_d_triple_list.reserve(matrix_dx_triple_list.size() + matrix_dy_triple_list.size());
		matrix_d_triple_list.insert(matrix_d_triple_list.end(), matrix_dx_triple_list.begin(), matrix_dx_triple_list.end());
		matrix_d_triple_list.insert(matrix_d_triple_list.end(), matrix_dy_triple_list.begin(), matrix_dy_triple_list.end());
		matrix_d.setFromTriplets(matrix_d_triple_list.begin(), matrix_d_triple_list.end());
		matrix_d.makeCompressed();

		Eigen::Map<Eigen::VectorX<double>> vec_gx5(mgx5_nan_value_vec.data(), mgx5_nan_value_vec.size());
		Eigen::Map<Eigen::VectorX<double>> vec_gy5(mgy5_nan_value_vec.data(), mgy5_nan_value_vec.size());
		Eigen::VectorX<double> matrix_g(vec_gx5.size() + vec_gy5.size());
		matrix_g << vec_gx5, vec_gy5;;

		Eigen::SparseMatrix<double> matrix_d_ = matrix_d.cast<double>();
		matrix_d_.makeCompressed();


		matrix_dx_triple_list.resize(0);
		matrix_dy_triple_list.resize(0);

	
		Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::NaturalOrdering<int>> solver;
		
		
		//! QR分解的结果不同
		
		solver.compute(matrix_d_);

		solver.info() == Eigen::Success ? void(0) : throw SparseQRException();

		Eigen::VectorX<double> result = solver.solve(matrix_g);

		solver.info() == Eigen::Success ? void(0) : throw SparseQRException();

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> z2d_hfli2(nrows, ncols);



		try{
			
			HIOlAB_CXX_14::reshape<double>(result, nrows, ncols, z2d_hfli2);
		}
		catch (const VectorNullException& matrixNullException){

			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}

		catch (const OutOfRangeException& outOfRangeException) {

			HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
		}
		

		z2d_wfr = cv::Mat::zeros(z2d_hfli2.rows(), z2d_hfli2.cols(),CV_64F);
		cv::eigen2cv(z2d_hfli2, z2d_wfr);

		try{
			
			HIOlAB_CXX_14::getMaskedMatrix<double>(z2d_wfr, mask, nan_double_value);
		}
		catch (const VectorNullException& matrxiNullException){
			
			HIOlAB_CXX_14::MessagePrint(matrxiNullException.what());
		}

		catch (const ShapeException& shapeException) {

			HIOlAB_CXX_14::MessagePrint(shapeException.what());
		}
		
	}

};



