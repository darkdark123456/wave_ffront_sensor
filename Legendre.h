#pragma once
/*****************************************************************//**
 * \file   Legendre.h
 * \brief  使用Legendre多项式进行目标特征的提取
 * 
 * \author Administrator
 * \date   December 2023
 *********************************************************************/


#include    <Eigen/Dense>
#include    <Eigen/core>
#include	<opencv.hpp>
#include	<opencv2/core/eigen.hpp>
#include	"CustomException.h"


namespace Legendre {


	//! 递归获得一维中具有特定阶数的勒让德多项式的值
	//! Pn(X)=(2n-1)/n*X.Pn-1(X)-(n-1)/n*Pn-2(X)
	static cv::Mat legendrePolynomial(int order, cv::Mat& x2d) {

		cv::Mat l2d;
		switch (order) {
		case 0:
			l2d = cv::Mat::ones(x2d.rows, x2d.cols, CV_64F);
			break;
		case 1:
			l2d = x2d.clone();
			break;
		default:

			l2d = (2.0 * order - 1) / order * x2d.mul(legendrePolynomial(order - 1, x2d)) - (order - 1.0) / order * legendrePolynomial(order - 2, x2d);
			break;
		}
		return l2d;
	}



	//! 递归获得勒让德多项式的微分
	static cv::Mat legendreDerivative(int order, cv::Mat& x2d) {

		cv::Mat d2d;
		switch (order) {
		case 0:
			d2d = cv::Mat::zeros(x2d.rows,x2d.cols,x2d.type());
			break;
		case 1:
			d2d = cv::Mat::ones(x2d.rows, x2d.cols, x2d.type());
			break;
		default:
			float cofficient = (2. * order - 1) / order;
			d2d = cofficient * legendrePolynomial(order - 1, x2d) + cofficient * x2d.mul(legendreDerivative(order - 1, x2d)) - (order - 1.) / order* legendreDerivative(order - 2, x2d);
			break;
		}
		return d2d;
	}



	//! 得到一维中的P1型勒让德多项式模及其一阶导数
	static void  legendreP1(cv::Mat& X2D,cv::Mat& orders,std::vector<cv::Mat> & L3D,std::vector<cv::Mat>& LX3D) {
		int  order_num = orders.rows * orders.cols;
		int  order;

		L3D.resize(order_num);
		LX3D.resize(order_num);

		for (int idx = 0; idx < order_num; idx++) {
			order = orders.at<int>(0, idx);
				try{
					
					L3D.at(idx)  = legendrePolynomial(order, X2D);
					LX3D.at(idx) = legendreDerivative(order, X2D);
				}

				catch (const std::exception& outOfRangeException) {

					HIOlAB_CXX_14::MessagePrint(outOfRangeException.what());
				}
			}
		
	}

	//! 计算勒让德多项式基于（x,y,n,m）
	static void legendre_xynm(  cv::Mat& x2d,cv::Mat& y2d,cv::Mat& nld,cv::Mat& mld,
								std::vector<cv::Mat>& Z3D,
								std::vector<cv::Mat>& ZX3D,
								std::vector<cv::Mat>& ZY3D) {
		std::vector<cv::Mat> LM3D; 
		std::vector<cv::Mat> LMY3D;
		std::vector<cv::Mat> LN3D;
		std::vector<cv::Mat> LNX3D;

		legendreP1(y2d, mld, LM3D, LMY3D);
		legendreP1(x2d, nld, LN3D, LNX3D);

		Z3D.resize(LM3D.size());
		ZX3D.resize(LM3D.size());
		ZY3D.resize(LMY3D.size());

		HIOlAB_CXX_14::VectorMul<float>(LN3D,LM3D,Z3D);
		HIOlAB_CXX_14::VectorMul<float>(LNX3D, LM3D,ZX3D);
		HIOlAB_CXX_14::VectorMul<float>(LN3D, LMY3D ,ZY3D);
	}


	//! jld 转换为(nld,mld)
	static void convert_j_to_nm(cv::Mat& jld, cv::Mat& nld, cv::Mat& mld) {

		nld = cv::Mat::zeros(jld.rows, jld.cols,jld.type());
		mld = cv::Mat::zeros(jld.rows, jld.cols, jld.type());
		cv::Mat tmp;
		jld.convertTo(jld, CV_64F);

		cv::sqrt(jld, tmp);
		cv::Mat bld = tmp.clone();

		HIOlAB_CXX_14::ceild<double>(bld);
		cv::pow(bld, 2, tmp);

		cv::Mat ald = tmp - jld + 1;

		HIOlAB_CXX_14::Mod(ald, tmp, 2);
		tmp.convertTo(tmp, CV_64F);
		ald.convertTo(ald, CV_64F);
		cv::Mat nsm = (-0.5*ald).mul(1.0- tmp) + ((0.5  * (ald - 1.0)).mul(tmp));
		HIOlAB_CXX_14::floor<double>(nsm);

		cv::Mat nam = 2 * bld - cv::abs(nsm) - 2;
		HIOlAB_CXX_14::floor<double>(nam);

		nld = (nam + nsm) * 0.5;
		HIOlAB_CXX_14::floor<double>(nld);
		mld = (nam - nsm) * 0.5;
		HIOlAB_CXX_14::floor<double>(mld);
	}



	//! 计算勒让德多项式基于（x,y,j,c）  
	static void legendre_xyjc( cv::Mat& x2d, cv::Mat& y2d, cv::Mat& jld, 
							   cv::Mat& cld,
							   cv::Mat& z2d,cv::Mat& zx2d,cv::Mat& zy2d,
							   std::vector<cv::Mat>& zm3d,std::vector<cv::Mat>& zxm3d,std::vector<cv::Mat>& zym3d,
							   cv::Mat& mld,cv::Mat& nld) 
	{
		if (cld.empty()) {

			cld = cv::Mat::ones(jld.rows, jld.cols, jld.type());
		}

		convert_j_to_nm(jld, nld, mld);
		std::vector<cv::Mat> tmp;

		nld.convertTo(nld, CV_32S);
		mld.convertTo(mld, CV_32S);

		legendre_xynm(x2d, y2d, nld, mld,zm3d,zxm3d,zym3d); 
		cld.convertTo(cld, CV_64F);

		tmp.clear();
	 	HIOlAB_CXX_14::VectorMul<double>(zm3d, cld, tmp);
		HIOlAB_CXX_14::sum<double>(tmp, z2d);

		tmp.clear();
		HIOlAB_CXX_14::VectorMul<double>(zxm3d, cld, tmp);
		HIOlAB_CXX_14::sum<double>(tmp, zx2d);
		
		tmp.clear();
		HIOlAB_CXX_14::VectorMul<double>(zym3d, cld, tmp);
		HIOlAB_CXX_14::sum<double>(tmp, zy2d);

		tmp.clear();
	}



	//! 勒让德多项式分解
	static void decompose(cv::Mat& z2d, cv::Mat& x2d, cv::Mat& y2d, cv::Mat& jld, cv::Mat& xy_norm,cv::Mat& z2d_rec,cv::Mat& coef_est) {

		double x_norm;
		double y_norm;
		if (xy_norm .empty()) {
			double max_v, min_v;
			cv::Point max_pos, min_pos;
			cv::minMaxLoc(x2d,&min_v, &max_v, &max_pos, &min_pos);
			x_norm = (max_v - min_v) / 2;
			cv::minMaxLoc(y2d, &min_v, &max_v, &max_pos, &min_pos);
			y_norm = (max_v - min_v) / 2;
			xy_norm = cv::Mat(cv::Size(2, 1), CV_64F);
			xy_norm.at<double>(0, 0) = x_norm;
			xy_norm.at<double>(0, 1) = y_norm;


		}

		else{
			x_norm = xy_norm.at<double>(0, 0);
			y_norm = xy_norm.at<double>(0, 1);
		}
		cv::Mat z2d_,cld,zx2d, zy2d;
		std::vector<cv::Mat> zm3d, zxm3d, zym3d;
		cv::Mat mld, nld;
		cv::Mat xn2d = (x2d - cv::mean(x2d)[0]) / x_norm;
		cv::Mat yn2d = (y2d - cv::mean(y2d)[0]) / y_norm;
		legendre_xyjc(xn2d, yn2d, jld, cld, z2d_, zx2d, zy2d, zm3d, zxm3d, zym3d, mld, nld);
		zx2d.release();
		zy2d.release();
		cld.release();
		mld.release();
		nld.release();
		zxm3d.resize(0);
		zym3d.resize(0);

		cv::Mat zm2d;
		cv::Mat z1d = z2d.reshape(1, z2d.rows * z2d.cols).t();
		HIOlAB_CXX_14::DimensionReduce<double>(zm3d, zm2d);
		cv::Mat mask = cv::Mat::zeros(z1d.rows, z1d.cols, CV_8U);
		HIOlAB_CXX_14::isfinite<double>(z1d, mask);

		cv::Mat zm2d_(z1d.cols - cv::countNonZero(mask), zm2d.cols, CV_64F);
		HIOlAB_CXX_14::removeNanValue<double>(zm2d, mask,zm2d_,0);
		zm2d.release();

		Eigen::MatrixXd matrix(zm2d_.rows, zm2d_.cols);
		cv::cv2eigen(zm2d_, matrix);
		zm2d_.release();

		Eigen::MatrixXd matrix_inverse = matrix.completeOrthogonalDecomposition().pseudoInverse();
		std::vector<double> z1d_nan_vector;
		HIOlAB_CXX_14::Update<double>(z1d, mask, z1d_nan_vector, 0);
		z1d.release();

		cv::Mat tmp(matrix_inverse.rows(), matrix_inverse.cols(), CV_64F);
		cv::eigen2cv(matrix_inverse, tmp);

		cv::Mat tmp_2 = cv::Mat(z1d_nan_vector).t();

		HIOlAB_CXX_14::dot<double>(tmp, tmp_2, coef_est);


		zm3d.resize(0);
		legendre_xyjc(xn2d,yn2d,jld,coef_est,z2d_rec,zx2d,zy2d,zm3d,zxm3d,zym3d,mld,nld);

	}


	
	/**
	 * .
	 * \param sx2d		x斜率分布  可能会出现nan值 需要特殊处理
	 * \param sy2d      y斜率分布  可能会出现nan值 需要特殊处理
	 * \param x2d		x坐标
	 * \param y2d		y坐标
	 * \param jld		序列
	 * \param xy_norm	xy方向上的归一化参数
	 */
	static void intergrate(cv::Mat& sx2d, cv::Mat& sy2d, cv::Mat& x2d, cv::Mat& y2d,
						   cv::Mat& jld, cv::Mat&z2d_wfr ,cv::Mat&zx , cv::Mat&zy , cv::Mat&wfr_coefs , 
						   std::vector<cv::Mat>& zxm3d, std::vector<cv::Mat>& zym3d, cv::Mat& xn2d, cv::Mat& yn2d,cv::Mat& xy_norm) {

		double x_norm;
		double y_norm;
		if (xy_norm.empty()) {

			double max_v, min_v;
			cv::Point max_pos, min_pos;
			cv::minMaxLoc(x2d, &min_v, &max_v, &max_pos, &min_pos);
			x_norm = (max_v - min_v) / 2;
			cv::minMaxLoc(y2d, &min_v, &max_v, &max_pos, &min_pos);
			y_norm = (max_v - min_v) / 2;
			xy_norm = cv::Mat(cv::Size(2, 1), CV_64F);
			xy_norm.at<double>(0, 0) = x_norm;
			xy_norm.at<double>(0, 1) = y_norm;

		}

		else {

			x_norm = xy_norm.at<double>(0, 0);
			y_norm = xy_norm.at<double>(0, 1);
		}


		cv::Mat cld, z2d, zx2d, zy2d;
		std::vector<cv::Mat> zm3d;
		cv::Mat mld, nld;
		xn2d = (x2d - cv::mean(x2d)[0]) / x_norm;
		yn2d = (y2d - cv::mean(y2d)[0]) / y_norm;

		cv::Mat sxn2d = sx2d * x_norm;
		cv::Mat syn2d = sy2d * y_norm;


		try{

			legendre_xyjc(xn2d, yn2d, jld, cld, z2d, zx2d, zy2d, zm3d, zxm3d, zym3d, mld, nld); 
		}

		catch (const VectorNullException& matrixNullException) {


			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}

		catch (const MatrixMultDimException& matrixMulDimException) {

			HIOlAB_CXX_14::MessagePrint(matrixMulDimException.what());

		}

		catch (const ShapeException& matrixShapeException) {

			HIOlAB_CXX_14::MessagePrint(matrixShapeException.what());
		}
	
		cv::Mat zxm2d;
		
		try{
			
			HIOlAB_CXX_14::DimensionReduce<double>(zxm3d, zxm2d);
		}

		catch (const VectorNullException& matrixNullMatrixException){

			HIOlAB_CXX_14::MessagePrint(matrixNullMatrixException.what());
		}
		

		cv::Mat zym2d;

		try{
			
			HIOlAB_CXX_14::DimensionReduce<double>(zym3d, zym2d);
		}
		catch (const VectorNullException& matrixNullMatrixException){

			HIOlAB_CXX_14::MessagePrint(matrixNullMatrixException.what());
		}

		

		cv::Mat sm2d;
		cv::vconcat(zxm2d, zym2d, sm2d);

		cv::Mat ssn;
		cv::vconcat(sxn2d, syn2d, ssn);
	

		cv::Mat ssn_ravel = ssn.reshape(1, ssn.rows * ssn.cols).t();
		cv::Mat ssn_ravel_valid;


		cv::Mat mask = cv::Mat::zeros(ssn_ravel.rows, ssn_ravel.cols, CV_8U);
		try{
			
			
			HIOlAB_CXX_14::isfinite<double>(ssn_ravel, mask);
		}
		catch (const VectorNullException& matrixNullMatrix){

			HIOlAB_CXX_14::MessagePrint("In judge matrix value is finite operatortion ,matrix null exception");
		}


		
		int nan_count = cv::countNonZero(mask);


		try{

			HIOlAB_CXX_14::updateMatrix<double>(ssn_ravel, ssn_ravel_valid, nan_count);
		}

		catch (const std::exception& matrixNullException){

			HIOlAB_CXX_14::MessagePrint("In ravel operatortion,matrix is null exception");
		}

		
		ssn_ravel.release();
		cv::transpose(mask, mask);
		
		cv::Mat nan_sm2d(sm2d.rows - nan_count, sm2d.cols, sm2d.type());
		HIOlAB_CXX_14::removeNanValue(sm2d, mask,nan_sm2d);
	

		Eigen::MatrixXd eigen_sm2d(nan_sm2d.rows,nan_sm2d.cols);
		cv::cv2eigen(nan_sm2d, eigen_sm2d);


		Eigen::MatrixXd sm2d_inverse = eigen_sm2d.completeOrthogonalDecomposition().pseudoInverse(); //! 启用 O3 优化后 SVD的速度在75ms
		nan_sm2d = 0;
		cv::eigen2cv(sm2d_inverse, nan_sm2d);

		try{
			
			HIOlAB_CXX_14::dot<double>(nan_sm2d, ssn_ravel_valid, wfr_coefs);
			
		}

		catch (const VectorNullException& matrixNullException){

			HIOlAB_CXX_14::MessagePrint("In matrix dpt operatortion,inputarray is null exception");
		}

		catch (const ShapeException& matrixShapeException) {

			HIOlAB_CXX_14::MessagePrint("In matrix dpt operatortion,inputarray shape exception");
		}

		z2d = zx2d = zy2d = 0;
		std::vector<cv::Mat> tmp;

		try{
			
			HIOlAB_CXX_14::VectorMul<double>(zxm3d, wfr_coefs, tmp);
			HIOlAB_CXX_14::sum<double>(tmp, zx2d);

			HIOlAB_CXX_14::VectorMul<double>(zm3d, wfr_coefs, tmp);
			HIOlAB_CXX_14::sum<double>(tmp, z2d);

			HIOlAB_CXX_14::VectorMul<double>(zym3d, wfr_coefs, tmp);
			HIOlAB_CXX_14::sum<double>(tmp, zy2d);
		}

		catch (const VectorNullException& matrixNullException ){


			HIOlAB_CXX_14::MessagePrint(matrixNullException.what());
		}
		
		catch (const MatrixMultDimException& matrixMulDimException) {

			HIOlAB_CXX_14::MessagePrint(matrixMulDimException.what());

		}

		catch (const ShapeException& matrixShapeException) {

			HIOlAB_CXX_14::MessagePrint(matrixShapeException.what());
		}

	}
};
