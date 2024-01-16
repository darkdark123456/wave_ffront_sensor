#pragma once
/*****************************************************************//**
 * \file   Zernike.h
 * \brief  使用Zernike多项式进行目标特征的区域提取
 * 
 * \author Administrator
 * \date   December 2023
 *********************************************************************/

namespace Zernike {



	//! 整合Zernike多项式

	static void intergrate( cv::Mat& sx2d, cv::Mat& sy2d, cv::Mat& x2d, cv::Mat& y2d,
						    cv::Mat& jld, cv::Mat& z2d_wfr, cv::Mat& zx, cv::Mat& zy, cv::Mat& wfr_coefs,
							std::vector<cv::Mat>& zxm3d, std::vector<cv::Mat>& zym3d, 
							cv::Mat& xn2d, cv::Mat& yn2d, cv::Mat& xy_norm) {

		printf("未完善的接口 \n");
	}
};
