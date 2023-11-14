#pragma once
#include	<qmap.h>



class Configuration {
private:
	QMap<QString, double> conf;
public:
	/** 静态配置 */
	Configuration() {
		conf.insert("nu_detector", 512);
		conf.insert("nv_detector", 512);
		conf.insert("upsampling", 2);
		conf.insert("grid_period", 20.e-06);
		conf.insert("dist_mask_to_detector", 0.2);
		conf.insert("lowest_fringe_order", 8);
		conf.insert("centroid_power", 1.7);
		conf.insert("detector_pixel_size", 1.48e-06);
		conf.insert("wavelength", 1.0972e-10);
	}

	/** 保留一个动态配置接口 */
	/**
	 * nu_detector nu检测器
	 * nv_detector nv检测器
	 * unsmapling 上采样
	 * grid period 网格周期
	 * dist_mask_to_detector 区域掩码检测器
	 * lowest_fringe_order 最低边缘？？
	 * centroid_power 质心
	 * detector_pixel_sie 检测像素点的区域大小
	 * wave length 波长
	 */
	Configuration(int& nu_detector_size, int& nv_detector_size,
		double& upsampling, double& grid_period,
		double& dist_mask_to_detector, int& lowest_fringe_order,
		double& centroid_power, double& detector_pixel_size,
		double& wavelength) {

		conf.insert("nu_detector", static_cast<double>(nu_detector_size));
		conf.insert("nv_detector", static_cast<double>(nv_detector_size));
		conf.insert("upsampling", upsampling);
		conf.insert("grid_period", grid_period);
		conf.insert("dist_mask_to_detector", dist_mask_to_detector);
		conf.insert("lowest_fringe_order", static_cast<double>(lowest_fringe_order));
		conf.insert("centroid_power", centroid_power);
		conf.insert("detector_pixel_size", detector_pixel_size);
		conf.insert("wavelength", wavelength);
	}

	auto operator[](const QString& key) -> decltype(conf[key]) {
		if (conf.find(key) == conf.end()) {

			conf.insert(key, 0.);

		}
		return conf[key];
	}


	~Configuration() {

	}
};
