#include "ConfigurationWidget.h"
#include	<iostream>

ConfigurationWidget::ConfigurationWidget(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::ConfigurationWidgetClass())
{
	ui->setupUi(this);
	initConfigurationWidget();
}

ConfigurationWidget::~ConfigurationWidget()
{
	
	free();
	delete ui;
}
	

void ConfigurationWidget::initConfigurationWidget(){


	setWindowTitle(u8"配置");
	setFixedSize(400, 550);
	
	QPalette palette;

	palette.setBrush(QPalette::Background, QBrush(QPixmap(":/WaveFrontSensor/back/b3.png")));
	setPalette(palette);

	QIcon icon;
	QPixmap pixmap_(":/WaveFrontSensor/back/icon2.png");
	icon.addPixmap(pixmap_);
	setWindowIcon(icon);
	grid_lay_out = new QGridLayout(this);
	config_label_vec = new QVector<QLabel*>();
	config_label_vec->resize(9);
	double_spin_box_vec = new QVector<QDoubleSpinBox*>(5);
	int_spin_box_vec = new QVector<QSpinBox*>(4);

	config_label_vec->push_back(new QLabel("nu_detector"));
	config_label_vec->push_back(new QLabel("nv_detector"));
	config_label_vec->push_back(new QLabel("upsampling"));
	config_label_vec->push_back(new QLabel("lowest_fringe_order"));

	config_label_vec->push_back(new QLabel("grid_period"));
	config_label_vec->push_back(new QLabel("dist_mask_to_detector"));
	config_label_vec->push_back(new QLabel("centroid_power"));
	config_label_vec->push_back(new QLabel("detector_pixel_size"));
	config_label_vec->push_back(new QLabel("wave_legenth"));
	config_label_vec->push_back(new QLabel("method "));
	config_label_vec->push_back(new QLabel("model"));
	config_label_vec->push_back(new QLabel("is_show"));

	
	int_spin_box_vec->push_back(new QSpinBox());
	int_spin_box_vec->back()->setValue(0);
	
	int_spin_box_vec->push_back(new QSpinBox());
	int_spin_box_vec->back()->setValue(0);

	int_spin_box_vec->push_back(new QSpinBox());
	int_spin_box_vec->back()->setValue(2);

	int_spin_box_vec->push_back(new QSpinBox());
	int_spin_box_vec->back()->setValue(8);




	double_spin_box_vec->push_back(new QDoubleSpinBox());
	double_spin_box_vec->back()->setDecimals(15);
	double_spin_box_vec->back()->setValue(20.e-06);

	double_spin_box_vec->push_back(new QDoubleSpinBox());
	double_spin_box_vec->back()->setDecimals(15);
	double_spin_box_vec->back()->setValue(0.2);

	double_spin_box_vec->push_back(new QDoubleSpinBox());
	double_spin_box_vec->back()->setDecimals(5);
	double_spin_box_vec->back()->setValue(1.7);

	double_spin_box_vec->push_back(new QDoubleSpinBox());
	double_spin_box_vec->back()->setDecimals(15);
	double_spin_box_vec->back()->setValue(1.48e-06);

	double_spin_box_vec->push_back(new QDoubleSpinBox());
	double_spin_box_vec->back()->setDecimals(15);
	double_spin_box_vec->back()->setValue(1.0972e-10);

	model_combox = new QComboBox();
	method_combox = new QComboBox();
    is_show_combox = new QComboBox();
	
	model_combox->addItems(QStringList() << "Legendre" << "Zernike");
	method_combox->addItem(QString("zonal"));

	is_show_combox->addItems(QStringList() << "True" << "False");


	ok_button = new QToolButton();
	reset_button = new QToolButton();
	ok_button->setText(u8"确认");
	reset_button->setText(u8"重置");
	ok_button->setStyleSheet("QToolButton {"
		"font: 700 10pt Courier New;"
		"color:rgb(0, 0, 255) ;"
		"border-radius: 8px;"
		"border: 2px solid #ccf;"
		"border-color: rgb(0, 0, 255);"
		"}");
	
	reset_button->setStyleSheet("QToolButton {"
		"font: 700 10pt Courier New;"
		"color:rgb(0, 0, 255) ;"
		"border-radius: 8px;"
		"border: 2px solid #ccf;"
		"border-color: rgb(0, 0, 255);"
		"}");


	int index = 0;
	for (QVector<QSpinBox*>::iterator iter = int_spin_box_vec->begin(); iter != int_spin_box_vec->end(); iter++) {

		if (*iter) {
			(*iter)->setStyleSheet("color:rgb(85, 85, 127); border-radius: 8px; border: 2px solid #ccf;");
			(*iter)->setMinimum(0);
			(*iter)->setMaximum(INT_MAX);
		}

	}


	for (QVector<QDoubleSpinBox*>::iterator iter = double_spin_box_vec->begin(); iter != double_spin_box_vec->end(); iter++) {

		if (*iter) {
			(*iter)->setStyleSheet("QDoubleSpinBox {"
					"color:rgb(170, 85, 127) ;"
					"border-radius: 8px;"
					"border: 2px solid #ccf;"
					"}");
			(*iter)->setMinimum(0);
			(*iter)->setMaximum(DBL_MAX);
		}
	}


	for (QVector<QLabel*>::iterator iter = config_label_vec->begin(); iter != config_label_vec->end(); iter++) {

		if (*iter) {
			(*iter)->setStyleSheet("font: 700 11pt Courier New;color: rgb(0, 170, 127); ");
			grid_lay_out->addWidget(*iter, index++, 0);
		}
	}
	
	grid_lay_out->addWidget(ok_button, index++, 0);


	index = 0;
	for (QVector<QSpinBox*>::iterator iter = int_spin_box_vec->begin(); iter != int_spin_box_vec->end(); iter++) {

		if (*iter) { 
			grid_lay_out->addWidget(*iter,index++,1);
		}
	}

	for (QVector<QDoubleSpinBox*>::iterator iter = double_spin_box_vec->begin(); iter != double_spin_box_vec->end(); iter++) {

		if (*iter) { 
			grid_lay_out->addWidget(*iter,index++,1);
		}
	}
	

	model_combox->setStyleSheet("font: 700 10pt Courier New; color: rgb(85, 170, 127); border-radius: 8px;");
	method_combox->setStyleSheet("font: 700 10pt Courier New; color: rgb(255, 85, 0); border-radius: 8px;");
	is_show_combox->setStyleSheet("font: 700 10pt Courier New; color: rgb(170, 0, 0); border-radius: 8px;");
	grid_lay_out->addWidget(model_combox, index++, 1);
	grid_lay_out->addWidget(method_combox, index++, 1);
	grid_lay_out->addWidget(is_show_combox, index++, 1);
	grid_lay_out->addWidget(reset_button, index++,1);
	
	signalFuncOperation();


}


void ConfigurationWidget::signalFuncOperation() {

	connect(reset_button, &QToolButton::clicked, this, [&] () -> void {
		Configuration configuration;
		QVector<double> parms_vec(9);
		parms_vec[0]=(configuration["grid_period"]);
		parms_vec[1]=(configuration["dist_mask_to_detector"]);
		parms_vec[2]=(configuration["centroid_power"]);
		parms_vec[3]=(configuration["detector_pixel_size"]);
		parms_vec[4]=(configuration["wavelength"]);

		parms_vec[5] = (configuration["nu_detector"]);
		parms_vec[6] = (configuration["nv_detector"]);
		parms_vec[7] = (configuration["upsampling"]);
		parms_vec[8] = (configuration["lowest_fringe_order"]);

	
		char i = 0;
		for (QVector<QDoubleSpinBox*>::iterator iter = double_spin_box_vec->begin(); iter != double_spin_box_vec->end(); iter++) {

			if (*iter) { (*iter)->setValue(parms_vec[i++]); }
		}

		for (QVector<QSpinBox*>::iterator iter = int_spin_box_vec->begin(); iter != int_spin_box_vec->end(); iter++) {

			if (*iter) { (*iter)->setValue(static_cast<int>(parms_vec[i++])); }
		}


		QMessageBox::information(nullptr, " ", u8"已重置为默认参数 ！");

		});


	connect(ok_button, &QToolButton::clicked, this, [&]() -> void {
		QVector<double> parms_vec(9);
		char i = 0;
		for (QVector<QDoubleSpinBox*>::iterator iter = double_spin_box_vec->begin(); iter != double_spin_box_vec->end(); iter++) {

			if (*iter) { parms_vec[i++] = (*iter)->value(); }
		}
		
		for (QVector<QSpinBox*>::iterator iter = int_spin_box_vec->begin(); iter != int_spin_box_vec->end(); iter++) {

			if (*iter) { parms_vec[i++] = static_cast<double>((*iter)->value()); }
		}
		
		conf["grid_period"]=parms_vec[0];
		conf["dist_mask_to_detector"] = parms_vec[1];
		conf["centroid_power"] = parms_vec[2];
		conf["detector_pixel_size"] = parms_vec[3];
		conf["wavelength"] = parms_vec[4];

		conf["nu_detector"] = parms_vec[5];
		conf["nv_detector"] = parms_vec[6];
		conf["upsampling"] = parms_vec[7];
		conf["lowest_fringe_order"] = parms_vec[8];
		
		QMessageBox::information(nullptr, u8"信息", u8"参数设置成功！ 返回主界面点击开始选择数据！");
		
		});

	connect(is_show_combox, QOverload<int>::of(&QComboBox::currentIndexChanged), [=](int index) -> void  {
		
		is_show = static_cast<bool>(index);
		
		});


	connect(method_combox, QOverload<int>::of(&QComboBox::currentIndexChanged), [=](int index) -> void {
			
		method_index = static_cast<short>(index);
		
		});


	connect(model_combox, QOverload<int>::of(&QComboBox::currentIndexChanged), [=](int index) ->void {
		
		model = model_combox->currentText();

		});


	


}


std::tuple<Configuration,short,QString,bool> ConfigurationWidget::getConfiguraion() {

	return std::make_tuple(conf, method_index, model,is_show);
}



void ConfigurationWidget::free(){

	for (QVector<QLabel*>::iterator iter = config_label_vec->begin(); iter != config_label_vec->end(); iter++) {

		if (*iter) { delete *iter; }

	}

	for (QVector<QDoubleSpinBox*>::iterator iter = double_spin_box_vec->begin(); iter != double_spin_box_vec->end(); iter++) {

		if (*iter) { delete *iter; }
	}

	for (QVector<QSpinBox*>::iterator iter = int_spin_box_vec->begin(); iter != int_spin_box_vec->end(); iter++) {

		if (*iter) { delete* iter; }
	}



	if (model_combox != nullptr) {

		delete model_combox;
	}

	if (is_show_combox != nullptr) {

		delete is_show_combox;
	}


	if (ok_button != nullptr) {

		delete ok_button;
	}

	if (reset_button != nullptr) {

		delete reset_button;
	}

	if (grid_lay_out) {

		delete grid_lay_out;
	}


}
