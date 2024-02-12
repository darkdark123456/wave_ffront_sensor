#pragma once

#include	<QWidget>
#include	<qsharedpointer.h>
#include	<qvector.h>
#include	<qlabel.h>
#include	<qgridlayout.h>
#include	<qgroupbox.h>
#include	<qspinbox.h>
#include	<qcombobox.h>
#include	<qtoolbutton.h>
#include	<qmessagebox.h>
#include	"Configuration.h"
#include	"ui_ConfigurationWidget.h"

QT_BEGIN_NAMESPACE
namespace Ui { class ConfigurationWidgetClass; };
QT_END_NAMESPACE

class ConfigurationWidget : public QWidget{

	Q_OBJECT
public:
	ConfigurationWidget(QWidget *parent = nullptr);
	~ConfigurationWidget();
public:
	void initConfigurationWidget();
	void free();
public:
	void signalFuncOperation();
public:
	std::tuple<Configuration,short,QString,bool> getConfiguraion();
private:
	Ui::ConfigurationWidgetClass  *ui;
	QVector<QLabel*>*             config_label_vec;
	QVector<QDoubleSpinBox*>*     double_spin_box_vec;
	QVector<QSpinBox*>*           int_spin_box_vec;
	QGridLayout*				   grid_lay_out;
	QComboBox*					   model_combox;
	QComboBox*					   method_combox;
	QComboBox*					   is_show_combox;
	QToolButton*				   ok_button;
	QToolButton*				   reset_button;
	Configuration				   conf;
	short 						   method_index;
	QString						   model;
	bool						   is_show;
};
