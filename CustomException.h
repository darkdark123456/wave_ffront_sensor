#pragma once
#include	<exception>

struct KeyNotFindException : public std::exception {

	KeyNotFindException() : std::exception("Key Not Find Exception ", 1) {


	}

	KeyNotFindException(const char* const _Message) : std::exception(_Message, 1) {


	}

};


struct  RunTimeException : public std::exception {

	RunTimeException() : std::exception("Programme frame  run time exception ", 1) {

	}

	RunTimeException(const char* const _Message) : std::exception(_Message, 1) {


	}

};


struct FileOpenException : public std::exception {

	FileOpenException() : std::exception("File open exception ", 1) {

	}

	FileOpenException(const char* const _Message) : std::exception(_Message, 1) {


	}
};


struct OutOfRangeException : public std::exception {

	OutOfRangeException() : std::exception("Index  out of range exception ", 1) {


	}

	OutOfRangeException(const char* const _Message) : std::exception(_Message, 1) {


	}


};


struct DimConvertException : public std::exception {

	DimConvertException() : std::exception("Dim convert exception ", 1) {


	}


	DimConvertException(const char* const _Message) : std::exception(_Message, 1) {

	}
};

struct VectorNullException : public std::exception {
	VectorNullException() :std::exception("Vector NUll Exception", 1) {


	}

	VectorNullException(const char* const _Message) :std::exception(_Message,1) {



	}

};

struct DenominatorZeroException : std::exception {

	DenominatorZeroException() : std::exception("Denominator Zero Exception ",1) {

	}

	DenominatorZeroException(const char* const _Message) :std::exception(_Message, 1) {

	}
};



struct MatrixMultDimException : std::exception {

	MatrixMultDimException() : std::exception(" Matrix Mult Exception ", 1) {


	}

	MatrixMultDimException(const char* const _Message) : std::exception(_Message, 1) {


	}


};

struct TypeException : std::exception {

	TypeException() : std::exception("Type Exception", 1) {

	}

	TypeException(const char* const Message_) : std::exception(Message_, 1) {

	}

};


struct FftException : public std::exception {

	FftException() : std::exception(" FFT Exception ", 1) {

	}

	FftException(const char* const _Message) : std::exception(_Message, 1) {


	}

};



struct  ShapeException : std::exception
{
public:
	ShapeException() : std::exception("Shape exception throw ",1) {

	}

	ShapeException(const char* const message) : std::exception(message, 1) {


	}

};


struct  SparseQRException  : std::exception
{
public:
	SparseQRException() : std::exception("Sparse QR exception throw",1) {

	}
	SparseQRException(const char* const message) : std::exception(message, 1) {

	}
};


struct PlotException : std::exception
{
public:
	PlotException() : std::exception(u8" <--!!异常 在非第一次计算时关闭已经显示的结果图像,否则会抛出plt::show()异常!---> ", 1) {

	}

	PlotException(const char* const message) : std::exception(message, 1) {

	}
};