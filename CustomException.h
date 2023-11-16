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

