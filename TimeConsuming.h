#pragma once
/*****************************************************************//**
 * \file    TimeConsuming.h
 * \brief   �������㵥�̻߳��߶��߳�ִ�к����ĸ��Ӷ�
 * \author  Administrator
 * \date    November 2023
 *********************************************************************/
#include <chrono>


class TimeConsuming{
private:
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point  end;
public:
	TimeConsuming() {

	}
	~TimeConsuming() = default;
	TimeConsuming(const TimeConsuming& object) = default;
	TimeConsuming(TimeConsuming&& object) = default;
public:

	void setStartTime(std::chrono::steady_clock::time_point& start_) {

		start = start;
	}

	void setEndTime(std::chrono::steady_clock::time_point& end_) {

		end = end_;
	}

	std::chrono::duration<double> calculateDuration() {

		return std::chrono::duration_cast<std::chrono::duration<double>>(end-start);

	}
};
