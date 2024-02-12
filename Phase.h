#pragma once
#include <opencv.hpp>


template<typename T>
static void QGPU2SC_(cv::Mat& wp, cv::Mat& quality_map, cv::Mat& start, cv::Mat& uwp, int num_of_quality_steps = 128) {



    cv::Size size = wp.size();
    if (!quality_map.rows && !quality_map.cols) {

        cv::Mat mat1 = cv::Mat(HIOlAB_CXX_14::createHanningWindow<int>(size.height));
        cv::Mat mat2 = cv::Mat(HIOlAB_CXX_14::createHanningWindow<int>(size.width));
        quality_map = mat1.mul(mat2);
    }

    double mat_min_value, mat_max_value;
    cv::Point min_postion, max_postion;

    cv::minMaxLoc(quality_map, &mat_min_value, &mat_max_value, &min_postion, &max_postion);
    double quality_thr = mat_min_value;
    cv::Mat_<bool> mask = cv::Mat::ones(size, CV_32S) / 1;

    int start_row;
    int start_col;


    //! 为什么在进行判空之后生成了 还进行判空呢？？？？
    if (start.empty()) {
        try {

            std::pair<std::vector<int>, std::vector<int>> index_vector_pair = HIOlAB_CXX_14::where<T>(quality_map, mat_max_value);
            start_row = index_vector_pair.first.at(0);
            start_col = index_vector_pair.second.at(0);
        }
        catch (const VectorNullException& matNullException) {

            HIOlAB_CXX_14::ExceptionInfoPrint(matNullException.what());
        }
    }

    else {
        start_row = start.at<int>(0, 0);
        start_col = start.at<int>(0, 1);

    }


    if (mat_min_value != mat_max_value) {

        cv::Mat OutputArray = ((quality_map - mat_min_value) / (mat_max_value - mat_min_value)) * (static_cast<double>(num_of_quality_steps) - 1);
        OutputArray.convertTo(OutputArray, CV_32S);
        OutputArray += 1;
        quality_map.release();
        quality_map = OutputArray;

        //! No tested branch
        if (quality_thr >= mat_min_value) {
            auto value = ((quality_thr - mat_min_value) / (mat_max_value - mat_min_value)) * (static_cast<double>(num_of_quality_steps) - 1);
            quality_thr = std::round(value) + 1;
        }
        else
        {
            quality_thr = 1;
        }
    }
    else {
        if (mat_min_value != 0) {
            quality_map /= mat_min_value;
        }
        quality_map.convertTo(quality_map, CV_32S);
        quality_thr = 1;
    }

    cv::Mat stack_chain = cv::Mat::zeros(1, num_of_quality_steps + 1, CV_32S);
    cv::Mat uwg_row = cv::Mat::zeros(1, wp.rows * wp.cols, CV_32S);
    cv::Mat uwg_col = cv::Mat::zeros(1, wp.rows * wp.cols, CV_32S);
    cv::Mat uwd_row = cv::Mat::zeros(1, wp.rows * wp.cols, CV_32S);
    cv::Mat uwd_col = cv::Mat::zeros(1, wp.rows * wp.cols, CV_32S);

    cv::Mat stack_n = cv::Mat::zeros(1, wp.rows * wp.cols, wp.type());
    uwp = cv::Mat::zeros(wp.rows, wp.cols, wp.type());
    cv::Mat path_map = uwp.clone();
    cv::Mat_<bool>  queued_flag = cv::Mat::zeros(wp.rows, wp.cols, CV_8U) / 1;


    int quality_max = quality_map.at<int>(start_row, start_col);

    stack_chain.at<int>(0, quality_max) = 1;
    int pointer = 1;
    int unwr_order = 0;

    uwd_row.at<int>(0, stack_chain.at<int>(0, quality_max)) = start_row;
    uwd_col.at<int>(0, stack_chain.at<int>(0, quality_max)) = start_col;
    uwg_row.at<int>(0, stack_chain.at<int>(0, quality_max)) = start_row;
    uwg_col.at<int>(0, stack_chain.at<int>(0, quality_max)) = start_col;


    path_map.at<double>(start_row, start_col) = 1;
    queued_flag.at<bool>(start_row, start_col) = 1;
    uwp.at<double>(start_row, start_col) = wp.at<double>(start_row, start_col);

    //! quality_max>=quality_thr flood filling
    while (quality_max >= quality_thr) {

        
       
        if (stack_chain.at<int>(0, quality_max) == 0) {

            --quality_max;
        }

        else {

           // std::cout << quality_max << std::endl;

            int    column = stack_chain.at<int>(0, quality_max);
            int    uwdrow = uwd_row.at<int>(0, column);
            int    uwdcol = uwd_col.at<int>(0, column);
            double a = uwp.at<double>(uwdrow, uwdcol);

            int uwgrow = uwg_row.at<int>(0, column);
            int uwgcol = uwg_col.at<int>(0, column);
            double b = wp.at<double>(uwgrow, uwgcol);

            uwp.at<double>(uwgrow, uwgcol) = b - 2 * M_PI * std::round(static_cast<double>(b - a) / (2 * M_PI));

            int temp_row = uwg_row.at<int>(0, column);
            int temp_col = uwg_col.at<int>(0, column);

            //! update path_map
            path_map.at<double>(temp_row, temp_col) = unwr_order++;
            stack_chain.at<int>(0, quality_max) = static_cast<int>(stack_n.at<double>(0, column));

            if (temp_row > 0) {
                //! check unwrapped status and mask validity
                if (!(queued_flag.at<bool>(temp_row - 1, temp_col)) && mask.at<bool>(temp_row - 1, temp_col)) {


                    //! upper (row-1,col)
                    uwg_row.at<int>(0, pointer) = temp_row - 1;
                    uwg_col.at<int>(0, pointer) = temp_col;
                    uwd_row.at<int>(0, pointer) = temp_row;
                    uwd_col.at<int>(0, pointer) = temp_col;

                    //! push stack_chain to the stack__n at pointer
                    int i = uwg_row.at<int>(0, pointer);
                    int j = uwg_col.at<int>(0, pointer);



                    stack_n.at <double>(0, pointer) = stack_chain.at<int>(0, quality_map.at<int>(i, j));

                    //! push pointer to stack_chain
                    stack_chain.at<int>(0, quality_map.at<int>(i, j)) = pointer;

                    //! if the quality value of pused point is bigger than the current quality_max value,
                    //! set the quality_max as the quality value of pused point 
                    quality_map.at<int>(i, j) > quality_max ? quality_max = quality_map.at<int>(i, j) : void(0);

                    //! queue the point
                    queued_flag.at<bool>(i, j) = 1;
                    ++pointer;

                    

                }
               // std::cout << "row >0 succ\n";

            }

            //! the nether neighboring point (row+1,col) check  dimensional validity
            if (temp_row < size.height - 1) {
        
                //! check unwrapped status and mask validity
                if (!(queued_flag.at<bool>(temp_row + 1, temp_col)) && mask.at<bool>(temp_row + 1, temp_col)) {


             
                    //! upper (row-1,col)
                    uwg_row.at<int>(0, pointer) = temp_row + 1;
                    uwg_col.at<int>(0, pointer) = temp_col;
                    uwd_row.at<int>(0, pointer) = temp_row;
                    uwd_col.at<int>(0, pointer) = temp_col;

    
                    //! push stack_chain to the stack__n at pointer
                    int i = uwg_row.at<int>(0, pointer);
                    int j = uwg_col.at<int>(0, pointer);
                    stack_n.at <double>(0, pointer) = stack_chain.at<int>(0, quality_map.at<int>(i, j));

                   
                    //! push pointer to stack_chain
                    stack_chain.at<int>(0, quality_map.at<int>(i, j)) = pointer;
 
                    //! if the quality value of pused point is bigger than the current quality_max value,
                    //! set the quality_max as the quality value of pused point 
                    quality_map.at<int>(i, j) > quality_max ? quality_max = quality_map.at<int>(i, j) : void(0);

                    //! queue the point
                    queued_flag.at<bool>(i, j) = 1;
                    ++pointer;

                    
                } 
                //std::cout << "row < h-1 succ\n";
            }


            //! the nether neighboring point (row,col-1) check  dimensional validity
            if (temp_col > 0) {
                //! check unwrapped status and mask validity
                if (!(queued_flag.at<bool>(temp_row, temp_col - 1)) && mask.at<bool>(temp_row, temp_col - 1)) {


                    //! upper (row-1,col)
                    uwg_row.at<int>(0, pointer) = temp_row;
                    uwg_col.at<int>(0, pointer) = temp_col - 1;
                    uwd_row.at<int>(0, pointer) = temp_row;
                    uwd_col.at<int>(0, pointer) = temp_col;

                    //! push stack_chain to the stack__n at pointer
                    int i = uwg_row.at<int>(0, pointer);
                    int j = uwg_col.at<int>(0, pointer);



                    stack_n.at <double>(0, pointer) = stack_chain.at<int>(0, quality_map.at<int>(i, j));

                    //! push pointer to stack_chain
                    stack_chain.at<int>(0, quality_map.at<int>(i, j)) = pointer;

                    //! if the quality value of pused point is bigger than the current quality_max value,
                    //! set the quality_max as the quality value of pused point 
                    quality_map.at<int>(i, j) > quality_max ? quality_max = quality_map.at<int>(i, j) : void(0);

                    //! queue the point
                    queued_flag.at<bool>(i, j) = 1;
                    ++pointer;

                }
                //std::cout << "row >0 succ\n";
            }



            //! right neighboring point (row col+1) ,check dimensional validity
            if (temp_col < size.width - 1) {
                //! check unwrapped status and mask validity
                if (!(queued_flag.at<bool>(temp_row, temp_col + 1)) && mask.at<bool>(temp_row, temp_col + 1)) {


                    //! upper (row-1,col)
                    uwg_row.at<int>(0, pointer) = temp_row;
                    uwg_col.at<int>(0, pointer) = temp_col + 1;
                    uwd_row.at<int>(0, pointer) = temp_row;
                    uwd_col.at<int>(0, pointer) = temp_col;

                    //! push stack_chain to the stack__n at pointer
                    int i = uwg_row.at<int>(0, pointer);
                    int j = uwg_col.at<int>(0, pointer);



                    stack_n.at <double>(0, pointer) = stack_chain.at<int>(0, quality_map.at<int>(i, j));

                    //! push pointer to stack_chain
                    stack_chain.at<int>(0, quality_map.at<int>(i, j)) = pointer;

                    //! if the quality value of pused point is bigger than the current quality_max value,
                    //! set the quality_max as the quality value of pused point 
                    quality_map.at<int>(i, j) > quality_max ? quality_max = quality_map.at<int>(i, j) : void(0);

                    //! queue the point
                    queued_flag.at<bool>(i, j) = 1;
                    ++pointer;

                }

                //std::cout << "col <w-1 succ\n";
            }


        }
        
    }
}