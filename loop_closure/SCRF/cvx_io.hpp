//
//  cvx_io.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-20.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvx_io_cpp
#define cvx_io_cpp

// open CV input and output
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"

class cvx_io
{
public:
    // unit: millimeter
    static bool imread_depth_16bit_to_32f(const char *file, cv::Mat & depth_img);
    // unit: millimeter
    static bool imread_depth_16bit_to_64f(const char *filename, cv::Mat & depth_img);
    static bool imread_rgb_8u(const char *file_name, cv::Mat & rgb_img);
    
    // write depth image as 8u for visualization purpose
    static void imwrite_depth_8u(const char *file, const cv::Mat & depth_img);
};

// all Mat type is 64F
class ms_7_scenes_util
{
public:
    // read camera pose file
    static cv::Mat read_pose_7_scenes(const char *file_name);
    
    // invalid depth is 0.0
    static cv::Mat camera_depth_to_world_depth(const cv::Mat & camera_depth_img, const cv::Mat & pose);
    
    // camera_depth_img 16 bit
    // return CV_64_FC3 for x, y, z, unit in meter
    static cv::Mat camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img, const cv::Mat & camera_to_world_pose);   
    
    
    static inline int invalid_camera_depth(){return 65535;}
};




#endif /* cvx_io_cpp */
