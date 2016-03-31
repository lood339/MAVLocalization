//
//  cvx_io.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-20.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvx_io.hpp"
#include <iostream>

using cv::Mat;
using std::cout;
using std::endl;


bool cvx_io::imread_depth_16bit_to_32f(const char *file, cv::Mat & depth_img)
{
    depth_img = cv::imread(file, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if (depth_img.empty()) {
        printf("Error: can not read image from %s\n", file);
        return false;
    }
    assert(depth_img.type() == CV_16UC1);
    depth_img.convertTo(depth_img, CV_32F);
    return true;
}

bool cvx_io::imread_depth_16bit_to_64f(const char *filename, cv::Mat & depth_img)
{
    depth_img = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if (depth_img.empty()) {
        printf("Error: can not read image from %s\n", filename);
        return false;
    }
    assert(depth_img.type() == CV_16UC1);
    depth_img.convertTo(depth_img, CV_64F);
    return true;
}

bool cvx_io::imread_rgb_8u(const char *file_name, cv::Mat & rgb_img)
{
    rgb_img = cv::imread(file_name, CV_LOAD_IMAGE_COLOR);
    if (rgb_img.empty()) {
        printf("Error: can not read image from %s\n", file_name);
        return false;
    }
    assert(rgb_img.type() == CV_8UC3);
    return true;
}

void cvx_io::imwrite_depth_8u(const char *file, const cv::Mat & depth_img)
{
    assert(depth_img.type() == CV_32F || depth_img.type() == CV_64F);
    assert(depth_img.channels() == 1);
    
    double minv = 0.0;
    double maxv = 0.0;
    cv::minMaxLoc(depth_img, &minv, &maxv);
    
    printf("min, max values are: %lf %lf\n", minv, maxv);
    
    
    cv::Mat shifted_depth_map;
    depth_img.convertTo(shifted_depth_map, CV_32F, 1.0, -minv);
    cv::Mat depth_8u;
    shifted_depth_map.convertTo(depth_8u, CV_8UC1, 255/(maxv - minv));
    
    cv::imwrite(file, depth_8u);
    printf("save to: %s\n", file);
}



/********     ms_7_scenes_util      ************/
Mat ms_7_scenes_util::read_pose_7_scenes(const char *file_name)
{
    Mat P = Mat::zeros(4, 4, CV_64F);
    FILE *pf = fopen(file_name, "r");
    assert(pf);
    for (int row = 0; row<4; row++) {
        for (int col = 0; col<4; col++) {
            double v = 0;
            fscanf(pf, "%lf", &v);
            P.at<double>(row, col) = v;
        }
    }
    fclose(pf);
//    cout<<"pose is "<<P<<endl;
    return P;
}

// return CV_64F
Mat ms_7_scenes_util::camera_depth_to_world_depth(const cv::Mat & camera_depth_img, const cv::Mat & pose)
{
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    cv::Mat world_depth_img = cv::Mat::zeros(height, width, CV_64F);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c);
            if ((int)camera_depth == 65535) {
                // invalid depth
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/z;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            Mat x_world = pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_depth_img.at<double>(r, c) = x_world.at<double>(2, 0); // save depth in world coordinate
        }
    }
    world_depth_img /= 1000.0;
    return world_depth_img;
}

cv::Mat ms_7_scenes_util::camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img, const cv::Mat & camera_to_world_pose)
{
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;
    
    Mat inv_K = K.inv();
    
    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c);
            if ((int)camera_depth == 65535) {
                // invalid depth
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/z;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;
            
            Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
          //  world_depth_img.at<double>(r, c) = x_world.at<double>(2, 0); // save depth in world coordinate
            world_coordinate_img.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }
    world_coordinate_img /= 1000.0;
    return world_coordinate_img;
}

