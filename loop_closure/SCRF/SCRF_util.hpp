//
//  SCRF_util.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-19.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef SCRF_util_cpp
#define SCRF_util_cpp

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include <vector>

using std::vector;


class SCRF_learning_sample
{
public:
    cv::Vec2i p_;   // 2d location
    cv::Point3d p3d_; // 3d coordinate, only used for training, invalid when in testing
    double depth_;    // depth in camera coordinate
    double inv_depth_; // inverted depth
    int image_index_; // image index
    
public:
    cv::Vec2i get_displacement(const double dx, const double dy) const
    {
        int x = cvRound(p_[0] + dx * inv_depth_);
        int y = cvRound(p_[1] + dy * inv_depth_);
        return cv::Vec2i(x, y);
    }
    
    cv::Vec2i get_displacement(const cv::Vec2d & offset) const
    {
        int x = cvRound(p_[0] + offset[0] * inv_depth_);
        int y = cvRound(p_[1] + offset[1] * inv_depth_);
     //   printf("offset is %lf, %lf\n", offset[0] * inv_depth_, offset[1] * inv_depth_);
        return cv::Vec2i(x, y);
    } 
    
};

class SCRF_testing_sample
{
public:
    cv::Vec2i p_;   // 2d location
    double depth_;    // depth in camera coordinate
    double inv_depth_; // inverted depth
    cv::Point3d predict_p3d_; // predicted world coordinate
    
    SCRF_testing_sample(const SCRF_learning_sample & learning_sampe)
    {
        p_ = learning_sampe.p_;
        depth_ = learning_sampe.depth_;
        inv_depth_ = learning_sampe.inv_depth_;
    }
    
    cv::Vec2i get_displacement(const double dx, const double dy) const
    {
        int x = cvRound(p_[0] + dx * inv_depth_);
        int y = cvRound(p_[1] + dy * inv_depth_);
        return cv::Vec2i(x, y);
    }
    
    cv::Vec2i get_displacement(const cv::Vec2d & offset) const
    {
        int x = cvRound(p_[0] + offset[0] * inv_depth_);
        int y = cvRound(p_[1] + offset[1] * inv_depth_);
        return cv::Vec2i(x, y);
    }

};

struct SCRF_tree_parameter
{
    int max_depth_;
    int min_leaf_node_;
    int image_width_;
    int image_height_;
    
    SCRF_tree_parameter()
    {
        max_depth_ = 15;
        min_leaf_node_ = 50;
    }
};

class SCRF_Util
{
public:
    // centroid of all indixed locations
    static cv::Point3d mean_location(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices);
    
    // spatial variance of selected samples
    static double spatial_variance(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices);
    
    static inline bool is_inside_image(const int width, const int height, const int x, const int y)
    {
        return x >= 0 && y >= 0 && x < width && y < height;
    }
};


#endif /* SCRF_util_cpp */
