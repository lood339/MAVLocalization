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

class SCRF_testing_result
{
public:
    cv::Point3d predict_p3d_;   // predicted world coordinate
    cv::Point3d predict_error;  // prediction - ground truth
    
    cv::Vec3d  std_;     // prediction standard deviation
};


struct SCRF_tree_parameter
{
    int tree_num_;           // number of trees
    int max_depth_;
    int min_leaf_node_;
    
    int max_pixel_offset_;  // in pixel
    int pixel_offset_candidate_num_;  // large number less randomness
    int split_candidate_num_;  // number of split in [v_min, v_max]
    
    SCRF_tree_parameter()
    {
        tree_num_ = 5;
        max_depth_ = 15;
        min_leaf_node_ = 50;
        
        max_pixel_offset_ = 131;
        pixel_offset_candidate_num_ = 100;
        split_candidate_num_ = 20;
    }
    
    void printSelf() const
    {
        printf("SCRF tree parameters:\n");
        printf("tree_num: %d\t max_depth: %d\t min_leaf_node: %d\n", tree_num_, max_depth_, min_leaf_node_);
        printf("max_pixel_offset: %d\t pixel_offset_candidate_num: %d\t split_candidate_num %d\n\n", max_pixel_offset_, pixel_offset_candidate_num_, split_candidate_num_);
    }
};

class SCRF_Util
{
public:
    // centroid of all indixed locations
    static cv::Point3d mean_location(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices);
    
    // spatial variance of selected samples
    static double spatial_variance(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices);
    
    static void mean_std_position(const vector<cv::Point3d> & points, cv::Point3d & mean_pos, cv::Vec3d & std_pos);
    
    static cv::Point3d prediction_error_stddev(const vector<SCRF_testing_result> & results);
    
    static inline bool is_inside_image(const int width, const int height, const int x, const int y)
    {
        return x >= 0 && y >= 0 && x < width && y < height;
    }
    
    // randomly sample learning samples from ground truth camera_pose_file
    static vector<SCRF_learning_sample> randomSampleFromRgbdImages(const char * rgb_img_file,
                                                                   const char * depth_img_file,
                                                                   const char * camera_pose_file,
                                                                   const int num_sample,
                                                                   const int image_index);
    static bool readTreeParameter(const char *file_name, SCRF_tree_parameter & tree_param);

};


#endif /* SCRF_util_cpp */
