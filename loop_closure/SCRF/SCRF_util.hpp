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
    cv::Point2i p2d_;    // 2d location
    cv::Point3d p3d_;   // 3d coordinate, only used for training, invalid when in testing
    double depth_;      // depth in camera coordinate
    double inv_depth_;  // inverted depth
    int image_index_;  // image index
    
public:
    cv::Point2i add_offset(const cv::Point2d & offset) const
    {
        int x = cvRound(p2d_.x + offset.x * inv_depth_);
        int y = cvRound(p2d_.y + offset.y * inv_depth_);
        
        return cv::Point2i(x, y);
    }
};

class SCRF_testing_result
{
public:
    cv::Point2d p2d_;           // image position
    cv::Point3d gt_p3d_;        // as ground truth, not used in prediction
    cv::Point3d predict_p3d_;   // predicted world coordinate
    
    double error_distance() const
    {
        double dis = 0.0;
        cv::Point3d dif = predict_p3d_ - gt_p3d_;
        dis += dif.x * dif.x;
        dis += dif.y * dif.y;
        dis += dif.z * dif.z;
        return sqrt(dis);
    }
    
};


struct SCRF_tree_parameter
{
    int tree_num_;           // number of trees
    int max_depth_;
    int min_leaf_node_;
    
    int max_pixel_offset_;  // in pixel
    int pixel_offset_candidate_num_;  // large number less randomness
    int split_candidate_num_;  // number of split in [v_min, v_max]
    int weight_candidate_num_;
    
  //  bool verbose_;
    bool verbose_leaf_;
    bool verbose_split_;
    
    SCRF_tree_parameter()
    {
        tree_num_ = 5;
        max_depth_ = 15;
        min_leaf_node_ = 50;
        
        max_pixel_offset_ = 131;
        pixel_offset_candidate_num_ = 20;
        split_candidate_num_ = 20;
        weight_candidate_num_ = 10;
        verbose_leaf_ = true;
        verbose_split_ = true;
    }
    
    void printSelf() const
    {
        printf("SCRF tree parameters:\n");
        printf("tree_num: %d\t max_depth: %d\t min_leaf_node: %d\n", tree_num_, max_depth_, min_leaf_node_);
        printf("max_pixel_offset: %d\t pixel_offset_candidate_num: %d\t split_candidate_num %d\n", max_pixel_offset_, pixel_offset_candidate_num_, split_candidate_num_);
        printf("weight_candidate_num_: %d\n\n", weight_candidate_num_);
    }
};

class SCRF_Util
{
public:
    // centroid of all indixed locations
    static cv::Point3d mean_location(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices);
    
    static void mean_stddev(const vector<SCRF_learning_sample> & sample,
                            const vector<unsigned int> & indices,
                            cv::Point3d & mean_pt,
                            cv::Vec3d & stddev);
    
    // box filter
    // criteria.epsilon: squred distance in meter
    // return: number of points in the box
    static int mean_shift(const vector<cv::Point3d> & pts,
                           cv::Point3d & mean_pt,
                           cv::Vec3d & stddev,
                           const cv::TermCriteria & criteria);
    
    static void mean_shift(const vector<SCRF_learning_sample> & sample,
                           const vector<unsigned int> & indices,
                           cv::Point3d & mean_pt,
                           cv::Vec3d & stddev);

    
    // spatial variance of selected samples
    static double spatial_variance(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices);
    
    
    static void mean_std_position(const vector<cv::Point3d> & points, cv::Point3d & mean_pos, cv::Vec3d & std_pos);
    
    static vector<double> prediction_error_distance(const vector<SCRF_testing_result> & results);
    
    // approximate median
    static cv::Point3d appro_median_error(const vector<SCRF_testing_result> & results);
    
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
