//
//  SCRF_tree.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-19.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef SCRF_tree_cpp
#define SCRF_tree_cpp

// Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include <vector>
#include "SCRF_util.hpp"

using std::vector;

class SCRF_tree_node;

class SCRF_tree
{
    SCRF_tree_node *root_;
    cv::RNG rng_;   // random number generator
    
public:
    SCRF_tree(){root_ = NULL;}
    ~SCRF_tree(){;} //@todo release memory
    
    // training random forest by build a decision tree
    // samples: sampled image pixel locations
    // rgbImages: same size, rgb, 8bit image
    // depthImages: same size, 16bit image
    bool build(const vector<SCRF_learning_sample> & samples,
               const vector<cv::Mat> & rgbImages,               
               const SCRF_tree_parameter & param);
    
    // output: depth in world coordinate (in testing_sample)
    bool predict(SCRF_testing_sample & testing_sample,
                 const cv::Mat & rgbImage) const;
    
private:
    bool configure_node(const vector<SCRF_learning_sample> & samples,
                        const vector<cv::Mat> & rgbImages,                        
                        const vector<unsigned int> & indices,
                        int depth,
                        SCRF_tree_node *node,
                        const SCRF_tree_parameter & param);
    
    bool predict(const SCRF_tree_node * const node,
                 SCRF_testing_sample & sample,
                 const cv::Mat & rgbImage) const;
    
    
    
    
};

struct SCRF_split_parameter
{
    cv::Vec2d d1_;  // displacement in image coordinate
    cv::Vec2d d2_;
    int c1_;        // rgb image channel
    int c2_;
    double threhold_;  // threshold of splitting
    
public:
    SCRF_split_parameter()
    {
        c1_ = 0;
        c2_ = 0;
        threhold_ = 0.0;
    }
};

class SCRF_tree_node
{
public:
    SCRF_tree_node *left_node_;
    SCRF_tree_node *right_node_;
    
    SCRF_split_parameter split_param_;
    cv::Point3d p3d_; // world coordinate of the node, only available in leaf node
    
    bool is_leaf_;
    int depth_;
    
    SCRF_tree_node()
    {
        left_node_ = NULL;
        right_node_ = NULL;
        is_leaf_ = false;
        depth_ = 0;
    }
    
};



#endif /* SCRF_tree_cpp */
