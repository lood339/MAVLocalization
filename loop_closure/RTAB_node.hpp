//
//  RTAB_node.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-02-13.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef RTAB_node_cpp
#define RTAB_node_cpp

// location node "Appearance-Based Loop Closure Detection for Online Large-Scale and Long-Term Operation" 2013
// Real-Time Appearance-Based Mapping RTAB

#include "cvxImage_300.h"
#include <vector>

using std::vector;
using cv::Mat;


enum RTAB_memory{senscory, short_term, working, long_term};

class RTAB_node
{
    unsigned long image_index_;
    double weight_;
    Mat descriptors_;         // image discriptors, can be SURF or SIFT
    RTAB_memory memory_type_;
};

#endif /* RTAB_node_cpp */
