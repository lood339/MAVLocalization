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
#include <list>

using std::vector;
using std::list;
using cv::Mat;

// memory type
enum RTAB_memory{senscory, short_term, working, long_term};


class RTAB_node
{
    // property from images
    unsigned long image_index_;    // used as time index (age)
    double weight_;                // section III B
    Mat signature_;                // image feature, discriptors, can be SURF or SIFT
    RTAB_memory memory_type_;      // Fig.2
    
    // links in the graph
    RTAB_node *pre_;               // neighbor in previous (time). Fig.1
    RTAB_node *next_;              // neighbor in next (time)
    list<RTAB_node *> loop_closure_neighbors_;
    
public:
    RTAB_node()
    {
        image_index_ = 0;
        weight_ = 0;
        memory_type_ = senscory;
        
        pre_ = NULL;
        next_ = NULL;
    }
    
    RTAB_node(unsigned long index, double w, const Mat & signature, RTAB_memory type)
    {
        image_index_ = index;
        weight_ = w;
        signature_ = signature;
        memory_type_ = type;
        
        pre_ = NULL;
        next_ = NULL;
    }
    
    ~RTAB_node()
    {
        pre_ = NULL;
        next_ = NULL;
    }
};

// @todo Algorithm 2
RTAB_node * RTAB_create_location(const Mat & image);

#endif /* RTAB_node_cpp */
