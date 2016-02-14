//
//  RTAB_map.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-02-13.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "RTAB_map.hpp"
#include "RTAB_parameter.h"

RTAB_map::RTAB_map()
{
    
}
RTAB_map::~RTAB_map()
{
    
}

/*
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
 */

bool RTAB_map::weight_update(RTAB_node * l_t, RTAB_node * l_c)
{
    assert(l_t);
    assert(l_c);
    assert(l_t->signature_.rows >= RTAB_parameter::surf_T_min_feature_
           && l_c->signature_.rows >= RTAB_parameter::surf_T_min_feature_);
    // equation (1)
    int N_pair = 1.0; // @ todo match from l_t to l_c
    double s_zt_zc = 1.0 * N_pair/std::max(l_t->signature_.rows, l_c->signature_.rows);
    
    if (s_zt_zc > RTAB_parameter::T_similarity_) {
        //Lc merge to Lt
    }
    
    return true;
}
bool RTAB_map::bayesian_filter_update()
{
    return true;
}
bool RTAB_map::loop_closure_hypothesis_selection()
{
    return true;
}
bool RTAB_map::retrieval()
{
    return true;
}
bool RTAB_map::transfer()
{
    return true;
}