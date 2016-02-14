//
//  RTAB_map.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-02-13.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef RTAB_map_cpp
#define RTAB_map_cpp

#include "RTAB_node.hpp"
#include <list>

using std::list;

class RTAB_map
{
    list<RTAB_node *> short_term_memory_pool_;
    list<RTAB_node *> working_memory_pool_;
    list<RTAB_node *> long_term_memory_pool_;
    
    
public:
    RTAB_map();
    ~RTAB_map();
    
    // l_t: current location
    // l_c: last one in STM
    bool weight_update(RTAB_node * l_t, RTAB_node * l_c);
    bool bayesian_filter_update();
    bool loop_closure_hypothesis_selection();
    bool retrieval();
    bool transfer();    
};


#endif /* RTAB_map_cpp */
