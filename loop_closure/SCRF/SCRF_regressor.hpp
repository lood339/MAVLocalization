//
//  SCRF_regressor.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-26.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef SCRF_regressor_cpp
#define SCRF_regressor_cpp

#include "SCRF_tree.hpp"
#include <vector>

using std::vector;

class SCRF_regressor
{
    friend class SCRF_regressor_builder;
    
    vector<SCRF_tree *> trees_;
    
public:
    SCRF_regressor();
    ~SCRF_regressor();
    
    
    bool predict(const SCRF_learning_sample & sample,
                 const cv::Mat & rgbImage,
                 SCRF_testing_result & predict) const;
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
    
};

#endif /* SCRF_regressor_cpp */
