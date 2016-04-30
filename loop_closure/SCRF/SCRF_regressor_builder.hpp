//
//  SCRF_regressor_builder.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-26.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef SCRF_regressor_builder_cpp
#define SCRF_regressor_builder_cpp

#include "SCRF_regressor.hpp"
#include <string>


class SCRF_regressor_builder
{
    
    
    SCRF_tree_parameter tree_param_;
    
public:
    SCRF_regressor_builder()
    {
        
    }
    ~SCRF_regressor_builder();
    
    
    void setTreeParameter(const SCRF_tree_parameter & param)
    {
        tree_param_ = param;
    }
    
    //Build model from data    
    bool build_model(SCRF_regressor& model,
                     const vector<SCRF_learning_sample> & samples,
                     const vector<cv::Mat> & rgbImages,
                     const char *model_file_name = NULL) const;
   
    
    //: Name of the class
    std::string is_a() const {return std::string("SCRF_regressor_builder");};
    
public:
    static void outof_bag_sampling(const unsigned int N,
                                   vector<unsigned int> & bootstrapped,
                                   vector<unsigned int> & outof_bag);
    
};

#endif /* SCRF_regressor_builder_cpp */
