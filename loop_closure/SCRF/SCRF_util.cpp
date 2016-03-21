//
//  SCRF_util.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-19.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "SCRF_util.hpp"

using cv::Point3d;
cv::Point3d SCRF_Util::mean_location(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices)
{
    assert(indices.size() > 0);
    
    double x = 0;
    double y = 0;
    double z = 0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        x += samples[index].p3d_.x;
        y += samples[index].p3d_.y;
        z += samples[index].p3d_.z;
    }
    
    x /= indices.size();
    y /= indices.size();
    z /= indices.size();
    
    return cv::Point3d(x, y, z);
}

double SCRF_Util::spatial_variance(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices)
{
    cv::Point3d p_mean = SCRF_Util::mean_location(samples, indices);
    
    double var = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        Point3d dif = p_mean - samples[index].p3d_;
      //  var += dif.x * dif.x + dif.y * dif.y + dif.z * dif.z;
        var += fabs(dif.x) + fabs(dif.y) + fabs(dif.z);
    }
  //  var /= indices.size();
    return var;
}