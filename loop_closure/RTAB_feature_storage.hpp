//
//  RTAB_feature_storage.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-02-19.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef RTAB_feature_storage_cpp
#define RTAB_feature_storage_cpp

#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

// wrap opencv for feature storage
using std::vector;
using std::string;
using cv::String;
class RTAB_feature_storage
{
    
public:
    // save features to xml file
    static bool save_features(const char *xml_fileName,
                              const vector<String> & image_names,
                              const vector<vector<cv::KeyPoint> > & keypointsSeq,
                              const vector<cv::Mat> & descriptorSeq);
    
    // assume features are from a sequence of images
    static bool read_features(const char *xml_fileName,
                              int num_image,
                              vector<String> & image_names,
                              vector<vector<cv::KeyPoint> > & keypointsSeq,
                              vector<cv::Mat> & descriptorSeq);
    
    
    
};


#endif /* RTAB_feature_storage_cpp */
