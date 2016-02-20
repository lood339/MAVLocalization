//
//  RTAB_feature_storage.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-02-19.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "RTAB_feature_storage.hpp"
#include "cvxImage_310.hpp"

using cv::FileStorage;

bool RTAB_feature_storage::save_features(const char *xml_fileName,
                                         const vector<String> & image_names,
                                         const vector<vector<cv::KeyPoint> > & keypointsSeq,
                                         const vector<cv::Mat> & descriptorSeq)
{
    assert(xml_fileName);
    assert(image_names.size() == keypointsSeq.size());
    assert(image_names.size() == descriptorSeq.size());
    
    cv::FileStorage fs;
    bool isOpen = fs.open(xml_fileName, FileStorage::WRITE);
    if (!isOpen) {
        printf("file open error: %s\n", xml_fileName);
        return false;
    }
    
    for (int i = 0; i<image_names.size(); i++) {
        string str_index = std::to_string(i);
        string image_name = string("image_") + str_index;
        string keypoint_name = string("keypoint_") + str_index;
        string descriptor_name = string("descriptor_") + str_index;
        
        fs<<image_name.c_str()<<image_names[i];
        fs<<keypoint_name.c_str()<<keypointsSeq[i];
        fs<<descriptor_name.c_str()<<descriptorSeq[i];
    }
    printf("write %lu image and their descriptors\n", image_names.size());
    
    return true;
}

bool RTAB_feature_storage::read_features(const char *fileName,
                                         int num_image,
                                         vector<String> & image_names,
                                         vector<vector<cv::KeyPoint> > & keypointsSeq,
                                         vector<cv::Mat> & descriptorSeq)
{
    cv::FileStorage fs;
    bool isOpen = fs.open(fileName, FileStorage::READ);
    if (!isOpen) {
        printf("file open error: %s\n", fileName);
        return false;
    }
    for (int i = 0; i<num_image; i++) {
        string str_index = std::to_string(i);
        string image_name = string("image_") + str_index;
        string keypoint_name = string("keypoint_") + str_index;
        string descriptor_name = string("descriptor_") + str_index;
        
        cv::String im_name;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        fs[image_name]>>im_name;
        fs[keypoint_name]>>keypoints;
        fs[descriptor_name]>>descriptor;
        
        image_names.push_back(im_name);
        keypointsSeq.push_back(keypoints);
        descriptorSeq.push_back(descriptor);
    }
    
    printf("read %lu images and their features\n", image_names.size());
    return true;
}
