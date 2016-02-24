//
//  main.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-02-16.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include <iostream>
#include "cvxImage_310.hpp"
#include <vector>
#include <string>

using namespace cv::xfeatures2d;
using std::vector;
using std::string;
using namespace cv;

#if 0
static void help()
{
    printf("command: program         image_folder feature_type hessian_parameter max_feature_num save_file\n");
    printf("example: extract_feature  a/*.jpg     0            800               400             surf.xml\n");
    printf("feature_type: 0 --> surf\n");
    printf("hessian_parameter: larger --> few features\n");
}

int main(int argc, const char * argv[])
{
    if (argc != 6) {
        printf("need 6 parameters, get %d\n", argc);
        help();
        return -1;
    }
    
    const char *folder        = argv[1];
    const int feature_type    = std::stoi(string(argv[2]));
    double hessian_parameter  = std::stod(string(argv[3]));
    const int max_feature_num = std::stoi(string(argv[4]));
    const char *save_file = argv[5];
    
    assert(feature_type == 0);
    
    cv::FileStorage xml_saver(save_file, cv::FileStorage::WRITE);
    
    vector<String> files;
    cv::glob(folder, files);
    printf("load %lu files\n", files.size());
    
    
    cv::Ptr<cv::Feature2D> surf = SURF::create(hessian_parameter);
    for (int i = 0; i<files.size(); i++) {
        Mat image = cv::imread(files[i]);       
        
        vector<cv::KeyPoint> pts;
        Mat descriptors;
        surf->detect(image, pts);
        if (pts.size() > max_feature_num) {
            pts.resize(max_feature_num);
        }
        surf->compute(image, pts, descriptors);
        
        string str_index = std::to_string(i);
        string save_image_name = string("image_") + str_index;
        string keypoint_name = string("keypoint_") + str_index;
        string descript_name = string("descriptor_") + str_index;
        
        xml_saver<<save_image_name.c_str()<<files[i];
        xml_saver<<keypoint_name.c_str()<<pts;
        xml_saver<<descript_name.c_str()<<descriptors;
        
        printf("finish %d, number of feature %lu\n", i, pts.size());
    }
    
    printf("save to %s\n", save_file);
    return 0;
}

#endif
