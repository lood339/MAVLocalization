//
//  UT_cv_feature.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-02-16.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "UT_cv_feature.hpp"
#include "cvxImage_310.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <time.h>
#include <string>
#include "RTAB_feature_storage.hpp"

using namespace cv::xfeatures2d;
using namespace cv;
using std::vector;
using std::string;

void test_cv_feature()
{
//    test_cv_surf_feature();
    test_save_surf_features();
}

void test_cv_surf_feature()
{
    double hessian = 800.0;
    Ptr<Feature2D> surf = SURF::create(hessian);
    
    string image_name("/Users/jimmy/Desktop/images/SLAM_data/new_college/PrivateSign/StereoImages/StereoImage__1225719884.662058-left.pnm");
    Mat image = cv::imread("/Users/jimmy/Desktop/images/SLAM_data/new_college/PrivateSign/StereoImages/StereoImage__1225719884.662058-left.pnm");
    //    Mat image = cv::imread("/Users/jimmy/Desktop/images/WWoS_soccer/06_30_07_00/camera1_0630/images/00023400.jpg");
    
    Mat mask;
    vector<cv::KeyPoint> pts;
    Mat descriptors;
    double tt = clock();
    surf->detectAndCompute(image, mask, pts, descriptors, false);
    printf("surf cost time %f\n", (clock() - tt)/CLOCKS_PER_SEC);
    
    pts.resize(500);
    for (int i = 0; i<pts.size(); i++) {
        cout<<pts[i].response<<endl;
    }
    
    FileStorage fs("features.xml", FileStorage::WRITE);
    
    fs<<"surf_feature"<<descriptors;
    fs<<"image_name"<<image_name;
    fs.~FileStorage();
    
    
    printf("feature data type is %d, number of feature %d , dimension of feature %d \n", descriptors.type(), descriptors.rows, descriptors.cols);
    
    {
        // read from .xml
        FileStorage fread("features.xml", FileStorage::READ);
        
        Mat data;
        string image_name;
        fread["surf_feature"]>>data;
        fread["image_name"]>>image_name;
        printf("read data dimension %d %d %d, image name %s\n", data.rows, data.cols, data.type(), image_name.c_str());
    }
    
    Mat outImage;
    cv::drawKeypoints(image, pts, outImage);
    cv::imshow("surf", outImage);
    
    cv::waitKey();
}

void test_save_surf_features()
{
    // glob
    // void glob(String pattern, std::vector<String>& result, bool recursive = false);
    FileStorage xml_saver("new_college_PrivateSign.xml", FileStorage::WRITE);
    
    vector<String> files;
    cv::glob("/Users/jimmy/Desktop/images/SLAM_data/new_college/PrivateSign/StereoImages/*.pnm", files);
    
    double hessian = 800.0;
    Ptr<Feature2D> surf = SURF::create(hessian);
    
    for (int i = 0; i<files.size(); i+= 10) {
        Mat image = cv::imread(files[i]);
        
        Mat mask;
        vector<cv::KeyPoint> pts;
        Mat descriptors;
        surf->detect(image, pts);
        if (pts.size() < 500) {
            printf("num_surf less than 500\n");
            continue;
        }
        pts.resize(500);
        surf->compute(image, pts, descriptors);
        
        string str_index = std::to_string(i);
        string save_image_name = string("image_") + str_index;
        string surf_name = string("descriptor_") + str_index;
        string surf_keypoint = string("keypoint_") + str_index;
        xml_saver<<save_image_name.c_str()<<files[i];
        xml_saver<<surf_name.c_str()<<descriptors;
        xml_saver<<surf_keypoint.c_str()<<pts;
        
        printf("finish %d\n", i);
    }
    
    xml_saver.release();
    
    {
        // test read
        int num_image = files.size()/10;
        std::vector<String> image_names;
        vector<vector<cv::KeyPoint> > keypointsSeq;
        vector<cv::Mat> descriptorSeq;
        RTAB_feature_storage::read_features("new_college_PrivateSign.xml", num_image,
                                            image_names,
                                            keypointsSeq,
                                            descriptorSeq);
        
        
    }
}