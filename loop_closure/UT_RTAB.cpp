//
//  UT_RTAB.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-02-13.
//  Copyright © 2016 jimmy. All rights reserved.
//


#include "UT_RTAB.hpp"
#include "RTAB_feature_extraction.h"
//#include "RTAB_node.hpp"
//#include "RTAB_map.hpp"
//#include "RTAB_incremental_vocabulary.hpp"
//#include "RTAB_feature_extraction.hpp"

void test_RTAB()
{
    //RTAB_node node();

    //vocabulary32 voc;
}

void test_feature_extraction()
{
    string newImageName="/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_000351.ppm";
    Mat newImg=imread(newImageName, CV_LOAD_IMAGE_GRAYSCALE);

    RTAB_feature_extraction featureExtrac;
    Mat descriptors;
    featureExtrac.RTAB_feature_extraction_exe(newImg, descriptors);
}

