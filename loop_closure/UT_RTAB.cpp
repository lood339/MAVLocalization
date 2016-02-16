//
//  UT_RTAB.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy & Lili on 2016-02-13.
//  Copyright © 2016 jimmy & Lili. All rights reserved.
//


#include "UT_RTAB.hpp"
//#include "RTAB_feature_extraction.h"
//#include "RTAB_node.hpp"
//#include "RTAB_map.hpp"
#include "RTAB_incremental_vocabulary.hpp"
//#include "RTAB_feature_extraction.hpp"

using namespace cv;

void test_RTAB()
{
    //RTAB_node node();

    //vocabulary32 voc;
}

/*
void test_feature_extraction()
{
    string newImageName="/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_000351.ppm";
    cv::Mat newImg=imread(newImageName, CV_LOAD_IMAGE_GRAYSCALE);

    RTAB_feature_extraction featureExtrac;
    Mat descriptors;
    featureExtrac.RTAB_feature_extraction_exe(newImg, descriptors);
}*/

void test_incremental_vocabulary()
{
    int imgNum=300;

    vector<Mat>  imgVec;
    imgVec.resize(imgNum);

    vector<string> nameVec;
    nameVec.resize(imgNum);

    vector<vector<KeyPoint> > keyPointsVec;
    keyPointsVec.resize(imgNum);

    vector<Mat> descriptorsVec;
    descriptorsVec.resize(imgNum);

    for(int i=0; i<imgNum; i++)
    {
        char fileName[1024] ={NULL};

        sprintf(fileName, "/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_%06d.ppm", i);

        nameVec[i]=string(fileName);

        imgVec[i]=imread(nameVec[i], CV_LOAD_IMAGE_GRAYSCALE);
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector(minHessian);

    SurfDescriptorExtractor extractor;

    vector<unsigned int> labels;
    for(int i=0; i<imgNum; i++)
    {
        detector.detect(imgVec[i], keyPointsVec[i]);

        extractor.compute(imgVec[i], keyPointsVec[i], descriptorsVec[i]);
        for(int j = 0; j<descriptorsVec[i].rows; j++)
        {
            labels.push_back(i);
        }
    }

    Mat all_descriptors;

    for(int i = 0; i<descriptorsVec.size(); i++)
    {
        all_descriptors.push_back(descriptorsVec[i]);
    }

    assert(labels.size() == all_descriptors.rows);
    cout<<"all_descriptors.rows "<<all_descriptors.rows<<endl;
    cout<<"hahha1 "<<endl;


    RTAB_incremental_vocabulary<float> vocab(imgNum);

    vocab.indexedDescriptors_ = all_descriptors;


    vector<KeyPoint> newKeypoints;
    Mat newDescriptors;

    ///add new image to the randomized kd tree
    {
        string newImageName="/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_000350.ppm";
        Mat newImg=imread(newImageName, CV_LOAD_IMAGE_GRAYSCALE);
        detector.detect(newImg, newKeypoints);
        extractor.compute(newImg, newKeypoints, newDescriptors);
        cout<<"newDescriptors.rows: "<<newDescriptors.rows<<endl;
    }

    vocab.notIndexedDescriptors_ = newDescriptors;

    ///clustering
    int clustersNum;
    Mat clusters(15000,64,CV_32F);
    //Mat float_all_descriptors;


    clustersNum=vocab.clustering(all_descriptors, clusters);
    cout<<"clustersNum  "<<clustersNum<<endl;

    ///flann build tree
    clock_t begin1 = clock();
    vocab.create_and_update_vocab();
    clock_t end1 = clock();
    double buildTree_time = double(end1 - begin1) / CLOCKS_PER_SEC;
    cout.precision(5);
    cout<<"buildTree time "<<buildTree_time<<endl;


    cout<<"hahha2 "<<endl;
    vector<KeyPoint> queryKeypoints;
    Mat queryDescriptors;

    ///QueryImage
    {
        string queryImageName="/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_000381.ppm";
        //string queryImageName="/home/lili/workspace/SLAM/evaluation/DatasetLoopClosure/New_College_Images/0001.jpg";
        Mat queryImg=imread(queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
        detector.detect(queryImg, queryKeypoints);
        extractor.compute(queryImg, queryKeypoints, queryDescriptors);
        cout<<"queryDescriptors.rows: "<<queryDescriptors.rows<<endl;
    }

    Mat indices;
    Mat results;
    Mat dists;
    int k=1;
    vector<int> imageLabelsVec;

    multimap<int, int> imageScore;
    vector<RankedScore> rankedScore;
    vocab.search_image(queryDescriptors, indices, dists, k, imgNum, labels, imageLabelsVec, imageScore, rankedScore);

    vector<double> likelihood;
    vocab.computeLikelihood(imgNum, rankedScore, likelihood);


}
