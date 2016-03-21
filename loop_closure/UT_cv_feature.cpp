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
#include "RTAB_incremental_kdtree.hpp"
#include <unordered_map>
#include <algorithm>
#include "RTAB_bag_of_words.hpp"

using namespace cv::xfeatures2d;
using cv::Mat;
using std::vector;
using std::string;
using std::unordered_map;

void test_cv_feature()
{
//    test_cv_surf_feature();
    test_save_surf_features();
}

void test_cv_surf_feature()
{
    double hessian = 800.0;
    cv::Ptr<cv::Feature2D> surf = SURF::create(hessian);
    
  //  string image_name("/Users/jimmy/Desktop/images/SLAM_data/new_college/PrivateSign/StereoImages/StereoImage__1225719884.662058-left.pnm");
    string image_name("/Users/jimmy/Desktop/images/LoopClosureDetection/Lip6/Lip6OutdoorDataSet/Images/outdoor_kennedylong_000000.ppm");
    cv::Mat image = cv::imread(image_name.c_str());
    //    Mat image = cv::imread("/Users/jimmy/Desktop/images/WWoS_soccer/06_30_07_00/camera1_0630/images/00023400.jpg");
    
    
    Mat mask;
    vector<cv::KeyPoint> pts;
    Mat descriptors;
    double tt = clock();
    surf->detectAndCompute(image, mask, pts, descriptors, false);
    printf("surf cost time %f\n", (clock() - tt)/CLOCKS_PER_SEC);
    
  //  pts.resize(500);
    for (int i = 0; i<pts.size(); i++) {
    //    cout<<pts[i].response<<endl;
    }
    
    cv::FileStorage fs("features.xml", cv::FileStorage::WRITE);
    
    fs<<"surf_feature"<<descriptors;
    fs<<"image_name"<<image_name;
    fs.~FileStorage();
    
    
    printf("feature data type is %d, number of feature %d , dimension of feature %d \n", descriptors.type(), descriptors.rows, descriptors.cols);
    
    {
        // read from .xml
        cv::FileStorage fread("features.xml", cv::FileStorage::READ);
        
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
    cv::FileStorage xml_saver("new_college_PrivateSign.xml", cv::FileStorage::WRITE);
    
    vector<String> files;
    cv::glob("/Users/jimmy/Desktop/images/SLAM_data/new_college/PrivateSign/StereoImages/*.pnm", files);
    
    double hessian = 800.0;
    cv::Ptr<cv::Feature2D> surf = SURF::create(hessian);
    
    for (int i = 0; i<files.size(); i += 10) {
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
        
        string str_index = std::to_string(i/10);
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
        int num_image = files.size();
        std::vector<String> image_names;
        vector<vector<cv::KeyPoint> > keypointsSeq;
        vector<cv::Mat> descriptorSeq;
        RTAB_feature_storage::read_features("new_college_PrivateSign.xml", num_image,
                                            image_names,
                                            keypointsSeq,
                                            descriptorSeq);
        
    }
}

void test_incremental_kdtree()
{
    flann::IndexParams params = flann::KDTreeIndexParams(4);
    float rebuild_threshold = 1.1;
    RTAB_incremental_kdtree<float> dynamic_kd_tree =  RTAB_incremental_kdtree<float>(params, rebuild_threshold);
    
    // read surf features
    std::vector<String> image_names;
    vector<vector<cv::KeyPoint> > keypointsSeq;
    vector<cv::Mat> descriptorSeq;
    RTAB_feature_storage::read_features("/Users/jimmy/Desktop/images/SLAM_data/new_college/privateSign_surf_40.xml",
                                        40,
                                        image_names,
                                        keypointsSeq,
                                        descriptorSeq);
    cv::Mat dataset_features;
    vector<unsigned long> data_index;
    
    unordered_map<int, bool> internal_indices;
    for (int i = 0; i<descriptorSeq.size(); i++) {
        internal_indices[i] = false;
    }
    for (int i = 0; i<descriptorSeq.size()/2; i++) {
        dataset_features.push_back(descriptorSeq[i]);
        internal_indices[i] = true;
    }
    for (int i = 0; i<dataset_features.rows; i++) {
        data_index.push_back(i);
    }
    
    // initialize kd tree
    dynamic_kd_tree.create_tree(dataset_features, data_index);
    dynamic_kd_tree.print_state();
    
    
    // randomly add and remove features
    int feature_num = 500;
    for (int i = 0; i< 40; i++) {
        int index = rand()%descriptorSeq.size();
        
        if (internal_indices[index] == true) {
            // remove features
            vector<unsigned long> data_index;
            for (int j = 0; j < descriptorSeq[index].rows; j++) {
                data_index.push_back(j + index * feature_num);
            }
            dynamic_kd_tree.remove_data(data_index);
            internal_indices[index] = false;
            
        }
        else
        {
            // add features
            vector<unsigned long> data_index;
            for (int j = 0; j < descriptorSeq[index].rows; j++) {
                data_index.push_back(j + index * feature_num);
            }
            double tt = clock();
            dynamic_kd_tree.add_data(descriptorSeq[index], data_index);
            printf("add points cost time %f\n", (clock() - tt)/CLOCKS_PER_SEC);
            internal_indices[index] = true;
        }
        dynamic_kd_tree.print_state();
        
        // search for another random index
        int search_index = rand()%descriptorSeq.size();
        if (internal_indices[search_index] == false)
        {
            printf("start search\n");
           
            vector<vector<int>>  indices;
            vector<vector<float> > dists;
            dynamic_kd_tree.search(descriptorSeq[search_index], indices, dists, 2);
            
            // simple query the most possible image
            vector<int> image_index_histogram(descriptorSeq.size(), 0);
            for (int j = 0; j<dists.size(); j++) {
                double dis1 = dists[j][0];
                double dis2 = dists[j][1];
                if (dis1 < dis2 * 0.7) {
                    image_index_histogram[indices[j][0]/feature_num] += 1;
                }
            }
            
            vector<int>::iterator iter_max = std::max_element(image_index_histogram.begin(), image_index_histogram.end());
            int target_index = std::distance(image_index_histogram.begin(), iter_max);
            
            Mat image1 = cv::imread(image_names[search_index]);
            Mat image2 = cv::imread(image_names[target_index]);
            
            printf("search result: %d %d\n", search_index, target_index);
            cv::imshow("search image", image1);
            cv::imshow("target image", image2);
            cv::waitKey();
            
        }
       // printf("i = %d\n", i);
    }    
}

void test_bayesian_filter()
{
    vector<int> data;
    data.push_back(1);
    data.push_back(4);
    data.push_back(3);
    data.push_back(2);
    
    int max_element_index = std::distance(data.begin(), std::max_element(data.begin(), data.end()));
    
    printf("max lement index is %d, value is %d\n", max_element_index, data[max_element_index]);
    
    cv::Mat a = cv::Mat::zeros(1, 3, CV_32F);
    cv::Mat b = cv::Mat::zeros(1, 3, CV_32F);
    a.at<float>(0, 1) = 1.0;
    a.at<float>(0, 2) = 2.0;
    b.at<float>(0, 1) = 0.4;
    b.at<float>(0, 2) = 0.25;
    cout<<a.dot(b)<<endl;
    
}

void test_loop_closure_detection()
{
    test_keyframe_selection();
}

void test_bag_of_words()
{
    // read all feature
    const char data_set_file[] = "/Users/jimmy/Desktop/images/LoopClosureDetection/Lip6/Lip6OutdoorDataSet/lip6_outdoor_surf_1063.xml";
    vector<cv::String> image_names;
    vector<vector<cv::KeyPoint> > keypointsSeq;
    vector<cv::Mat> descriptorSeq;
    RTAB_feature_storage::read_features(data_set_file, 1063, image_names, keypointsSeq, descriptorSeq);
    
    // randomly select 20% as feature data and another 80% as query feature
    vector<int> image_index(image_names.size());
    for (int i = 0; i<image_index.size(); i++) {
        image_index[i] = i;
    }
    std::random_shuffle(image_index.begin(), image_index.end());
    
    int stop_index = 0.1 * image_index.size();
    vector<int> train_index(image_index.begin(), image_index.begin() + stop_index);
    vector<int> test_index(image_index.begin() + stop_index, image_index.end());
    
    Mat feature_database;
    for (int i = 0; i<train_index.size(); i++) {
        feature_database.push_back(descriptorSeq[train_index[i]]);
    }
    
    RTAB_bag_of_words bow;
    int num_k_mean = 500;
    bow.generate_visual_words(feature_database, num_k_mean);
    printf("feature number is %d, and num of k mean is %d\n", feature_database.rows, num_k_mean);
    
    cv::Mat vocabulary; // vocabulary from training features
    for (int i = 0; i<train_index.size(); i++) {
        cv::Mat word_frequency;
        bow.quantize_features(descriptorSeq[train_index[i]], word_frequency);
        vocabulary.push_back(word_frequency);
    }
    
    // test on the training index
   // test_index = train_index;
    // query for the test index
    vector<std::pair<int, int> > matched_images_pairs; // test index and database image index pair
    for (int i = 0; i<test_index.size(); i++) {
        int index = test_index[i];
        cv::Mat test_word_fre;
        bow.quantize_features(descriptorSeq[index], test_word_fre);
        
        // keep minimum distance index
        float max_dot_product = 0;
        int max_index = -1;
        for (int j = 0; j<vocabulary.rows; j++) {
            double dot = test_word_fre.dot(vocabulary.row(j));  // dot product, the larger the better
            if (dot > max_dot_product) {
                max_dot_product = dot;
                max_index = train_index[j];
            }
        }
        matched_images_pairs.push_back(std::pair<int, int>(index, max_index));
        
        // show paired image
        cv::Mat query_image  = cv::imread(image_names[index].c_str());
        cv::Mat result_image = cv::imread(image_names[max_index].c_str());
        cv::imshow("query image", query_image);
        cv::imshow("result image", result_image);
        printf("dot product is %f\n", max_dot_product);
        cv::waitKey();
    }
}

void test_tf_idf_bag_of_words()
{
    // read all feature
    const char data_set_file[] = "/Users/jimmy/Desktop/images/LoopClosureDetection/Lip6/Lip6OutdoorDataSet/lip6_outdoor_surf_1063.xml";
    vector<cv::String> image_names;
    vector<vector<cv::KeyPoint> > keypointsSeq;
    vector<cv::Mat> descriptorSeq;
    RTAB_feature_storage::read_features(data_set_file, 1063, image_names, keypointsSeq, descriptorSeq);
    
    // randomly select 20% as feature data and another 80% as query feature
    vector<int> image_index(image_names.size());
    for (int i = 0; i<image_index.size(); i++) {
        image_index[i] = i;
    }
    std::random_shuffle(image_index.begin(), image_index.end());
    
    int stop_index = 0.05 * image_index.size();
    vector<int> train_index(image_index.begin(), image_index.begin() + stop_index);
    vector<int> test_index(image_index.begin() + stop_index, image_index.end());
    
    Mat feature_database;
    vector<Mat> training_features;
    for (int i = 0; i<train_index.size(); i++) {
        feature_database.push_back(descriptorSeq[train_index[i]]);
        training_features.push_back(descriptorSeq[train_index[i]]);
    }
                                                  
    RTAB_bag_of_words bow;
    int num_k_mean = 500;
    bow.generate_tf_idf_visual_words(training_features, num_k_mean);
    printf("feature number is %d, and num of k mean is %d\n", feature_database.rows, num_k_mean);
    
    cv::Mat vocabulary; // vocabulary from training features
    for (int i = 0; i<train_index.size(); i++) {
        cv::Mat word_frequency;
        bow.quantize_features_it_idf(descriptorSeq[train_index[i]], word_frequency);
        vocabulary.push_back(word_frequency);
    }
    
    // test on the training index
    // test_index = train_index;
    // query for the test index
    vector<std::pair<int, int> > matched_images_pairs; // test index and database image index pair
    for (int i = 0; i<test_index.size(); i++) {
        int index = test_index[i];
        {
            cv::Mat test_word_fre;
            bow.quantize_features_it_idf(descriptorSeq[index], test_word_fre);
            
            // keep minimum distance index
            float max_dot_product = 0;
            int max_index = -1;
            for (int j = 0; j<vocabulary.rows; j++) {
                double dot = test_word_fre.dot(vocabulary.row(j));  // dot product, the larger the better
                if (dot > max_dot_product) {
                    max_dot_product = dot;
                    max_index = train_index[j];
                }
            }
            matched_images_pairs.push_back(std::pair<int, int>(index, max_index));
            
            // show paired image
            cv::Mat query_image  = cv::imread(image_names[index].c_str());
            cv::Mat it_idf_result_image = cv::imread(image_names[max_index].c_str());
            cv::imshow("query image", query_image);
            cv::imshow("it-idf result image", it_idf_result_image);
            printf("it-idf dot product is %f\n", max_dot_product);
        }
        
        {
            cv::Mat test_word_fre;
            bow.quantize_features(descriptorSeq[index], test_word_fre);
            
            // keep minimum distance index
            float max_dot_product = 0;
            int max_index = -1;
            for (int j = 0; j<vocabulary.rows; j++) {
                double dot = test_word_fre.dot(vocabulary.row(j));  // dot product, the larger the better
                if (dot > max_dot_product) {
                    max_dot_product = dot;
                    max_index = train_index[j];
                }
            }
            matched_images_pairs.push_back(std::pair<int, int>(index, max_index));
            
            // show paired image
           
            cv::Mat word_frequency_result_image = cv::imread(image_names[max_index].c_str());
            cv::imshow("word frequency result image", word_frequency_result_image);
            printf("word frequency dot product is %f\n\n", max_dot_product);
        }
        
        cv::waitKey();
    }
}

void test_keyframe_selection()
{
    // assume all frame are seen before
    // read all feature
    const char data_set_file[] = "/Users/jimmy/Desktop/images/LoopClosureDetection/Lip6/Lip6OutdoorDataSet/lip6_outdoor_surf_1063.xml";
    vector<cv::String> image_names;
    vector<vector<cv::KeyPoint> > keypointsSeq;
    vector<cv::Mat> descriptorSeq;
    RTAB_feature_storage::read_features(data_set_file, 1063, image_names, keypointsSeq, descriptorSeq);
    
    Mat database_feature;
    for (int i = 0; i<descriptorSeq.size(); i++) {
        database_feature.push_back(descriptorSeq[i]);
    }
    
    // train
    RTAB_bag_of_words bow;
  //  int num_k_mean = 500;
  //  bow.generate_visual_words(database_feature, num_k_mean);
 //   printf("feature number is %d, and num of k mean is %d\n", database_feature.rows, num_k_mean);
    
 //   bow.write_vocabulary("bow.xml");
    bow.read_vocabulary("/Users/jimmy/Desktop/images/LoopClosureDetection/Lip6/Lip6OutdoorDataSet/vocabulary_500.xml");
    
    // randomly select 20% as feature data and another 80% as query feature
    vector<int> image_index(image_names.size());
    for (int i = 0; i<image_index.size(); i++) {
        image_index[i] = i;
    }
    std::random_shuffle(image_index.begin(), image_index.end());
    
    int stop_index = 0.2 * image_index.size();
    vector<int> train_index(image_index.begin(), image_index.begin() + stop_index);
    vector<int> test_index(image_index.begin() + stop_index, image_index.end());
    
    cv::Mat vocabulary; // vocabulary from training features
    for (int i = 0; i<train_index.size(); i++) {
        cv::Mat word_frequency;
        bow.quantize_features(descriptorSeq[train_index[i]], word_frequency);
        vocabulary.push_back(word_frequency);
    }
    
    // test on the training index
    // test_index = train_index;
    // query for the test index
    vector<std::pair<int, int> > matched_images_pairs; // test index and database image index pair
    for (int i = 0; i<test_index.size(); i++) {
        int index = test_index[i];
        cv::Mat test_word_fre;
        bow.quantize_features(descriptorSeq[index], test_word_fre);
        
        // keep minimum distance index
        float max_dot_product = 0;
        int max_index = -1;
        for (int j = 0; j<vocabulary.rows; j++) {
            double dot = test_word_fre.dot(vocabulary.row(j));  // dot product, the larger the better
            if (dot > max_dot_product) {
                max_dot_product = dot;
                max_index = train_index[j];
            }
        }
        matched_images_pairs.push_back(std::pair<int, int>(index, max_index));
        
        // show paired image
        cv::Mat query_image  = cv::imread(image_names[index].c_str());
        cv::Mat result_image = cv::imread(image_names[max_index].c_str());
        cv::imshow("query image", query_image);
        cv::imshow("result image", result_image);
        printf("dot product is %f\n", max_dot_product);
        cv::waitKey();
    }
}







