//
//  RTAB_bag_of_words.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-05.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "RTAB_bag_of_words.hpp"
#include <algorithm>
#include <iostream>
#include <unordered_map>

using std::unordered_map;

bool RTAB_bag_of_words::generate_visual_words(const cv::Mat & feature_data, int num_k_Mean)
{
    assert(feature_data.type() == CV_32F);
    
    // cluster in original data
    cv::kmeans(feature_data, num_k_Mean, partition_,
               cv::TermCriteria( cv::TermCriteria::EPS+ cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_RANDOM_CENTERS, cluster_centers_);
    printf("initial number of visual word is: %d\n", cluster_centers_.rows);
    assert(num_k_Mean == cluster_centers_.rows);
    assert(cluster_centers_.type() == CV_32F);
    
    printf("calculate stop list\n");
    const double top_ratio    = 0.05;
    const double bottom_ratio = 0.1;
    vector<float> word_frequency(num_k_Mean, 0);
    for (int i = 0; i<partition_.rows; i++) {
        int cluster_center_id = partition_.at<int>(i, 0);
        assert(cluster_center_id >= 0 && cluster_center_id < num_k_Mean);
        word_frequency[cluster_center_id] += 1.0;
    }
    // normalize frequency
    for (int i = 0; i<word_frequency.size(); i++) {
        word_frequency[i] /= partition_.rows;
    }
 //   std::cout<<"word frequency is "<<cv::Mat(word_frequency)<<std::endl;
   
    // sort ratio
    vector<size_t> sorted_index = sort_indices<float>(word_frequency);
    double accumulated_ratio = 0.0;
    // lest frequnt rare words
    for (int i = 0; i<sorted_index.size() && accumulated_ratio < bottom_ratio; i++) {
        lest_frequent_words_index_.push_back(sorted_index[i]);
        accumulated_ratio += word_frequency[sorted_index[i]];
    }
    printf("remove %lu lest frequent words\n", lest_frequent_words_index_.size());
    
    // most frequent words
    accumulated_ratio = 0.0;
    for(size_t i = sorted_index.size()-1; i >= 0 && accumulated_ratio < top_ratio; i--)
    {
        most_frequent_words_index_.push_back(sorted_index[i]);
        accumulated_ratio += word_frequency[sorted_index[i]];
    }
    printf("remove %lu most frequent words\n", most_frequent_words_index_.size());
    
    // create kd tree to store cluster centers
    kdtree_.create_tree(cluster_centers_);
    return true;
}

bool RTAB_bag_of_words::generate_tf_idf_visual_words(const vector<cv::Mat> & database_features, int num_k_mean)
{
    assert(database_features.size() > 2);
    
    // generate general (non-tf-idf vocabulary tree)
    cv::Mat all_features;
    for (int i = 0; i<database_features.size(); i++) {
        all_features.push_back(database_features[i]);
    }
    generate_visual_words(all_features, num_k_mean);
    
    // generate tf-idf parameters
    num_images_ = (int)database_features.size();
    
    num_image_contain_term_ = vector<size_t>(num_k_mean, 0);
    // i: image
    // j: features in each image
    // k: partition
    for (int i = 0, k = 0; i<database_features.size(); i++) {
        vector<bool> occur(num_k_mean, false);
        for (int j = 0; j<database_features[i].rows; j++, k++) {
            assert(k < partition_.rows);
            int kmean_index = partition_.at<int>(k, 0);
            assert(kmean_index >= 0 && kmean_index < num_k_mean);
            occur[kmean_index] = true;
        }
        for (int j = 0; j<occur.size(); j++) {
            // term j occurs in image i
            if (occur[j]) {
                num_image_contain_term_[j] += 1;
            }
        }
    }
    log_N_ni_ = vector<double>(num_k_mean, 1.0);
    for (int i = 0; i<log_N_ni_.size(); i++) {
        log_N_ni_[i] = log(1.0*num_images_/num_image_contain_term_[i]);
    //    printf("num_image_contain_term_ %d \n", num_image_contain_term_[i]);
    }
    
    has_tf_idf_ = true;
    return true;
}


bool RTAB_bag_of_words::quantize_features(const cv::Mat & features, cv::Mat & word_frequency) const
{
    assert(features.type() == CV_32F);
    
    //
    vector<vector<int>> indices;
    vector<vector<float> > dists;
    kdtree_.search(features, indices, dists, 1);
    const int K = cluster_centers_.rows;
    
    word_frequency = cv::Mat::zeros(1, K, CV_32F);
    for (int i = 0; i<indices.size(); i++) {
        word_frequency.at<float>(0, indices[i][0]) += 1.0;
    }
    
    cv::Mat n_word;
    cv::normalize(word_frequency, n_word);
    word_frequency = n_word;
    return true;
}

bool RTAB_bag_of_words::quantize_features_it_idf(const cv::Mat & descriptors,
                                                 cv::Mat & word) const
{
    assert(has_tf_idf_);
    assert(descriptors.type() == CV_32F);
    
    // search
    vector<vector<int>> indices;
    vector<vector<float> > dists;
    kdtree_.search(descriptors, indices, dists, 1);
    const int K = cluster_centers_.rows;
    assert(K == log_N_ni_.size());
    
    // n_id
    word = cv::Mat::zeros(1, K, CV_32F);
    for (int i = 0; i<indices.size(); i++) {
        word.at<float>(0, indices[i][0]) += 1.0;
    }
    
    // tf-idf
    double nd = descriptors.rows;
    for (int i = 0; i<word.rows; i++) {
        double n_id = word.at<float>(0, i);
        word.at<float>(0, i) = n_id/nd*log_N_ni_[i];
    }
    
    cv::Mat n_word;
    cv::normalize(word, n_word);
    word = n_word;
    return true;
}




