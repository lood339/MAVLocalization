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




