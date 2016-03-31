//
//  SCRF_regressor_builder.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-26.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "SCRF_regressor_builder.hpp"
#include "SCRF_util.hpp"



SCRF_regressor_builder::~SCRF_regressor_builder()
{
    
}


void SCRF_regressor_builder::outof_bag_sampling(const unsigned int N,
                                                vector<unsigned int> & bootstrapped,
                                                vector<unsigned int> & outof_bag)
{
    vector<bool> isPicked(N, false);
    for (int i = 0; i<N; i++) {
        int rnd = rand()%N;
        bootstrapped.push_back(rnd);
        isPicked[rnd] = true;
    }
    
    for (int i = 0; i<N; i++) {
        if (!isPicked[i]) {
            outof_bag.push_back(i);
        }
    }
}



//Build model from data
bool SCRF_regressor_builder::build_model(SCRF_regressor& model,
                                         const vector<SCRF_learning_sample> & samples,
                                         const vector<cv::Mat> & rgbImages) const
{
    vector<SCRF_tree *> trees;
    
    double training_error = 0;
    double out_of_bag_error = 0;
    const unsigned int N = (unsigned int)samples.size();
    
    
    tree_param_.printSelf();
    
    for (unsigned int i = 0; i<tree_number_; i++) {
        vector<unsigned int> training_data_indices;
        vector<unsigned int> outof_bag_data_indices;
        
        SCRF_regressor_builder::outof_bag_sampling(N, training_data_indices, outof_bag_data_indices);
    
        SCRF_tree *tree = new SCRF_tree();
        assert(tree);
        tree->verbose_ = false;
        tree->build(samples, training_data_indices, rgbImages, tree_param_);
        trees.push_back(tree);
        
        // training error
        vector<SCRF_testing_result> training_predictions;
        cv::Point3d train_error_stddev(0, 0, 0);
        for (int j = 0; j<training_data_indices.size(); j++) {
            int index = training_data_indices[j];
            SCRF_testing_result result;
            bool isPredict = tree->predict(samples[index], rgbImages[samples[index].image_index_], result);
            if (isPredict) {
                training_predictions.push_back(result);
            }
        }
        
        vector<SCRF_testing_result> validation_predictions;
        for (int j = 0; j<outof_bag_data_indices.size(); j++) {
            int index = outof_bag_data_indices[j];
            SCRF_testing_result result;
            bool isPredict = tree->predict(samples[index], rgbImages[samples[index].image_index_], result);
            if (isPredict) {
                validation_predictions.push_back(result);
            }
        }
        
        cv::Point3d training_error   = SCRF_Util::prediction_error_stddev(training_predictions);
        cv::Point3d validation_error = SCRF_Util::prediction_error_stddev(validation_predictions);
       
        printf("tree index is %d\n", i);
        printf("predicted training            percentage: %f stddev: %f %f %f\n", 1.0*training_predictions.size()/training_data_indices.size(),
                                                                         training_error.x, training_error.y, training_error.z);
        printf("predicted outOfBag validation percentage: %f stddev: %f %f %f\n", 1.0*validation_predictions.size()/outof_bag_data_indices.size(),
                                                                         validation_error.x, validation_error.y, validation_error.z);
        
        
    }
    model.trees_ = trees;   
    
    return true;
}
