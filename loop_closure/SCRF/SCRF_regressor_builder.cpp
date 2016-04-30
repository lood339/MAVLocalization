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
                                         const vector<cv::Mat> & rgbImages,
                                         const char *model_file_name) const
{
    vector<SCRF_tree *> trees;
    
    const unsigned int N = (unsigned int)samples.size();    
    tree_param_.printSelf();
    const int tree_number = tree_param_.tree_num_;
    for (unsigned int i = 0; i<tree_number; i++) {
        vector<unsigned int> training_data_indices;
        vector<unsigned int> outof_bag_data_indices;
        
        SCRF_regressor_builder::outof_bag_sampling(N, training_data_indices, outof_bag_data_indices);
    
        SCRF_tree *tree = new SCRF_tree();
        assert(tree);
        tree->setTreeParameter(tree_param_);
        tree->build(samples, training_data_indices, rgbImages, tree_param_);
        trees.push_back(tree);
        
        // training error
        vector<SCRF_testing_result> training_predictions;
        cv::Point3d train_error_stddev(0, 0, 0);
        for (int j = 0; j<training_data_indices.size(); j++) {
            int index = training_data_indices[j];
            SCRF_testing_result result;
            result.p2d_    = samples[index].p2d_;
            result.gt_p3d_ = samples[index].p3d_;
            bool isPredict = tree->predict(samples[index], rgbImages[samples[index].image_index_], result);
            if (isPredict) {
                training_predictions.push_back(result);
            }
        }
        
        vector<SCRF_testing_result> validation_predictions;
        for (int j = 0; j<outof_bag_data_indices.size(); j++) {
            int index = outof_bag_data_indices[j];
            SCRF_testing_result result;
            result.p2d_    = samples[index].p2d_;
            result.gt_p3d_ = samples[index].p3d_;
            bool isPredict = tree->predict(samples[index], rgbImages[samples[index].image_index_], result);
            if (isPredict) {
                validation_predictions.push_back(result);
            }
        }
      
        vector<double> training_err_dis   = SCRF_Util::prediction_error_distance(training_predictions);
        vector<double> validation_err_dis = SCRF_Util::prediction_error_distance(validation_predictions);
        
        std::sort(training_err_dis.begin(), training_err_dis.end());
        std::sort(validation_err_dis.begin(), validation_err_dis.end());
        double mdis_1 = training_err_dis[training_err_dis.size()/2];
        double mdis_2 = validation_err_dis[validation_err_dis.size()/2];
        
        cv::Point3d mp1 = SCRF_Util::appro_median_error(training_predictions);
        cv::Point3d mp2 = SCRF_Util::appro_median_error(validation_predictions);
        
        printf("tree index is %d\n", i);
        printf("training percentage: %f median distance: %f, error %lf %lf %lf \n", 1.0*training_predictions.size()/training_data_indices.size(),
               mdis_1, mp1.x, mp1.y, mp1.z);
        printf("outOfBag percentage: %f median distance: %f, error %lf %lf %lf \n", 1.0*validation_predictions.size()/outof_bag_data_indices.size(),
               mdis_2, mp2.x, mp2.y, mp2.z);
        
        model.trees_ = trees;
        if (model_file_name) {
            model.save(model_file_name);
        }
    }
    model.trees_ = trees;  
    
    return true;
}
