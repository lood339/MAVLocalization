//
//  SCRF_regressor.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-26.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "SCRF_regressor.hpp"
#include <string>

using std::string;

SCRF_regressor::SCRF_regressor()
{
    
}

SCRF_regressor::~SCRF_regressor()
{
    
}


bool SCRF_regressor::predict(const SCRF_learning_sample & sample,
                             const cv::Mat & rgbImage,
                             SCRF_testing_result & predict) const
{
    assert(trees_.size() > 0);
    
    vector<cv::Point3d> preds;   
    for (int i = 0; i<trees_.size(); i++) {
        SCRF_testing_result data;
        bool isPredict = trees_[i]->predict(sample, rgbImage, data);
        if (isPredict) {
            preds.push_back(data.predict_p3d_);
            
        }
    }
    if (preds.size() == 0) {
        return false;
    }
    
    SCRF_Util::mean_std_position(preds, predict.predict_p3d_, predict.std_);
    predict.predict_error = predict.predict_p3d_ - sample.p3d_;
    
    return true;
}

bool SCRF_regressor::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    //write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    SCRF_tree_parameter param = trees_[0]->param_;
    fprintf(pf, "tree_num max_depth min_leaf_node max_pixel_offset pixel_offset_num num_random_split\n");
    fprintf(pf, "%d\t %d\t %d\t %d\t %d\t %d\n", (int)trees_.size(), param.max_depth_, param.min_leaf_node_,
                param.max_pixel_offset_, param.pixel_offset_candidate_num_, param.split_candidate_num_);
    
    vector<string> tree_files;
    string baseName = string(fileName);
    baseName = baseName.substr(0, baseName.size()-4);
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string fileName = baseName + string(buf) + string(".txt");
        fprintf(pf, "%s\n", fileName.c_str());
        tree_files.push_back(fileName);
    }
    
    for (int i = 0; i<trees_.size(); i++) {
        SCRF_tree_node::write_tree(tree_files[i].c_str(), trees_[i]->root_node());
    }
    
    fclose(pf);
    printf("save to %s\n", fileName);
    return true;
}
bool SCRF_regressor::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf), pf);
    printf("%s\n", line_buf);
    
    int tree_num = 0;
    int max_depth = 0;
    int min_leaf_node = 0;
    int max_pixel_offset = 0;
    int pixel_offset_random_num = 0;
    int num_split_random = 0;
    int ret = fscanf(pf, "%d %d %d %d %d %d", &tree_num, &max_depth, &min_leaf_node, &max_pixel_offset, &pixel_offset_random_num, &num_split_random);
    assert(ret == 6);
    
    SCRF_tree_parameter param;
    param.max_depth_ = max_depth;
    param.min_leaf_node_ = min_leaf_node;
    param.max_pixel_offset_ = max_pixel_offset;
    param.pixel_offset_candidate_num_ = pixel_offset_random_num;
    param.split_candidate_num_ = num_split_random;
    
    vector<string> treeFiles;
    for (int i = 0; i<tree_num; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        treeFiles.push_back(string(buf));
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        delete trees_[i];
        trees_[i] = 0;
    }
    trees_.clear();
    
    
    for (int i = 0; i<treeFiles.size(); i++) {
        
        SCRF_tree_node * root = NULL;
        bool isRead = SCRF_tree_node::read_tree(treeFiles[i].c_str(), root);
        assert(isRead);
        SCRF_tree *tree = new SCRF_tree();
        tree->setRootNode(root);
        tree->setTreeParameter(param);
        trees_.push_back(tree);
    }
    printf("read from %s\n", fileName);   

    return true;
}


