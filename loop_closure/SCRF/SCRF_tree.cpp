//
//  SCRF_tree.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-19.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "SCRF_tree.hpp"
#include <algorithm>
#include <iostream>
#include <limits>


using cv::Vec2i;
using namespace std;


bool SCRF_tree::build(const vector<SCRF_learning_sample> & samples,
                      const vector<unsigned int> & indices,
                      const vector<cv::Mat> & rgbImages,
                      const SCRF_tree_parameter & param)
{
    root_ = new SCRF_tree_node();
    root_->depth_ = 0;
    
    // set random number
    rng_ = cv::RNG(std::time(0) + 10000);
    param_ = param;
    return this->configure_node(samples, rgbImages, indices, 0, root_, param);
}


static vector<double> random_number_from_range(double min_val, double max_val, int rnd_num)

{
    assert(rnd_num > 0);
    
    cv::RNG rng;
    vector<double> data;
    for (int i = 0; i<rnd_num; i++) {
        data.push_back(rng.uniform(min_val, max_val));
    }
    return data;
}

static double best_split_random_parameter(const vector<SCRF_learning_sample> & samples,
                                          const vector<cv::Mat> & rgbImages,
                                          const vector<unsigned int> & indices,
                                          const SCRF_split_parameter & split_param,
                                          int min_node_size,
                                          int num_split_random,
                                          vector<unsigned int> & left_indices,
                                          vector<unsigned int> & right_indices,
                                          double & split_threshold)
{
    double min_loss = std::numeric_limits<double>::max();
    
    // calculate pixel difference
    vector<double> feature_values(indices.size(), 0.0); // 0.0 for invalid pixels
    const int c1 = split_param.c1_;
    const int c2 = split_param.c2_;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < samples.size());
        SCRF_learning_sample smp = samples[index];
        cv::Point2i p1 = smp.p2d_;
        cv::Point2i p2 = smp.add_offset(split_param.offset2_);
        
        const cv::Mat rgb_image = rgbImages[smp.image_index_];
       
        
        bool is_inside_image2 = SCRF_Util::is_inside_image(rgb_image.cols, rgb_image.rows, p2.x, p2.y);
        double pixel_1_c = 0.0;   // out of image as black pixels
        double pixel_2_c = 0.0;
        
        cv::Vec3b pix_1 = rgb_image.at<cv::Vec3b>(p1.y, p1.x);    // (row, col)
        pixel_1_c = pix_1[c1];
        
        if (is_inside_image2) {
            cv::Vec3b pixel_2 = rgb_image.at<cv::Vec3b>(p2.y, p2.x);
            pixel_2_c = pixel_2[c2];
        }
        
        feature_values[i] = pixel_1_c * split_param.w1_ + pixel_2_c * split_param.w2_;
    }
    
    double min_v = *std::min_element(feature_values.begin(), feature_values.end());
    double max_v = *std::max_element(feature_values.begin(), feature_values.end());
    
    vector<double> split_values = random_number_from_range(min_v, max_v, num_split_random);  // num_split_random = 20
   // printf("number of randomly selected spliting values is %lu\n", split_values.size());
    
    // split data by pixel difference
    bool is_split = false;
    for (int i = 0; i<split_values.size(); i++) {
        double split_v = split_values[i];
        vector<unsigned int> cur_left_index;
        vector<unsigned int> cur_right_index;
        double cur_loss = 0;
        for (int j = 0; j<feature_values.size(); j++) {
            int index = indices[j];
            if (feature_values[j] < split_v) {
                cur_left_index.push_back(index);
            }
            else
            {
                cur_right_index.push_back(index);
            }
        }
        if (cur_left_index.size() *2 < min_node_size || cur_right_index.size() * 2 < min_node_size) {
            continue;
        }
        cur_loss = SCRF_Util::spatial_variance(samples, cur_left_index);
        if (cur_loss > min_loss) {
            continue;
        }
        cur_loss += SCRF_Util::spatial_variance(samples, cur_right_index);
        if (cur_loss < min_loss) {
            is_split = true;
            min_loss = cur_loss;
            left_indices  = cur_left_index;
            right_indices = cur_right_index;
            split_threshold = split_v;
        }
    }
    if (!is_split) {
        return min_loss;
    }
    assert(left_indices.size() + right_indices.size() == indices.size());
    
    return min_loss;
}

static double best_split_weight(const vector<SCRF_learning_sample> & samples,
                                const vector<cv::Mat> & rgbImages,
                                const vector<unsigned int> & indices,
                                const SCRF_split_parameter & split_param,  // inoutput
                                const vector<cv::Vec2d> & candidate_weights,
                                int min_node_size,
                                int num_split_random,
                                vector<unsigned int> & left_indices,
                                vector<unsigned int> & right_indices,
                                double & split_threshold,
                                double & w1,
                                double & w2)
{
    double min_loss = std::numeric_limits<double>::max();
    
    for (int i = 0; i<candidate_weights.size(); i++) {
        SCRF_split_parameter cur_split_param = split_param;
        cur_split_param.w1_ = candidate_weights[i][0];
        cur_split_param.w2_ = candidate_weights[i][1];
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        double cur_split_value = 0.0;
        double cur_loss = best_split_random_parameter(samples,
                                                      rgbImages,
                                                      indices,
                                                      cur_split_param,
                                                      min_node_size,
                                                      num_split_random,
                                                      cur_left_indices,
                                                      cur_right_indices,
                                                      cur_split_value);
        if (cur_loss < min_loss) {
            min_loss = cur_loss;
            split_threshold = cur_split_value;
            left_indices  = cur_left_indices;
            right_indices = cur_right_indices;
            w1 = cur_split_param.w1_;
            w2 = cur_split_param.w2_;
        }
    }
    
    return min_loss;
    
}


bool SCRF_tree::configure_node(const vector<SCRF_learning_sample> & samples,
                               const vector<cv::Mat> & rgbImages,
                               const vector<unsigned int> & indices,
                               int depth,
                               SCRF_tree_node *node,
                               const SCRF_tree_parameter & param)
                               
{
    assert(indices.size() <= samples.size());
    
    if (depth >= param.max_depth_ || indices.size() < param.min_leaf_node_) {
        node->depth_   = depth;
        node->is_leaf_ = true;
        node->node_size_ = (int)indices.size();
        SCRF_Util::mean_stddev(samples, indices, node->p3d_, node->stddev_);
        if (param_.verbose_leaf_) {
            printf("depth, num_leaf_node, %d, %lu\n", depth, indices.size());
            cout<<"mean      location: "<<node->p3d_<<endl;
            cout<<"standard deviation: "<<node->stddev_<<endl;
        }
       
        return true;
    }    
    
    const int max_pixel_offset = param.max_pixel_offset_;
    const int max_channel = 3;
    const int max_random_num = param.pixel_offset_candidate_num_;
    const int min_node_size  = param.min_leaf_node_;
    const int num_split_random = param.split_candidate_num_;
    const int num_weight = param.weight_candidate_num_;
    double min_loss = std::numeric_limits<double>::max();
    
    vector<cv::Vec2d> random_weights;
    random_weights.push_back(cv::Vec2d(1.0, -1.0));
    for (int i = 0; i<num_weight; i++) {
        double w1 = rng_.uniform(-1.0, 1.0);
        double w2 = rng_.uniform(-1.0, 1.0);
        random_weights.push_back(cv::Vec2d(w1, w2));
    }
    
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    SCRF_split_parameter split_param;
    bool is_split = false;
    for (int i = 0; i<max_random_num; i++) {
        double x2 = rng_.uniform(-(double)max_pixel_offset, (double)max_pixel_offset);
        double y2 = rng_.uniform(-(double)max_pixel_offset, (double)max_pixel_offset);
        int c1 = rand()%max_channel;
        int c2 = rand()%max_channel;
        
        SCRF_split_parameter cur_split_param;
        cur_split_param.offset2_ = cv::Point2d(x2, y2);
        cur_split_param.c1_ = c1;
        cur_split_param.c2_ = c2;
        
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        double cur_split_threshold = 0.0;
        double w1 = 0.0;
        double w2 = 0.0;
        
        double cur_loss = best_split_weight(samples, rgbImages, indices, cur_split_param,
                                            random_weights,
                                            min_node_size, num_split_random,
                                            cur_left_indices,
                                            cur_right_indices,
                                            cur_split_threshold,
                                            w1,
                                            w2);
        if (cur_loss < min_loss) {
            is_split = true;
            min_loss = cur_loss;
            left_indices  = cur_left_indices;
            right_indices = cur_right_indices;
            split_param = cur_split_param;
            split_param.threhold_ = cur_split_threshold;
            split_param.w1_ = w1;
            split_param.w2_ = w2;
        }
    }
    
    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (param_.verbose_split_) {
            printf("left, right node number is %lu %lu, percentage: %f loss: %lf\n",
                   left_indices.size(),
                   right_indices.size(),
                   1.0*left_indices.size()/indices.size(),
                   min_loss);
            split_param.print_self();
        }
        
        node->split_param_ = split_param;
        node->node_size_ = (int)indices.size();
    
        SCRF_tree_node *left_node = new SCRF_tree_node(depth);
        this->configure_node(samples, rgbImages, left_indices, depth + 1, left_node, param);
        node->left_child_ = left_node;
        
        SCRF_tree_node *right_node = new SCRF_tree_node(depth);
        this->configure_node(samples, rgbImages, right_indices, depth + 1, right_node, param);
        node->right_child_ = right_node;
        return true;
    }
    else
    {
     //   printf("split failed!\n");
        node->depth_   = depth;
        node->is_leaf_ = true;
        node->node_size_ = (int)indices.size();
        SCRF_Util::mean_stddev(samples, indices, node->p3d_, node->stddev_);
        if (param_.verbose_leaf_) {
            printf("depth, num_leaf_node, %d, %lu\n", depth, indices.size());
            cout<<"mean      location: "<<node->p3d_<<endl;
            cout<<"standard deviation: "<<node->stddev_<<endl;
        }
        return true;
    }
}

bool SCRF_tree::predict(const SCRF_learning_sample & sample,
                        const cv::Mat & rgbImage,
                        SCRF_testing_result & predict) const
{
    assert(root_);
    return this->predict(root_, sample, rgbImage, predict);
}

bool SCRF_tree::predict(const SCRF_tree_node * const node,
                        const SCRF_learning_sample & sample,
                        const cv::Mat & rgbImage,
                        SCRF_testing_result & predict) const
{
    if (node->is_leaf_) {
        predict.predict_p3d_  = node->p3d_;
        return true;
    }
    else
    {
        cv::Point2i p1 = sample.p2d_;
        cv::Point2i p2 = sample.add_offset(node->split_param_.offset2_);
        
        
        bool is_inside_image2 = SCRF_Util::is_inside_image(rgbImage.cols, rgbImage.rows, p2.x, p2.y);
        cv::Vec3b pixel_1 = rgbImage.at<cv::Vec3b>(p1.y, p1.x);
        
        double pixel_1_c = pixel_1[node->split_param_.c1_];
        double pixel_2_c = 0.0;
        
        if (is_inside_image2)
        {
            cv::Vec3b pixel_2 = rgbImage.at<cv::Vec3b>(p2.y, p2.x);
            pixel_2_c = pixel_2[node->split_param_.c2_];            
        }
        else
        {
            pixel_2_c = 0.0;
        }
        
        double pixel_dif = pixel_1_c * node->split_param_.w1_ + pixel_2_c * node->split_param_.w2_;
        if (pixel_dif < node->split_param_.threhold_ && node->left_child_) {
            return this->predict(node->left_child_, sample, rgbImage, predict);
        }
        else if(node->right_child_)
        {
            return this->predict(node->right_child_, sample, rgbImage, predict);
        }
        else
        {
            return false;
        }
    }
}



/**************************   SCRF_tree_node   ********************/
 

static void write_SCRF_prediction(FILE *pf, SCRF_tree_node * node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    // write current node
    SCRF_split_parameter param = node->split_param_;
    fprintf(pf, "%d %d\t %d %d\t %12.6f %12.6f %d\t %12.6f %12.6f %12.6f\t %12.6f %12.6f %12.6f\t %12.6f %12.6f %12.6f\n",
            node->depth_,
            (int)node->is_leaf_,
            node->node_size_,
            param.c1_,
            param.offset2_.x, param.offset2_.y, param.c2_,
            param.w1_, param.w2_, param.threhold_,
            node->p3d_.x, node->p3d_.y, node->p3d_.z,
            node->stddev_[0], node->stddev_[1], node->stddev_[2]);
    
    write_SCRF_prediction(pf, node->left_child_);
    write_SCRF_prediction(pf, node->right_child_);
}


bool SCRF_tree_node::write_tree(const char *fileName, SCRF_tree_node * root)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "depth\t isLeaf\t nodeSize\t c1\t displace2\t c2\t w1 w2\t threshold\t wld_3d\t stddev\n");
    
    write_SCRF_prediction(pf, root);
    fclose(pf);
    return true;
}


static void read_rf_constant_prediction(FILE *pf, SCRF_tree_node * & node)
{
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    if (!ret) {
        node = NULL;
        return;
    }
    if (lineBuf[0] == '#') {
        // empty node
        node = NULL;
        return;
    }
    
    // read node parameters
    node = new SCRF_tree_node();
    assert(node);
    int depth = 0;
    int isLeaf = 0;
    int nodeSize = 0;
    
    double d2[2] = {0.0};
    int c1 = 0;        // rgb image channel
    int c2 = 0;
    double wt[2] = {0.0};
    double threhold = 0;
    double xyz[3] = {0.0};
    double xyzStddev[3] = {0.0};
    
    int ret_num = sscanf(lineBuf, "%d %d %d %d %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf",  &depth, &isLeaf, &nodeSize,
           &c1,
           &d2[0], &d2[1], &c2,
           &wt[0], &wt[1],
           &threhold,
           &xyz[0], &xyz[1], &xyz[2],
           &xyzStddev[0], &xyzStddev[1], &xyzStddev[2]);
    assert(ret_num == 16);
    
    
    node->depth_ = depth;
    node->is_leaf_ = (isLeaf == 1);
    node->p3d_ = cv::Point3d(xyz[0], xyz[1], xyz[2]);
    node->stddev_ = cv::Vec3d(xyzStddev[0], xyzStddev[1], xyzStddev[2]);
    
    SCRF_split_parameter param;    
    param.offset2_ = cv::Point2d(d2[0], d2[1]);
    param.c1_ = c1;
    param.c2_ = c2;
    param.w1_ = wt[0];
    param.w2_ = wt[1];
    param.threhold_ = threhold;
    
    node->split_param_ = param;
    
    node->left_child_  = NULL;
    node->right_child_ = NULL;
    
    read_rf_constant_prediction(pf, node->left_child_);
    read_rf_constant_prediction(pf, node->right_child_);
}



bool SCRF_tree_node::read_tree(const char *fileName, SCRF_tree_node * & root)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    //read first line
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf), pf);
    printf("%s\n", line_buf);
    read_rf_constant_prediction(pf, root);
    fclose(pf);
    return true;
}







