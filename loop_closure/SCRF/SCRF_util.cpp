//
//  SCRF_util.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-19.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "SCRF_util.hpp"
#include "cvx_io.hpp"
#include <unordered_map>

using cv::Point3d;
using std::unordered_map;
using std::string;

cv::Point3d SCRF_Util::mean_location(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices)
{
    assert(indices.size() > 0);
    
    double x = 0;
    double y = 0;
    double z = 0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        x += samples[index].p3d_.x;
        y += samples[index].p3d_.y;
        z += samples[index].p3d_.z;
    }
    
    x /= indices.size();
    y /= indices.size();
    z /= indices.size();
    
    return cv::Point3d(x, y, z);
}

double SCRF_Util::spatial_variance(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices)
{
    cv::Point3d p_mean = SCRF_Util::mean_location(samples, indices);
    
    double var = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        Point3d dif = p_mean - samples[index].p3d_;
      //  var += dif.x * dif.x + dif.y * dif.y + dif.z * dif.z;
        var += fabs(dif.x) + fabs(dif.y) + fabs(dif.z);
    }
  //  var /= indices.size();
    return var;
}

void SCRF_Util::mean_std_position(const vector<cv::Point3d> & points, cv::Point3d & mean_pos, cv::Vec3d & std_pos)
{
    assert(points.size() > 0);
    
    const int N = (int)points.size();
    
    // mean position
    mean_pos = cv::Point3d(0, 0, 0);
    for (int i = 0; i<points.size(); i++) {
        mean_pos += points[i];
    }
    mean_pos /= (double)N;
    
    // standard deviation
    double dev_x = 0.0;
    double dev_y = 0.0;
    double dev_z = 0.0;
    for (int i = 0; i<points.size(); i++) {
        cv::Point3d dif = mean_pos - points[i];
        dev_x += dif.x * dif.x;
        dev_y += dif.y * dif.y;
        dev_z += dif.z * dif.z;
    }
    
    dev_x = sqrt(dev_x/N);
    dev_y = sqrt(dev_y/N);
    dev_z = sqrt(dev_z/N);
    
    std_pos = cv::Vec3d(dev_x, dev_y, dev_z);
}

cv::Point3d SCRF_Util::prediction_error_stddev(const vector<SCRF_testing_result> & results)
{
    assert(results.size() > 0);
    
    
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    for (int i = 0; i<results.size(); i++) {
        cv::Point3d error = results[i].predict_error;
        dx += error.x * error.x;
        dy += error.y * error.y;
        dz += error.z * error.z;
    }
    
    dx = sqrt(dx/results.size());
    dy = sqrt(dy/results.size());
    dz = sqrt(dz/results.size());
    
    cv::Point3d stddev(dx, dy, dz);
    return stddev;
}

vector<SCRF_learning_sample> SCRF_Util::randomSampleFromRgbdImages(const char * rgb_img_file,
                                                               const char * depth_img_file,
                                                               const char * camera_pose_file,
                                                               const int num_sample,
                                                               const int image_index)
{
    assert(rgb_img_file);
    assert(depth_img_file);
    assert(camera_pose_file);
    
    vector<SCRF_learning_sample> samples;
    
    cv::Mat camera_depth_img;
    cv::Mat rgb_img;
    bool is_read = cvx_io::imread_depth_16bit_to_64f(depth_img_file, camera_depth_img);
    assert(is_read);
    cvx_io::imread_rgb_8u(rgb_img_file, rgb_img);
    
    cv::Mat pose = ms_7_scenes_util::read_pose_7_scenes(camera_pose_file);
    
    const int width = rgb_img.cols;
    const int height = rgb_img.rows;
    
    cv::Mat world_depth_img      = ms_7_scenes_util::camera_depth_to_world_depth(camera_depth_img, pose);
    cv::Mat world_coordinate_img = ms_7_scenes_util::camera_depth_to_world_coordinate(camera_depth_img, pose);
    
    for (int i = 0; i<num_sample; i++) {
        int x = rand()%width;
        int y = rand()%height;
        double camera_depth = camera_depth_img.at<double>(y, x);
        if (camera_depth == 0.0 || camera_depth == ms_7_scenes_util::invalid_camera_depth()) {
            continue;
        }
        SCRF_learning_sample sp;
        sp.p_ = cv::Vec2i(x, y);
        sp.depth_ = camera_depth/1000.0; // to meter
        sp.inv_depth_ = 1.0/sp.depth_;
        sp.image_index_ = image_index;
        sp.p3d_.x = world_coordinate_img.at<cv::Vec3d>(y, x)[0];
        sp.p3d_.y = world_coordinate_img.at<cv::Vec3d>(y, x)[1];
        sp.p3d_.z = world_coordinate_img.at<cv::Vec3d>(y, x)[2];
        
        samples.push_back(sp);
    }
    printf("rgb image is %s\n", rgb_img_file);
    printf("depth image is %s\n", depth_img_file);
    printf("camera pose file is %s\n", camera_pose_file);
    
    printf("sampled %lu samples\n", samples.size());
    return samples;
}

bool SCRF_Util::readTreeParameter(const char *file_name, SCRF_tree_parameter & tree_param)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error: can not open %s\n", file_name);
        return false;
    }
    
    unordered_map<std::string, int> imap;
    while (1) {
        char s[1024] = {NULL};
        int val = 0;
        int ret = fscanf(pf, "%s %d", s, &val);
        if (ret != 2) {
            break;
        }
        imap[string(s)] = val;
    }
    assert(imap.size() == 6);
    
    tree_param.tree_num_ = imap[string("tree_num")];
    tree_param.max_depth_ = imap[string("max_depth")];
    tree_param.min_leaf_node_ = imap[string("min_leaf_node")];
    tree_param.max_pixel_offset_ = imap[string("max_pixel_offset")];
    tree_param.pixel_offset_candidate_num_ = imap[string("pixel_offset_candidate_num")];
    tree_param.split_candidate_num_ = imap[string("split_candidate_num")];
    
    fclose(pf);
    
    return true;
    
}








