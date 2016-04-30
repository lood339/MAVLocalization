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

void SCRF_Util::mean_stddev(const vector<SCRF_learning_sample> & samples,
                            const vector<unsigned int> & indices,
                            cv::Point3d & mean,
                            cv::Vec3d & stddev)
{
    assert(indices.size() > 0);
    
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
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
    
    mean = cv::Point3d(x, y, z);
    double devx = 0.0;
    double devy = 0.0;
    double devz = 0.0;
    for (int i = 0 ; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        double difx = samples[index].p3d_.x - x;
        double dify = samples[index].p3d_.y - y;
        double difz = samples[index].p3d_.z - z;
        devx += difx * difx;
        devy += dify * dify;
        devz += difz * difz;
    }
    devx = sqrt(devx/indices.size());
    devy = sqrt(devy/indices.size());
    devz = sqrt(devz/indices.size());
    stddev = cv::Vec3d(devx, devy, devz);
    return;
}

int SCRF_Util::mean_shift(const vector<cv::Point3d> & pts,
                           cv::Point3d & mean_pt,
                           cv::Vec3d & stddev,
                           const cv::TermCriteria & criteria)
{
    assert(pts.size() > 0);
    const int maxCount = criteria.maxCount;
    const double epsilon = criteria.epsilon;
    
    vector<bool> inliers(pts.size(), true);
    for (int i = 0; i<maxCount; i++) {
        cv::Point3d cur_mean(0, 0, 0);
        int num_inlier = 0;
        for (int j = 0; j<inliers.size(); j++) {
            if (inliers[j]) {
                cur_mean += pts[j];
                num_inlier++;
            }
        }
        cur_mean /= num_inlier;
        
        // reset inlier
        for (int j = 0; j<pts.size(); j++) {
            cv::Point3d dif = pts[j] - cur_mean;
            double dis2 = dif.x * dif.x + dif.y * dif.y + dif.z * dif.z;
            if (dis2 < epsilon) {
                inliers[j] = true;
            }
            else
            {
                inliers[j] = false;
            }
        }
    }
    
    
    // center point
    mean_pt = cv::Point3d(0, 0, 0);
    int num_inlier = 0;
    for (int i = 0; i<inliers.size(); i++) {
        if (inliers[i]) {
            mean_pt += pts[i];
            num_inlier++;
        }
    }
    assert(num_inlier > 0);
    mean_pt /= num_inlier;
    
    
    double devx = 0.0;
    double devy = 0.0;
    double devz = 0.0;
    for (int i = 0 ; i<inliers.size(); i++) {
        if (inliers[i]) {
            double difx = mean_pt.x - pts[i].x;
            double dify = mean_pt.y - pts[i].y;
            double difz = mean_pt.z - pts[i].z;
            devx += difx * difx;
            devy += dify * dify;
            devz += difz * difz;
        }
    }
    devx = sqrt(devx/num_inlier);
    devy = sqrt(devy/num_inlier);
    devz = sqrt(devz/num_inlier);
    stddev = cv::Vec3d(devx, devy, devz);
    
    return num_inlier;
}

void SCRF_Util::mean_shift(const vector<SCRF_learning_sample> & sample,
                           const vector<unsigned int> & indices,
                           cv::Point3d & mean_pt,
                           cv::Vec3d & stddev)
{
    cv::TermCriteria criteria;
    criteria.maxCount = 10;
    criteria.epsilon = 0.1 * 0.1;
    
    vector<cv::Point3d> pts;
    for (int i = 0; i<indices.size(); i++) {
        pts.push_back(sample[i].p3d_);
    }
    
    SCRF_Util::mean_shift(pts, mean_pt, stddev, criteria);
  //  int SCRF_Util::mean_shift(const vector<cv::Point3d> & pts,
       //                       cv::Point3d & mean_pt,
            //                  cv::Vec3d & stddev,
              //                const cv::TermCriteria & criteria)
    
}

double SCRF_Util::spatial_variance(const vector<SCRF_learning_sample> & samples, const vector<unsigned int> & indices)
{
    cv::Point3d p_mean = SCRF_Util::mean_location(samples, indices);
    
    double var = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        Point3d dif = p_mean - samples[index].p3d_;
        var += dif.x * dif.x + dif.y * dif.y + dif.z * dif.z;
      //  var += fabs(dif.x) + fabs(dif.y) + fabs(dif.z);
    }
    //var /= indices.size();
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

vector<double> SCRF_Util::prediction_error_distance(const vector<SCRF_testing_result> & results)
{
    assert(results.size() > 0);
    
    vector<double> distance;
    for (int i =0; i < results.size(); i++) {
        distance.push_back(results[i].error_distance());
    }
    return distance;
}

cv::Point3d SCRF_Util::appro_median_error(const vector<SCRF_testing_result> & results)
{
    assert(results.size() > 0);
    
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    
    vector<double> error_x;
    vector<double> error_y;
    vector<double> error_z;
    for (int i = 0; i<results.size(); i++) {
        cv::Point3d dif = results[i].gt_p3d_ - results[i].predict_p3d_;
        error_x.push_back(fabs(dif.x));
        error_y.push_back(fabs(dif.y));
        error_z.push_back(fabs(dif.z));
    }
    
    std::sort(error_x.begin(), error_x.end());
    std::sort(error_y.begin(), error_y.end());
    std::sort(error_z.begin(), error_z.end());
    
    mx = error_x[error_x.size()/2];
    my = error_y[error_y.size()/2];
    mz = error_z[error_z.size()/2];
    
    return cv::Point3d(mx, my, mz);
}


vector<SCRF_learning_sample>
SCRF_Util::randomSampleFromRgbdImages(const char * rgb_img_file,
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

    cv::Mat mask;
    cv::Mat world_coordinate = ms_7_scenes_util::camera_depth_to_world_coordinate(camera_depth_img, pose, mask);
    
    for (int i = 0; i<num_sample; i++) {
        int x = rand()%width;
        int y = rand()%height;
        
        // ignore bad depth point
        if (mask.at<unsigned char>(y, x) == 0) {
            continue;
        }
        double camera_depth = camera_depth_img.at<double>(y, x)/1000.0;
        SCRF_learning_sample sp;
        sp.p2d_ = cv::Vec2i(x, y);
        sp.depth_ = camera_depth; // to meter
        sp.inv_depth_ = 1.0/sp.depth_;
        sp.image_index_ = image_index;
        sp.p3d_.x = world_coordinate.at<cv::Vec3d>(y, x)[0];
        sp.p3d_.y = world_coordinate.at<cv::Vec3d>(y, x)[1];
        sp.p3d_.z = world_coordinate.at<cv::Vec3d>(y, x)[2];
        
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
    assert(imap.size() == 9);
    
    tree_param.tree_num_ = imap[string("tree_num")];
    tree_param.max_depth_ = imap[string("max_depth")];
    tree_param.min_leaf_node_ = imap[string("min_leaf_node")];
    tree_param.max_pixel_offset_ = imap[string("max_pixel_offset")];
    tree_param.pixel_offset_candidate_num_ = imap[string("pixel_offset_candidate_num")];
    tree_param.split_candidate_num_ = imap[string("split_candidate_num")];
    tree_param.weight_candidate_num_ = imap[string("weight_candidate_num")];
    tree_param.verbose_leaf_ = imap[string("verbose_leaf")];
    tree_param.verbose_split_ = imap[string("verbose_split")];
    
    fclose(pf);
    return true;    
}








