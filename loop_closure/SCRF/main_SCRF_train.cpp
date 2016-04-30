//
//  main.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-02-16.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include <iostream>
#include "cvxImage_310.hpp"
#include "SCRF_regressor.hpp"
#include "SCRF_regressor_builder.hpp"
#include <string>
#include "cvx_io.hpp"
#include "SCRF_util.hpp"

using std::string;

#if 0

static vector<string> read_file_names(const char *file_name)
{
    vector<string> file_names;
    FILE *pf = fopen(file_name, "r");
    assert(pf);
    while (1) {
        char line[1024] = {NULL};
        int ret = fscanf(pf, "%s", line);
        if (ret != 1) {
            break;
        }
        file_names.push_back(string(line));
    }
    printf("read %lu lines\n", file_names.size());
    fclose(pf);
    return file_names;
}

static void help()
{
    printf("program    RGBImageList depthImageList cameraPoseList decisionTreeParameterFile numSamplePerImage saveFile\n");
    printf("SCRF_train rgbs.txt     depth.txt      poses.txt      RF_param.txt              5000              scrf.txt\n");
    printf("parameter fits to MS 7 scenes dataset\n");
}


int main(int argc, const char * argv[])
{
    if (argc != 7) {
        printf("argc is %d, should be 7\n", argc);
        help();
        return -1;
    }
    const char * rgb_image_file = argv[1];
    const char * depth_image_file = argv[2];
    const char * camera_to_wld_pose_file = argv[3];
    const char * tree_param_file = argv[4];
    const int num_random_sample = (int)strtod(argv[5], NULL);
    const char * save_model_file = argv[6];
    
    assert(num_random_sample > 100);
    
    vector<SCRF_learning_sample> all_samples;
    vector<cv::Mat> rgb_images;
    
    vector<string> rgb_files = read_file_names(rgb_image_file);
    vector<string> depth_files = read_file_names(depth_image_file);
    vector<string> pose_files = read_file_names(camera_to_wld_pose_file);
    
    assert(rgb_files.size() == depth_files.size());
    assert(rgb_files.size() == pose_files.size());
    
    // read tree parameter
    SCRF_regressor_builder builder;
    SCRF_regressor model;
    SCRF_tree_parameter tree_param;
    bool is_read = SCRF_Util::readTreeParameter(tree_param_file, tree_param);
    assert(is_read);
    
    // read rgb files
    for (int i = 0; i<rgb_files.size(); i++) {
        const char *rgb_img_file     = rgb_files[i].c_str();
        const char *depth_img_file   = depth_files[i].c_str();
        const char *pose_file        = pose_files[i].c_str();
        
        cv::Mat camera_depth_img;
        cv::Mat rgb_img;
        bool is_read = cvx_io::imread_depth_16bit_to_64f(depth_img_file, camera_depth_img);
        assert(is_read);
        cvx_io::imread_rgb_8u(rgb_img_file, rgb_img);
        
        vector<SCRF_learning_sample> samples = SCRF_Util::randomSampleFromRgbdImages(rgb_img_file, depth_img_file, pose_file, num_random_sample, i);
        
        all_samples.insert(all_samples.end(), samples.begin(), samples.end());
        rgb_images.push_back(rgb_img);
    }
    
    printf("train image number is %lu, sample number is %lu\n", rgb_images.size(), all_samples.size());    
    
    builder.setTreeParameter(tree_param);
    builder.build_model(model, all_samples, rgb_images, save_model_file);
    
    model.save(save_model_file);
    printf("save model to %s\n", save_model_file);
    
    return 0;
}

#endif
