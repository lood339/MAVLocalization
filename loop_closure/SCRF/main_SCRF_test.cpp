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
    printf("program   modelFile RGBImageList depthImageList cameraPoseList numSamplePerImage saveFilePrefix\n");
    printf("SCRF_test scrf.txt  rgbs.txt     depth.txt      poses.txt      5000              result/rgb_to_3d \n");
    printf("parameter fits to MS 7 scenes dataset\n");
}


int main(int argc, const char * argv[])
{
    if (argc != 7) {
        printf("argc is %d, should be 7\n", argc);
        help();
        return -1;
    }
    
    const char * model_file = argv[1];
    const char * rgb_image_file = argv[2];
    const char * depth_image_file = argv[3];
    const char * camera_to_wld_pose_file = argv[4];
    const int num_random_sample = (int)strtod(argv[5], NULL);
    const char * prefix = argv[6];
    
    assert(num_random_sample > 100);
    
    //vector<SCRF_learning_sample> all_samples;
    //vector<cv::Mat> rgb_images;
    
    vector<string> rgb_files   = read_file_names(rgb_image_file);
    vector<string> depth_files = read_file_names(depth_image_file);
    vector<string> pose_files  = read_file_names(camera_to_wld_pose_file);
    
    assert(rgb_files.size() == depth_files.size());
    assert(rgb_files.size() == pose_files.size());
    
    vector<int> random_index;
    for (int i = 0; i<rgb_files.size(); i++) {
        random_index.push_back(i);
    }
    std::random_shuffle(random_index.begin(), random_index.end());
    
    // read model
    SCRF_regressor model;
    bool is_read = model.load(model_file);
    if (!is_read) {
        printf("Error: can not read from file %s\n", model_file);
        return -1;
    }
    
    // read images
    cv::Point3d all_test_error(0, 0, 0);
    vector<double> all_prediction_error_distance;
    for (int i = 0; i<random_index.size(); i++) {
        int index = random_index[i];
        const char *cur_rgb_img_file     = rgb_files[index].c_str();
        const char *cur_depth_img_file   = depth_files[index].c_str();
        const char *cur_pose_file        = pose_files[index].c_str();
        
        cv::Mat camera_depth_img;
        cv::Mat rgb_img;
        bool is_read = cvx_io::imread_depth_16bit_to_64f(cur_depth_img_file, camera_depth_img);
        assert(is_read);
        is_read      = cvx_io::imread_rgb_8u(cur_rgb_img_file, rgb_img);
        assert(is_read);
        
        vector<SCRF_learning_sample> samples = SCRF_Util::randomSampleFromRgbdImages(cur_rgb_img_file,
                                                                                     cur_depth_img_file,
                                                                                     cur_pose_file,
                                                                                     num_random_sample,
                                                                                     index);
        
        vector<SCRF_testing_result> predictions;
        for (int j = 0; j<samples.size(); j++) {
            SCRF_testing_result result;
            result.p2d_    = samples[j].p2d_;
            result.gt_p3d_ = samples[j].p3d_;
            bool is_predict = model.predict(samples[j], rgb_img, result);
            if (is_predict) {
                predictions.push_back(result);
            }
        }
        
        // get approximate median error
        cv::Point3d test_error = SCRF_Util::appro_median_error(predictions);
        printf("predicted %lu from %lu samples, ratio is %lf\n", predictions.size(), samples.size(), 1.0*predictions.size()/samples.size());
        cout<<"average testing error is "<<test_error<<endl<<endl;
        
        vector<double> distance = SCRF_Util::prediction_error_distance(predictions);
        all_prediction_error_distance.insert(all_prediction_error_distance.end(), distance.begin(), distance.end());
        std::sort(distance.begin(), distance.end());
        double m_dis = distance[distance.size()/2];
        printf("median error distance is %lf\n", m_dis);
        
        {
            char save_file[1024] = {NULL};
            sprintf(save_file, "%s_%06d.txt", prefix, index);
            FILE *pf = fopen(save_file, "w");
            assert(pf);
            fprintf(pf, "%s\n", cur_rgb_img_file);
            fprintf(pf, "%s\n", cur_depth_img_file);
            fprintf(pf, "%s\n", cur_pose_file);
            fprintf(pf, "image_location__prediction3d__groundTruth3d\n");
            for (int j = 0; j<predictions.size(); j++) {
                cv::Point2d p1 = predictions[j].p2d_;
                cv::Point3d p2 = predictions[j].predict_p3d_;
                cv::Point3d p3 = predictions[j].gt_p3d_;
                fprintf(pf, "%lf %lf\t %lf %lf %lf\t %lf %lf %lf\n", p1.x, p1.y, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z);
            }
            fclose(pf);
            printf("save to %s\n", save_file);
        }
        all_test_error += test_error;
    }
    all_test_error /= (double)rgb_files.size();
    cout<<"all test error are"<<all_test_error<<endl;
    
    //
    std::sort(all_prediction_error_distance.begin(), all_prediction_error_distance.end());
    double median_dis = all_prediction_error_distance[all_prediction_error_distance.size()/2];
    printf("median error distance is %lf\n", median_dis);
    
    return 0;
}

#endif
