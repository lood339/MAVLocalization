//
//  main.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2015-10-25.
//  Copyright © 2015 jimmy. All rights reserved.
//

#if 1

#include <iostream>
#include <time.h>
#include "vil_vlfeat_sift_feature.h"
#include "cvx_LSD.h"
#include "vil_util.hpp"
#include "vxl_head_file.h"
#include "vxl_feature_match.h"
#include "vil_bapl_sift.h"
#include "vil_draw.hpp"
#include "vxlOpenCV.h"
#include "vil_asift_feature.hpp"
#include "cvx_kvld.hpp"
#include "vnl_plus.h"
#include "rrel_plus.hpp"
#include "vgl_fundamental_ransac.hpp"

using std::vector;

static void help()
{
    printf("program pair_file    start_index end_index sample_num save_file             save_folder\n");
    printf("kvld    positive.txt 0           10        5          kvld_positive_H_F.mat result \n");
    
    printf("assume image files are in: \n");
    printf("MAV    file:  /Users/jimmy/Desktop/images/cpsc515/uzh_MAV/images/MAV_Images/left-%06d.jpg, id1 * 100\n");
    printf("Street file:  /Users/jimmy/Desktop/images/cpsc515/uzh_MAV/images/Street_View_Images/left-%03d.jpg, id2\n");
}


int main(int argc, const char * argv[])
{
    if(argc != 7)
    {
        help();
        printf("argc should is %d, should be 7.\n", argc);
        return -1;
    }
    
    const char *pair_file  = argv[1];    // initial mav - street pair indices
    int start_index  = (int)strtod(argv[2], NULL);
    int end_index    = (int)strtod(argv[3], NULL);
    int sample_num   = (int)strtod(argv[4], NULL);
    const char *save_file = argv[5];     // save number of kvld matching, it save to a .mat file
    const char *save_folder = argv[6];   // save images
    assert(start_index >= 0);
    
    // read indices
    std::vector<int> mav_indices;
    std::vector<int> street_indices;
    FILE *pf = fopen(pair_file, "r");
    assert(pf);
    int pair_num = 0;
    fscanf(pf, "%d", &pair_num);
    for (int i = 0; i<pair_num; i++) {
        int id1 = 0;
        int id2 = 0;
        fscanf(pf, "%d %d", &id1, &id2);
        mav_indices.push_back(id1);
        street_indices.push_back(id2);
    }
    fclose(pf);
    assert(mav_indices.size() == street_indices.size());
    printf("read %lu pairs from image id pair file\n", mav_indices.size());
    
    // numbers will be saved
    std::vector<int> kvld_matching_number;
    std::vector<int> kvld_homography_matching_number;
    std::vector<int> kvld_fundamental_mathching_number;
    std::vector<int> mav_ids;
    std::vector<int> street_ids;
    
    // angles for ASIFT
    vector<double> tilts;
    vector<double> rotations;
    tilts.push_back(45.0);
//    tilts.push_back(60.0);
    tilts.push_back(69.3);
    rotations.push_back(-40.0);
//    rotations.push_back(0.0);
    rotations.push_back(40.0);
    
    // SIFT feature parameter
    vl_feat_sift_parameter param;
    param.edge_thresh = 20;
    
    // loop each pair
    for (int k = start_index; k<mav_indices.size() && k<end_index; k += sample_num) {
        int id1 = mav_indices[k];
        int id2 = street_indices[k];
        printf("start index, id1, id2: (%d %d %d) \n", k, id1, id2);
        
        char mav_file[1024]    = {NULL};
        char street_file[1024] = {NULL};
        sprintf(mav_file, "/Users/jimmy/Desktop/images/cpsc515/uzh_MAV/images/MAV_Images/left-%06d.jpg", id1 * 100);
        sprintf(street_file, "/Users/jimmy/Desktop/images/cpsc515/uzh_MAV/images/Street_View_Images/left-%03d.jpg", id2);
        
        // count the number of kvld matching
        vil_image_view<vxl_byte> image1 = vil_load(mav_file);
        vil_image_view<vxl_byte> image2 = vil_load(street_file);
        
        // extract ASIFT feature
        double tt = clock();
        vcl_vector<vcl_vector<bapl_keypoint_sptr> >  keypoints_1;
        vcl_vector<vcl_vector<bapl_keypoint_sptr> >  keypoints_2;
        vil_asift_feature::asift_keypoints_extractor(image1, rotations, tilts, param, keypoints_1, true);
        vil_asift_feature::asift_keypoints_extractor(image2, rotations, tilts, param, keypoints_2, true);
        printf("extract ASIFT cost time: %f\n", (clock() - tt)/CLOCKS_PER_SEC);
        
        int num = 0;
        int H_num = 0;
        int F_num = 0;
        
        std::vector<vgl_point_2d<double> > kvld_inlier_pts1;
        std::vector<vgl_point_2d<double> > kvld_inlier_pts2;
        // loop all ASIFT pairs
        for (int i = 0; i<keypoints_1.size(); i++) {
            for (int j = 0; j<keypoints_2.size(); j++) {
                // initial match asift
                vcl_vector<bapl_key_match> matches;
                vcl_vector<vcl_pair<int, int> > matchedIndices;
                VxlFeatureMatch::siftMatchByRatio(keypoints_1[i], keypoints_2[j], matches, matchedIndices, 0.6, 0.5);
                
                // initial matching number too small
                if (matches.size() < 3) {
                    continue;
                }
                
                // kvld match
                std::vector<bool> is_valid;
                cvx_kvld_parameter kvld_param;
                kvld_param.matches = matchedIndices;
                kvld_param.keypoint_1 = keypoints_1[i];
                kvld_param.keypoint_2 = keypoints_2[j];
                vcl_vector<bapl_key_match> kvld_matches;
                bool isOk = cvx_kvld::kvld_matching(image1, image2, kvld_matches, is_valid, kvld_param);
                num += kvld_matches.size();
                if (isOk) {
                    std::vector<vgl_point_2d<double> > inlier_pts_1;
                    std::vector<vgl_point_2d<double> > inlier_pts_2;
                    VilBaplSIFT::getMatchingLocations(kvld_matches, inlier_pts_1, inlier_pts_2);
                    
                    kvld_inlier_pts1.insert(kvld_inlier_pts1.end(), inlier_pts_1.begin(), inlier_pts_1.end());
                    kvld_inlier_pts2.insert(kvld_inlier_pts2.end(), inlier_pts_2.begin(), inlier_pts_2.end());
                }
            }
        }
        printf("mav image: %d, street image: %d  kvld number is %d\n", id1, id2, num);
        kvld_matching_number.push_back(num);
        mav_ids.push_back(id1);
        street_ids.push_back(id2);
        
        {
            // save iamges for visual comparison
            char match_save_file[1024] = {NULL};
            sprintf(match_save_file, "%s/mav_%d_street_%d_kvld_%d.jpg", save_folder, id1, id2, num);
            vil_image_view<vxl_byte> matches;
            VilDraw::draw_match_vertical(image1, image2, kvld_inlier_pts1, kvld_inlier_pts2, matches);
            VilUtil::vil_save(matches, match_save_file);
        }
        
        // homography constraint
        if (num > 5) {
            homography_ransac_parameter homog_param;
            homog_param.max_outlier_frac = 0.5;
            homog_param.error_tolerance  = 4.0;
            std::vector<bool> inlier;
            vgl_h_matrix_2d< double > H;
            bool isH = RrelPlus::homography_RANSAC(kvld_inlier_pts1, kvld_inlier_pts2, inlier, H, homog_param);
            if (isH) {
                std::cout<<"homography is cacluated. "<<std::endl;
                vector<vgl_point_2d<double>> kvld_H_inlier1;
                vector<vgl_point_2d<double>> kvld_H_inlier2;
                for (int i = 0; i<inlier.size(); i++) {
                    if (inlier[i]) {
                        H_num++;
                        kvld_H_inlier1.push_back(kvld_inlier_pts1[i]);
                        kvld_H_inlier2.push_back(kvld_inlier_pts2[i]);
                    }
                }
                // save match image
                char match_save_file[1024] = {NULL};
                sprintf(match_save_file, "%s/mav_%d_street_%d_kvld_H_%d.jpg", save_folder, id1, id2, H_num);
                vil_image_view<vxl_byte> matches;
                VilDraw::draw_match_vertical(image1, image2, kvld_H_inlier1, kvld_H_inlier2, matches);
                VilUtil::vil_save(matches, match_save_file);
            }
        }
        kvld_homography_matching_number.push_back(H_num);
        
        // fundamental constraint
        if (num > 8) {
            vnl_matrix_fixed< double, 3, 3 > F;
            std::vector<bool> inlier;
            fundamental_ransac_parameter fundamental_param;
            fundamental_param.max_epipolar_distance = 3.0;
            fundamental_param.confidence_prob = 0.99;
            bool isF = vgl_fundamental_ransac_opencv(kvld_inlier_pts1, kvld_inlier_pts2, F, inlier, fundamental_param);
            if (isF) {
                vector<vgl_point_2d<double>> kvld_F_inlier1;
                vector<vgl_point_2d<double>> kvld_F_inlier2;
                for (int i = 0; i<inlier.size(); i++) {
                    if (inlier[i]) {
                        F_num++;
                        kvld_F_inlier1.push_back(kvld_inlier_pts1[i]);
                        kvld_F_inlier2.push_back(kvld_inlier_pts2[i]);
                    }
                }
                
                // save match image
                char match_save_file[1024] = {NULL};
                sprintf(match_save_file, "%s/mav_%d_street_%d_kvld_F_%d.jpg", save_folder, id1, id2, F_num);
                vil_image_view<vxl_byte> matches;
                VilDraw::draw_match_vertical(image1, image2, kvld_F_inlier1, kvld_F_inlier2, matches);
                VilUtil::vil_save(matches, match_save_file);
            }
        }
        kvld_fundamental_mathching_number.push_back(F_num);
        printf("init number, H number, F number are %d %d %d\n", num, H_num, F_num);
        
        
        // save result after one iteration
        {
            vnl_matlab_filewrite writer(save_file);
            writer.write(VnlPlus::vector_2_vec(kvld_matching_number), "kvld_matching");
            writer.write(VnlPlus::vector_2_vec(kvld_homography_matching_number), "homography_matching");
            writer.write(VnlPlus::vector_2_vec(kvld_fundamental_mathching_number), "fundamental_matching");
            writer.write(VnlPlus::vector_2_vec(mav_ids), "mav_id");
            writer.write(VnlPlus::vector_2_vec(street_ids), "street_id");
            printf("save to %s\n", save_file);
        }
    }
    
    // save final result
    vnl_matlab_filewrite writer(save_file);
    writer.write(VnlPlus::vector_2_vec(kvld_matching_number), "kvld_matching");
    writer.write(VnlPlus::vector_2_vec(kvld_homography_matching_number), "homography_matching");
    writer.write(VnlPlus::vector_2_vec(kvld_fundamental_mathching_number), "fundamental_matching");
    writer.write(VnlPlus::vector_2_vec(mav_ids), "mav_id");
    writer.write(VnlPlus::vector_2_vec(street_ids), "street_id");
    printf("save to %s\n", save_file);
    return 0;
}

#endif