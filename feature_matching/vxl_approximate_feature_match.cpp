//
//  vxl_approximate_feature_match.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2015-10-31.
//  Copyright © 2015 jimmy. All rights reserved.
//

#include "vxl_approximate_feature_match.hpp"
#include "vnl_flann.h"


void vxl_approximate_feature_match::sift_match_by_ratio(const vcl_vector<bapl_keypoint_sptr> & keypointsA,
                                                        const vcl_vector<bapl_keypoint_sptr> & keypointsB,
                                                        vcl_vector<bapl_key_match> & matches,
                                                        vcl_vector<vcl_pair<int, int> > & matchedIndices,
                                                        int search_leaf_num,
                                                        double ratio,
                                                        double feature_distance_threshold)
{
    vcl_vector<vnl_vector<double> > dataset;
    vcl_vector<vnl_vector<double> > querydata;
    for (int i = 0; i<keypointsB.size(); i++) {
        dataset.push_back(keypointsB[i]->descriptor().as_vector());
    }
    
    for (int i = 0; i<keypointsA.size(); i++) {
        querydata.push_back(keypointsA[i]->descriptor().as_vector());
    }
    
    vcl_vector<vcl_vector<int> > indices;
    vcl_vector<vcl_vector<double> > dists;
    int K = 2;
    
    vnl_flann flann;
    flann.set_data(dataset);
    flann.search(querydata, indices, dists, K, search_leaf_num);
    assert(indices.size() == querydata.size());
    
    for (int i = 0; i<indices.size(); i++) {
        int query_idx    = i;
        int searched_idx = indices[i][0];
        double dis1 = dists[i][0];
        double dis2 = dists[i][1];
        if (dis1 < dis2 * ratio && dis1 < feature_distance_threshold) {
            bapl_key_match k_p(keypointsA[query_idx], keypointsB[searched_idx]);
            matches.push_back(k_p);
            matchedIndices.push_back(vcl_pair<int,int>(query_idx, searched_idx));
        }
  //      printf("dis1, dis2 is %f %f\n", dis1, dis2);
    }
    
    printf("find %lu matches from %lu points\n", matches.size(), keypointsA.size());
}

void vxl_approximate_feature_match::asift_match_by_ratio(const vcl_vector< vcl_vector<bapl_keypoint_sptr> > & keypointsA,
                                                         const vcl_vector< vcl_vector<bapl_keypoint_sptr> > & keypointsB,
                                                         vcl_vector<bapl_key_match> & matches,
                                                         int search_leaf_num,
                                                         double ratio,
                                                         double feature_distance_threshold)
{
    
    
}





