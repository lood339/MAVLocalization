//
//  vxl_approximate_feature_match.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2015-10-31.
//  Copyright © 2015 jimmy. All rights reserved.
//

#ifndef vxl_approximate_feature_match_cpp
#define vxl_approximate_feature_match_cpp

#include <vil/vil_image_view.h>
#include <vgl/vgl_point_2d.h>
#include <vcl_vector.h>
#include <vpgl/vpgl_perspective_camera.h>
#include <bapl/bapl_lowe_keypoint.h>
#include <bapl/bapl_bbf_tree.h>
#include <bapl/bapl_keypoint_sptr.h>
#include <bapl/bapl_keypoint_set.h>
#include <vcl_utility.h>
#include <vgl/algo/vgl_h_matrix_2d.h>

class vxl_approximate_feature_match
{
public:
    // match from A to B
    // ratio: small --> fewer matches
    static void sift_match_by_ratio(const vcl_vector<bapl_keypoint_sptr> & keypointsA,
                                    const vcl_vector<bapl_keypoint_sptr> & keypointsB,
                                    vcl_vector<bapl_key_match> & matches,
                                    vcl_vector<vcl_pair<int, int> > & matchedIndices,
                                    int search_leaf_num,
                                    double ratio = 0.7,
                                    double feature_distance_threshold = 0.5);
    
    
    
};

#endif /* vxl_approximate_feature_match_cpp */
