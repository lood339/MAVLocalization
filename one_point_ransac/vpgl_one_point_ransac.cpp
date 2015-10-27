//
//  vpgl_one_point_ransac.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2015-10-26.
//  Copyright © 2015 jimmy. All rights reserved.
//

#include "vpgl_one_point_ransac.hpp"
#include <vnl/vnl_math.h>
#include <vnl/vnl_inverse.h>
#include <vnl/vnl_matlab_filewrite.h>
#include "vnl_plus.h"

bool vpgl_one_point_ransac::one_point_ransac(const vpgl_calibration_matrix<double> & K1,
                                             const vpgl_calibration_matrix<double> & K2,
                                             const vcl_vector<vgl_point_2d<double> > & points1,
                                             const vcl_vector<vgl_point_2d<double> > & points2,
                                             double resolution,
                                             vcl_vector<bool> & isInlier)
{
    assert(points1.size() == points2.size());
    
    // inverse intrinsic matrix
    vnl_matrix<double> inv_K1 = vnl_inverse(K1.get_matrix().as_matrix());
    vnl_matrix<double> inv_K2 = vnl_inverse(K2.get_matrix().as_matrix());
    
    vcl_vector<double> angles;
    vnl_vector<double> p_vec(3, 0);
    for (int i = 0; i<points1.size(); i++) {
        // back project to a unit sphere
        vnl_vector<double> p1;
        vnl_vector<double> p2;
        p_vec[0] = points1[i].x();
        p_vec[1] = points1[i].y();
        p_vec[2] = 1.0;
        p1 = inv_K1 * p_vec;
        
        p_vec[0] = points2[i].x();
        p_vec[1] = points2[i].y();
        p_vec[2] = 1.0;
        p2 = inv_K2 * p_vec;
        
        double angle = vpgl_one_point_ransac::angle(vgl_point_3d<double>(p1[0], p1[1], p1[2]),
                                                    vgl_point_3d<double>(p2[0], p2[1], p2[2]));
        angles.push_back(angle);
    }
    
    // median value
    std::nth_element(angles.begin(), angles.begin() + angles.size()/2, angles.end());
    double median =  angles[angles.size()/2];   
    
    
    isInlier.clear();
    for (int i = 0; i<angles.size(); i++) {
        double dif = angles[i] - median;
        if (fabs(dif) < resolution) {
            isInlier.push_back(true);
        }
        else
        {
            isInlier.push_back(false);
        }
    }
    
    // test histogram for the angle
//    void write_mat(const char *file, const vcl_vector<double> & data, const char *dataName = "data");
    VnlPlus::write_mat("angles.mat", angles, "angles");
    return true;
}


double vpgl_one_point_ransac::angle(const vgl_point_3d<double> & p1, const vgl_point_3d<double> & p2)
{
    double x1 = p1.x();
    double y1 = p1.y();
    double z1 = p1.z();
    
    double x2 = p2.x();
    double y2 = p2.y();
    double z2 = p2.z();
    
    double y = y2 * z1 - z2 * y1;
    double x = x2 * z1 + z2 * x1;
    
    double theta = -2.0 * atan2(y, x);  // equation (10) in the paper
    theta = -theta;
    return theta;
}