//
//  vpgl_one_point_ransac.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2015-10-26.
//  Copyright © 2015 jimmy. All rights reserved.
//

#ifndef vpgl_one_point_ransac_cpp
#define vpgl_one_point_ransac_cpp

// 1 Point RANSAC structure from motion fro vehicle-mounted cameras
// by exploiting non-holonomic constraint
// ijcv 2011

#include <vcl_vector.h>
#include <vgl/vgl_point_2d.h>
#include <vpgl/vpgl_perspective_camera.h>
#include <vgl/vgl_point_3d.h>


class vpgl_one_point_ransac
{
public:
    // points1, points2: image coordinate
    // resolution: angle in degree
    static bool one_point_ransac(const vpgl_calibration_matrix<double> & K1,
                                 const vpgl_calibration_matrix<double> & K2,
                                 const vcl_vector<vgl_point_2d<double> > & points1,
                                 const vcl_vector<vgl_point_2d<double> > & points2,
                                 double resolution,
                                 vcl_vector<bool> & isInlier);
    
    
private:
    // p1, p2 camera coordinate
    static double angle(const vgl_point_3d<double> & p1, const vgl_point_3d<double> & p2);
    
    
    
    
    
    
    
    
};

#endif /* vpgl_one_point_ransac_cpp */
