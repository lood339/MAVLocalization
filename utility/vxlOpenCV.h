//
//  vxlOpenCV.h
//  OnlineStereo
//
//  Created by jimmy on 1/26/15.
//  Copyright (c) 2015 Nowhere Planet. All rights reserved.
//

#ifndef __OnlineStereo__vxlOpenCV__
#define __OnlineStereo__vxlOpenCV__

// interface for algorithm in OpenCV to vxl
#include <vil/vil_image_view.h>
#include <vcl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include "cvxImage_300.h"

class VxlOpenCVImage
{
    public:
   // HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0 )
    // 1, CV_PI/180, 100
    
    // rho – Distance resolution of the accumulator in pixels.  0.5
    // theta – Angle resolution of the accumulator in radians.  vnl_math::pi/360.0
    // threshold – Accumulator threshold parameter. 100
    static void houghLines(const vil_image_view<vxl_byte> & image, vcl_vector<vnl_vector_fixed<double, 2> > & lines,
                           double rho, double theta, int threshold, double srn=0, double stn=0);
    
    //
    static void houghLinesFromMask(const vil_image_view<vxl_byte> & maskImage, const int mask, vcl_vector<vnl_vector_fixed<double, 2> > & lines,
                                   double rho, double theta, int threshold, double srn=0, double stn=0);
    
    
    
    // assume rgb image
    static Mat cv_image(const vil_image_view<vxl_byte> & vilImage);
    static vil_image_view<vxl_byte> to_vil_image_view(const cv::Mat & cvImage);
    
    // show Image
    static void imshow(const vil_image_view<vxl_byte> & vil_image, const char *name = "temp");
    
    // distort image
//    static vil_image_view<vxl_byte> distort_image(const vil_image_view<vxl_byte> & image, double lambda);
    
};

#endif /* defined(__OnlineStereo__vxlOpenCV__) */
