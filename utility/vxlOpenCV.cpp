//
//  vxlOpenCV.cpp
//  OnlineStereo
//
//  Created by jimmy on 1/26/15.
//  Copyright (c) 2015 Nowhere Planet. All rights reserved.
//

#include "vxlOpenCV.h"
#include "cvxImage_300.h"
//#include "cvxRobustHomoImageUndistortion.h"

using cvx::Rgb8UImage;
using cvx::Bw8UImage;
using namespace::std;
using namespace cv;

void VxlOpenCVImage::houghLines(const vil_image_view<vxl_byte> & image, vcl_vector<vnl_vector_fixed<double, 2> > & lines,
                                double rho, double theta, int threshold, double srn, double stn)
{
    assert(image.nplanes() == 3);
    
    double lowThreshold = 80;
    double ratio        = 2.0  ;
    
    Mat src = VxlOpenCVImage::cv_image(image);
    Mat src_gray;
    Mat detected_edges;
    
    /// Convert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );
    
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3));
    
    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, 3);
    
    // hough line detection
    vector<Vec2f> hough_lines;
    HoughLines(detected_edges, hough_lines, rho, theta, threshold, 0, 0 );
    for (int i = 0; i<hough_lines.size(); i++) {
        lines.push_back(vnl_vector_fixed<double, 2>(hough_lines[i][0], hough_lines[i][1]));
    }
}

void VxlOpenCVImage::houghLinesFromMask(const vil_image_view<vxl_byte> & maskImage, const int mask,
                                        vcl_vector<vnl_vector_fixed<double, 2> > & lines,
                                        double rho, double theta, int threshold, double srn, double stn)
{
    assert(maskImage.nplanes() == 1);
    
    Mat detected_edges = VxlOpenCVImage::cv_image(maskImage);
    vector<Vec2f> hough_lines;
    HoughLines(detected_edges, hough_lines, rho, theta, threshold, 0, 0 );
    for (int i = 0; i<hough_lines.size(); i++) {
        lines.push_back(vnl_vector_fixed<double, 2>(hough_lines[i][0], hough_lines[i][1]));
    }
}


Mat VxlOpenCVImage::cv_image(const vil_image_view<vxl_byte> & image)
{
    assert(image.nplanes() == 3 || image.nplanes() == 1);
    Mat cvImage;
    if (image.nplanes() == 3) {
        cvImage = cv::Mat(image.nj(), image.ni(), CV_8UC3);
        
        Rgb8UImage cv_img(&cvImage);
        for (int j = 0; j<image.nj(); j++) {
            for (int i = 0; i<image.ni(); i++) {
                cv_img[j][i].r = image(i, j, 0);
                cv_img[j][i].g = image(i, j, 1);
                cv_img[j][i].b = image(i, j, 2);
            }
        }
        return cvImage;
    }
    else if(image.nplanes() == 1)
    {
        cvImage = cv::Mat(image.nj(), image.ni(), CV_8UC1);
        
        Bw8UImage cv_img(&cvImage);
        for (int j = 0; j<image.nj(); j++) {
            for (int i = 0; i<image.ni(); i++) {
                cv_img[j][i] = image(i, j, 0);
            }
        }
        return cvImage;
    }
    return cvImage;
}

vil_image_view<vxl_byte> VxlOpenCVImage::to_vil_image_view(const cv::Mat & cvImage)
{
    assert(cvImage.type() == CV_8UC3 || cvImage.type() == CV_8UC1);
    
    vil_image_view<vxl_byte> image;
    if (cvImage.type() == CV_8UC3) {
        image = vil_image_view<vxl_byte>(cvImage.cols, cvImage.rows, 3);
        
        Rgb8UImage cv_img(&cvImage);
        for (int j = 0; j<image.nj(); j++) {
            for (int i = 0; i<image.ni(); i++) {
                image(i, j, 0) = cv_img[j][i].r;
                image(i, j, 1) = cv_img[j][i].g;
                image(i, j, 2) = cv_img[j][i].b;
            }
        }
        return image;
    }
    else if( cvImage.type() == CV_8UC1)
    {
        image = vil_image_view<vxl_byte>(cvImage.cols, cvImage.rows, 1);
        for (int j = 0; j<image.nj(); j++) {
            for (int i = 0; i<image.ni(); i++) {
                image(i, j, 0) = cvImage.at<unsigned char>(j, i);
            }
        }
        return image;
    }
    return image;   
}

void VxlOpenCVImage::imshow(const vil_image_view<vxl_byte> & vil_image, const char *name)
{
    Mat cvImage = VxlOpenCVImage::cv_image(vil_image);
    
    cv::imshow(name, cvImage);
    cv::waitKey(0);
}

/*
vil_image_view<vxl_byte> VxlOpenCVImage::distort_image(const vil_image_view<vxl_byte> & vil_image, double lambda)
{
    assert(vil_image.nplanes() == 3);
    
    Mat cvImage = VxlOpenCVImage::cv_image(vil_image);
    cv::Size pattern_size = cv::Size(14, 10);
	Mat dImage = CvxRobustHomoImageUndistortion::distortImage(cvImage, pattern_size, lambda);
	return VxlOpenCVImage::to_vil_image_view(dImage);
}
 */












