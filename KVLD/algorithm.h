/** @basic structures implementation
 ** @author Zhe Liu
 **/

/*
Copyright (C) 2011-12 Zhe Liu and Pierre Moulon.
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).
*/
#ifndef KVLD_ALGORITHM_H
#define KVLD_ALGORITHM_H

#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <sstream>
#include <numeric>
#include <memory>
#include <algorithm>
#include <functional>

#include <opencv\cv.hpp>

#ifdef _DEBUG
	#pragma comment(lib, "opencv_ts300d.lib")		
	#pragma comment(lib, "opencv_world300d.lib")
#else
	#pragma comment(lib, "opencv_ts300.lib")
	#pragma comment(lib, "opencv_world300.lib")
#endif

#ifndef OriSize
#define OriSize  8
#endif
#ifndef IndexSize
#define IndexSize  4
#endif
#ifndef VecLength
#define VecLength  IndexSize * IndexSize * OriSize
#endif


struct VLDKeyPoint
{
	cv::KeyPoint cvKeyPoint;
	float vec[VecLength];

	std::ifstream& readDetector(std::ifstream& in){
		float x, y;
		in >> x >> y >> cvKeyPoint.size >> cvKeyPoint.angle;
		cvKeyPoint.pt = cv::Point2f(x, y);
		//for(int i=0;i<128;i++)  {
		//  in>>point.vec[i];
		//}
		return in;
	}

	std::ofstream& writeDetector(std::ofstream& out) const {
		out << cvKeyPoint.pt.x << " " << cvKeyPoint.pt.y << " " << cvKeyPoint.size << " " << cvKeyPoint.angle << std::endl;
		/*for(int i=0;i<128;i++)
		out<<feature.vec[i]<<" ";
		out<<std::endl;*/
		return out;
	}
}; 


typedef cv::Mat Matrix;
typedef cv::Mat Matrixf;
const float PI = 4.0 * atan(1.0f);


//===================================== intergral image ====================================//
//It is used to efficiently constructe the pyramide of scale images in KVLD
struct IntegralImages{
	cv::Mat map;

	IntegralImages(const cv::Mat& I);
  
  inline double operator()(double x1, double y1,double x2,double y2)const{
		return get(x2,y2)-get(x1,y2)-get(x2,y1)+get(x1,y1);
	}

  inline double operator()(double x, double y, double size) const{
    double window=0.5*size;
    return (get(x+window,y+window)-get(x-window,y+window)-get(x+window,y-window)+get(x-window,y-window))/(4*window*window);
  }

private :
  inline double get(double x, double y)const{
		int ix=int(x), iy=int(y);
		double dx=x-ix, dy=y-iy;
		if (dx==0 && dy==0)
			return map.at<double>(iy,ix);
		if (dx==0)
			return map.at<double>(iy, ix)*(1 - dy) + map.at<double>(iy + 1, ix)*dy;
		if (dy==0)
			return map.at<double>(iy, ix)*(1 - dx) + map.at<double>(iy, ix + 1)*dx;

		return map.at<double>(iy, ix)*(1 - dx)*(1 - dy) +
			map.at<double>(iy + 1, ix)*dy*(1 - dx) +
			map.at<double>(iy, ix + 1)*(1 - dy)*dx +
			map.at<double>(iy + 1, ix + 1)*dx*dy;
	}
};


//======================================elemetuary operations================================//
template <typename T>
inline T point_distance(const T x1, const T y1, const T x2, const T y2){//distance of points
	float a=x1-x2, b=y1-y2;
	return sqrt(a*a+b*b);
}

template <typename T>
inline float point_distance(const T& P1,const T& P2){//distance of points
	return point_distance<float>(P1.cvKeyPoint.pt.x, P1.cvKeyPoint.pt.y, P2.cvKeyPoint.pt.x, P2.cvKeyPoint.pt.y);
}

inline bool inside(int w, int h, int x,int y,double radios){
	return (x-radios>=0 && y-radios>=0 && x+radios<w && y+radios<h);
}

inline bool anglefrom(const float& x, const float& y, float& angle){
	if (x!=0)
		angle=atan(y/x);
	else if (y>0)
		angle=PI/2;
	else if (y<0)
		angle=-PI/2;
	else return false;

	if (x<0)
		angle+=PI;
	while(angle<0)
		angle+=2*PI;
  while (angle>=2*PI)
		angle-=2*PI;
	assert(angle>=0 && angle<2*PI);
	return true;
}

inline double angle_difference(const double angle1, const double angle2){
	double angle=angle1-angle2;
	while (angle<0) angle+=2*PI;
	while (angle>=2*PI)	angle-=2*PI;

	assert(angle<=2*PI && angle>=0);
	return std::min(angle,2*PI-angle);
}

inline void max(double* list,double& weight, int size, int& index,int& second_index){
	index=0;
	second_index=-1;
	double best=list[index]-list[index+size/2];

	for (int i=0;i<size;i++){
			double value;
			if(i<size/2) value=list[i]-list[i+size/2];
			else value=list[i]-list[i-size/2];

			if (value>best){
				best=value;
				second_index=index;
				index=i;
			}
	}
	weight=best;
}

template<typename ARRAY>
inline void normalize_weight(ARRAY & weight){
  double total= std::accumulate(weight.begin(), weight.end(), 0.0);
	if (!total==0)
  	for (int i=0; i<weight.size();i++)
	  	weight[i]/=total;
}

template<typename T>
inline float consistent(const T& a1,const T& a2,const T& b1,const T& b2){
	float ax = float(a1.cvKeyPoint.pt.x - a2.cvKeyPoint.pt.x);
	float ay = float(a1.cvKeyPoint.pt.y - a2.cvKeyPoint.pt.y);
	float bx = float(b1.cvKeyPoint.pt.x - b2.cvKeyPoint.pt.x);
	float by = float(b1.cvKeyPoint.pt.y - b2.cvKeyPoint.pt.y);

	float angle1 = float(b1.cvKeyPoint.angle - a1.cvKeyPoint.angle);
	float angle2 = float(b2.cvKeyPoint.angle - a2.cvKeyPoint.angle);

	float ax1=cos(angle1)*ax-sin(angle1)*ay;
	ax1 *= float(b1.cvKeyPoint.size / a1.cvKeyPoint.size);
	float ay1=sin(angle1)*ax+cos(angle1)*ay;
	ay1 *= float(b1.cvKeyPoint.size / a1.cvKeyPoint.size);
	float d1=sqrt(ax1*ax1+ay1*ay1);
	float d1_error=sqrt((ax1-bx)*(ax1-bx)+(ay1-by)*(ay1-by));

	float ax2=float(cos(angle2)*ax-sin(angle2)*ay);
	ax2 *= float(b2.cvKeyPoint.size / a2.cvKeyPoint.size);
	float ay2=float(sin(angle2)*ax+cos(angle2)*ay);
	ay2 *= float(b2.cvKeyPoint.size / a2.cvKeyPoint.size);
	float d2=sqrt(ax2*ax2+ay2*ay2);
	float d2_error=sqrt((ax2-bx)*(ax2-bx)+(ay2-by)*(ay2-by));

	float d=std::min(d1_error/std::min(d1,point_distance(b1,b2)),d2_error/std::min(d2,point_distance(b1,b2)));
	return d;
}

float getRange(const cv::Mat& I,int a,const float p,const float ratio);


int read_detectors(const std::string& filename, std::vector<VLDKeyPoint>& feat);//reading openCV style detectors

int read_matches(const std::string& filename, std::vector<cv::DMatch>& matches);//reading openCV style matches

int Convert_image(const cv::Mat& In, cv::Mat& imag);

#endif //KVLD_ALGORITHM_H