/** @basic structures implementation
 ** @author Zhe Liu
 **/

/*
Copyright (C) 2011-12 Zhe Liu and Pierre Moulon.
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "algorithm.h"


IntegralImages::IntegralImages(const cv::Mat& I){
	map = cv::Mat(I.rows + 1, I.cols + 1, CV_64FC1);
		map = cv::Mat::zeros(map.rows, map.cols, map.type());
		for (int y=0;y<I.rows;y++)
			for (int x=0;x<I.cols;x++){
				map.at<double>(y + 1, x + 1) = double(I.at<float>(y, x)) + map.at<double>(y, x + 1) + map.at<double>(y + 1, x) - map.at<double>(y, x);
			}
	}

float getRange(const cv::Mat& I,int a,const float p, const float ratio){
  float range=ratio*sqrt(float(3*I.rows*I.cols)/(p*a*PI));
  std::cout<<"range ="<<range<<std::endl;
  return range;
}


//=============================IO interface, convertion of object types======================//
int read_detectors(const std::string& filename, std::vector<VLDKeyPoint>& feat)
{
	std::ifstream file(filename.c_str());
	if (!file.is_open()){
		std::cout << "error in reading detector files" << std::endl;
		return -1;
	}

	int size;
	file >> size;
	for (int i = 0; i<size; i++){
		feat.resize(feat.size() + 1);
		feat.back().readDetector(file);
	}

	file.close();
	return 0;
}

int read_matches(const std::string& filename, std::vector<cv::DMatch>& matches)
{
	std::ifstream file(filename.c_str());
	if (!file.is_open()){
		std::cout << "error in reading matches files" << std::endl;
		return -1;
	}

	int size;
	file >> size;
	for (int i = 0; i<size; i++){
		int  l, r;
		file >> l >> r;
		cv::DMatch m(l, r, 0);
		matches.push_back(m);
	}

	file.close();
	return 0;
}

int Convert_image(const cv::Mat& In, cv::Mat& imag)//convert only gray scale image of opencv
{
	imag = cv::Mat(In.rows, In.cols, CV_32FC1);
	int cn = In.channels();
	if (cn == 1)//gray scale
	{
		for (int i = 0; i < In.rows; ++i)
		{
			for (int j = 0; j < In.cols; ++j)
			{
				imag.at<float>(i, j) = In.at<unsigned char>(i, j);
			}
		}
	}
	else
	{
		for (int i = 0; i < In.rows; ++i)
		{
			for (int j = 0; j < In.cols; ++j)
			{
				//imag.at<float>(i, j) = (float(pixelPtr[(i*In.cols + j)*cn + 0]) * 29 + float(pixelPtr[(i*In.cols + j)*cn + 1]) * 150 + float(pixelPtr[(i*In.cols + j)*cn + 2]) * 77) / 255;
				//why not using uniform weights of the channels?
			}
		}
	}
	return 0;
}