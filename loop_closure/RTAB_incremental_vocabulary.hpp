//
//  RTAB_incremental_vocabulary.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-02-13.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef RTAB_incremental_vocabulary_cpp
#define RTAB_incremental_vocabulary_cpp

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <flann/flann.hpp>
#include <vector>

using std::vector;

// incremental vocabulary
template <class T>
class RTAB_incremental_vocabulary
{
    flann::Index<flann::L2<T> > index_;   // store kd tree
    int dim_;
    
public:
    
    RTAB_incremental_vocabulary(const flann::IndexParams& params = flann::KDTreeIndexParams(4)):index_(params)
    {
        dim_ = 0;
    }
    ~RTAB_incremental_vocabulary(){}
    
    void set_data(const cv::Mat & data);
    
    void search(const cv::Mat & query_data,
                vector<vector<int> > & indices,
                vector<vector<T> > & dists, const int knn) const;
    
    
};


// implemenation
template <class T>
void RTAB_incremental_vocabulary<T>::set_data(const cv::Mat & data)
{
    assert(data.type() == CV_32F || data.type() == CV_64F);
    if (data.type() == CV_32F) {
        assert(sizeof(T) == sizeof(float));
    }
    if (data.type() == CV_64F) {
        assert(sizeof(T) == sizeof(double));
    }
    
    flann::Matrix<T> dataset((T *)data.data, data.rows, data.cols);
    index_ = flann::Index<flann::L2<T> > (dataset, flann::KDTreeIndexParams(4));
    index_.buildIndex();
    dim_ = data.cols;
    
    printf("build tree from %d features.\n", data.rows);
}

template <class T>
void RTAB_incremental_vocabulary<T>::search(const cv::Mat & query_data,
                                            vector<vector<int> > & indices,
                                            vector<vector<T> > & dists,
                                            const int knn) const
{
    assert(query_data.type() == CV_32F || query_data.type() == CV_64F);
    if (query_data.type() == CV_32F) {
        assert(sizeof(T) == sizeof(float));
    }
    if (query_data.type() == CV_64F) {
        assert(sizeof(T) == sizeof(double));
    }

    
    flann::Matrix<T> query_data_wrap((T *)query_data, query_data.rows, query_data.cols);
    index_.knnSearch(query_data_wrap, indices, dists, knn, flann::SearchParams(128));
}


typedef RTAB_incremental_vocabulary<float> vocabulary32;
typedef RTAB_incremental_vocabulary<double> vocabulary64;



#endif /* RTAB_incremental_vocabulary_cpp */
