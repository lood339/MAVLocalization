//
//  RTAB_kdtree.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-05.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef RTAB_kdtree_cpp
#define RTAB_kdtree_cpp

// static kd tree using flann
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <flann/flann.hpp>
#include <vector>
#include <unordered_map>

using std::vector;

// incremental kdtree support: create, add, remove data
template <class T>
class RTAB_kdtree
{
    flann::Index<flann::L2<T> > kdtree_;   // store kd tree
    int dim_;                              // dimension of data
    
public:
    RTAB_kdtree(const flann::IndexParams& params = flann::KDTreeIndexParams(4))
    :kdtree_(params)
    {
        dim_ = 0;
    }
    ~RTAB_kdtree()
    {
        
    }
    
    // create data and index
    bool create_tree(const cv::Mat & data);
    
    // indices: external index
    void search(const cv::Mat & query_data,
                vector<vector<int> > & indices,
                vector<vector<T> > & dists,
                const int knn) const;
    
private:
    void set_data(const cv::Mat & data);
    
};

// set data and index
template <class T>
bool RTAB_kdtree<T>::create_tree(const cv::Mat & data)
{
    assert(data.type() == CV_32F || data.type() == CV_64F);
    if (data.type() == CV_32F) {
        assert(sizeof(T) == sizeof(float));
    }
    if (data.type() == CV_64F) {
        assert(sizeof(T) == sizeof(double));
    }
    
    flann::Matrix<T> dataset((T *)data.data, data.rows, data.cols);
    kdtree_ = flann::Index<flann::L2<T> > (dataset, flann::KDTreeIndexParams(4));
    kdtree_.buildIndex();
    dim_ = data.cols;
    
    printf("build tree from %d features.\n", data.rows);
    return true;
}

template <class T>
void RTAB_kdtree<T>::search(const cv::Mat & query_data,
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
    
    flann::Matrix<T> query_data_wrap((T *)query_data.data, query_data.rows, query_data.cols);
    kdtree_.knnSearch(query_data_wrap, indices, dists, knn, flann::SearchParams(128));
}




#endif /* RTAB_kdtree_cpp */
