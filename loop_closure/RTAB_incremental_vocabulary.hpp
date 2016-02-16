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
#include <unordered_map>

using std::vector;
using std::unordered_map;

// incremental vocabulary
template <class T>
class RTAB_incremental_vocabulary
{
    flann::Index<flann::L2<T> > kdtree_;   // store kd tree
    float rebuild_threshold_;
    int dim_;
    
    // key: index inside kd-tree, value: outside global index
    std::unordered_map<unsigned long, unsigned long> index_map_;
 //   unsigned long cur_max_index_;                    // current number of max index
public:
    
    RTAB_incremental_vocabulary(const flann::IndexParams& params = flann::KDTreeIndexParams(4), const float rebuild_threshold = 1.1)
                        :kdtree_(params),
                         rebuild_threshold_(rebuild_threshold)
    {
        dim_ = 0;
        
    }
    ~RTAB_incremental_vocabulary(){}
    
    
    
    // set data and index
    bool set_data(const cv::Mat & data, const vector<unsigned long> & data_index);
    
    // incrementally add data, rebuild tree
    bool add_data(const cv::Mat & data, const vector<unsigned long> & data_index);
    // remove data by index, rebuild tree
    bool remove_data(const vector<unsigned long> & data_index);
    
    void search(const cv::Mat & query_data,
                vector<vector<int> > & indices,
                vector<vector<T> > & dists, const int knn) const;
    
private:
    void set_data(const cv::Mat & data);
    void add_data(const cv::Mat & data);
    void add_index(const vector<unsigned long> & index);
    
};

// implemenation

// set data and index
template <class T>
bool RTAB_incremental_vocabulary<T>::set_data(const cv::Mat & data, const vector<unsigned long> & data_index)
{
    assert(data.rows == data_index.size());
    
    this->set_data(data);
    this->add_index(data_index);
    return true;
}

// incrementally add data, rebuild tree
template <class T>
bool RTAB_incremental_vocabulary<T>::add_data(const cv::Mat & data, const vector<unsigned long> & data_index)
{
    assert(data.rows == data_index.size());
    
    this->add_data(data);
    this->add_index(data_index);
    
    return true;
}

// remove data by index, rebuild tree
template <class T>
bool RTAB_incremental_vocabulary<T>::remove_data(const vector<unsigned long> & data_index)
{
    return true;
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
    kdtree_.knnSearch(query_data_wrap, indices, dists, knn, flann::SearchParams(128));
}


// private
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
    kdtree_ = flann::Index<flann::L2<T> > (dataset, flann::KDTreeIndexParams(4));
    kdtree_.buildIndex();
    dim_ = data.cols;
    
    printf("build tree from %d features.\n", data.rows);
}

template <class T>
void RTAB_incremental_vocabulary<T>::add_data(const cv::Mat & data)
{
    assert(data.type() == CV_32F || data.type() == CV_64F);
    if (data.type() == CV_32F) {
        assert(sizeof(T) == sizeof(float));
    }
    if (data.type() == CV_64F) {
        assert(sizeof(T) == sizeof(double));
    }
    
    flann::Matrix<T> dataset((T *)data.data, data.rows, data.cols);
    kdtree_.addPoints(dataset, rebuild_threshold_);
    
}

template <class T>
void RTAB_incremental_vocabulary<T>::add_index(const vector<unsigned long> & index)
{
    unsigned long cur_max_index = index_map_.size();
    for (int i = 0; i<index.size(); i++) {
        index_map_[cur_max_index + i] = index[i];
    }
}


typedef RTAB_incremental_vocabulary<float>  vocabulary32;
typedef RTAB_incremental_vocabulary<double> vocabulary64;



#endif /* RTAB_incremental_vocabulary_cpp */
