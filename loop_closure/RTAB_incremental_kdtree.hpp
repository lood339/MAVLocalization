//
//  RTAB_incremental_kdtree.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-02-16.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef RTAB_incremental_kdtree_cpp
#define RTAB_incremental_kdtree_cpp

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <flann/flann.hpp>
#include <vector>
#include <unordered_map>

using std::vector;
using std::unordered_map;

// incremental kdtree support: create, add, remove data
template <class T>
class RTAB_incremental_kdtree
{
    flann::Index<flann::L2<T> > kdtree_;   // store kd tree
    float rebuild_threshold_;              // > 1.0, the smaller, the more frequent rebuild
    int dim_;                              // dimension of data
    
    // key: internal index in kd-tree, value: external in global
    std::unordered_map<unsigned long, unsigned long> index_map_;
    // size() == size of the kdtree
    std::unordered_map<unsigned long, unsigned long> inverse_index_map_;
    
public:
    
    RTAB_incremental_kdtree(const flann::IndexParams& params = flann::KDTreeIndexParams(4),
                            const float rebuild_threshold = 1.1)
                            :kdtree_(params),
                            rebuild_threshold_(rebuild_threshold)
    {
        assert(rebuild_threshold > 1.0);
        dim_ = 0;
        
    }
    RTAB_incremental_kdtree()
    {
        
    }
    
    // create data and index
    bool create_tree(const cv::Mat & data, const vector<unsigned long> & data_index);
    
    // call crate_tree before use this function
    // incrementally add data, rebuild tree
    bool add_data(const cv::Mat & data, const vector<unsigned long> & data_index);
    
    // remove data by index, rebuild tree
    bool remove_data(const vector<unsigned long> & data_index);
    
    // indices: external index
    void search(const cv::Mat & query_data,
                vector<vector<int> > & indices,
                vector<vector<T> > & dists,
                const int knn) const;
    
    // internal state of the dynamic kdtree
    void print_state(void);
    
private:
    void set_data(const cv::Mat & data);
    void add_data(const cv::Mat & data);
    void add_index(const vector<unsigned long> & index);
    
};

// implemenation

// set data and index
template <class T>
bool RTAB_incremental_kdtree<T>::create_tree(const cv::Mat & data, const vector<unsigned long> & data_index)
{
    assert(data.rows == data_index.size());
    
    this->set_data(data);
    this->add_index(data_index);
    assert(inverse_index_map_.size() == kdtree_.size());
    return true;
}

// incrementally add data, rebuild tree
template <class T>
bool RTAB_incremental_kdtree<T>::add_data(const cv::Mat & data, const vector<unsigned long> & data_index)
{
    assert(data.rows == data_index.size());
    
    this->add_data(data);
    this->add_index(data_index);
    assert(inverse_index_map_.size() == kdtree_.size());
    return true;
}

// remove data by index, rebuild tree
template <class T>
bool RTAB_incremental_kdtree<T>::remove_data(const vector<unsigned long> & data_index)
{
    for (int i = 0; i < data_index.size(); i++) {
        unsigned long external_index = data_index[i];
        if (inverse_index_map_.find(external_index) != inverse_index_map_.end()) {
            kdtree_.removePoint(inverse_index_map_[external_index]);
            inverse_index_map_.erase(external_index);
        }
        else
        {
            printf("Error: kd tree remove points can not find internal index!\n");
            assert(0);
        }
    }
    assert(inverse_index_map_.size() == kdtree_.size());
    return true;
}


template <class T>
void RTAB_incremental_kdtree<T>::search(const cv::Mat & query_data,
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
    
    // transfer internal index to external index
    for (int i = 0; i<indices.size(); i++) {
        for (int j = 0; j<indices[i].size(); j++) {
            unsigned long index = indices[i][j];
            std::unordered_map<unsigned long, unsigned long>::const_iterator ite = index_map_.find(index);
            if (ite != index_map_.end()) {
                indices[i][j] = ite->second;
            }
            else
            {
                printf("Error: search can not find internal index!\n");
                assert(0);
            }
        }
    }
}

template <class T>
void RTAB_incremental_kdtree<T>::print_state(void)
{
    printf("Dynamic kdtree state begin ------------\n");
    printf("kd-tree  %lu number of features\n", kdtree_.size());
    printf("[internal external] table size: %lu\n", index_map_.size());
    printf("[external internal] table size: %lu\n", inverse_index_map_.size());
    printf("Dynamic kdtree state end   ------------\n");
    
}


// private functions
template <class T>
void RTAB_incremental_kdtree<T>::set_data(const cv::Mat & data)
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
void RTAB_incremental_kdtree<T>::add_data(const cv::Mat & data)
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
void RTAB_incremental_kdtree<T>::add_index(const vector<unsigned long> & index)
{
    unsigned long cur_max_index = index_map_.size();
    for (int i = 0; i<index.size(); i++) {
        index_map_[cur_max_index + i] = index[i];
        inverse_index_map_[index[i]] = cur_max_index + i;
    }
}


//typedef RTAB_incremental_kdtree<float>  dynamic_kdtree32;
//typedef RTAB_incremental_kdtree<double> dynamic_kdtree64;



#endif /* RTAB_incremental_kdtree_cpp */
