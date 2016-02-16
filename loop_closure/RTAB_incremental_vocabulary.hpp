//
//  RTAB_incremental_vocabulary.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy & Lili on 2016-02-13.
//  Copyright © 2016 jimmy & Lili. All rights reserved.
//

#ifndef RTAB_incremental_vocabulary_cpp
#define RTAB_incremental_vocabulary_cpp

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <flann/flann.hpp>


#include <cassert>
#include <map>
#include <vector>

#define CV24 1

#if CV24
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp> // for homography
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/flann/flann_base.hpp"
#include "opencv2/flann/flann.hpp"
#endif


using namespace std;

struct RankedScore {
    int imageIndex;
    float imageScore;
};

struct by_number {
    bool operator()(RankedScore const &left, RankedScore const &right) {
        return left.imageScore > right.imageScore;
    }
};

// incremental vocabulary
template <class T>
class RTAB_incremental_vocabulary
{
    //flann::Index<flann::L2<T> > index_;   // store kd tree
    int dim_;

public:

    RTAB_incremental_vocabulary(int imgNum)
    {
         nLabel_=imgNum;
    }

    ~RTAB_incremental_vocabulary(){}

    void set_data(const cv::Mat & data);

    static cv::flann::IndexParams * createFlannIndexParams(int index);

    cvflann::flann_distance_t getFlannDistanceType();

    int clustering(const cv::Mat& features, cv::Mat& centers);

    void create_and_update_vocab();

    void search_descriptors(const cv::Mat query_descriptors,
                                  cv::Mat & indices,
                                  cv::Mat & dists, int knn);

    void search_image(const cv::Mat & descriptors,
                            cv::Mat & results,
                            cv::Mat & dists,
                            int k,
                            int imgNum,
                            vector<unsigned int> labels,
                            vector<int> images,
                            multimap<int, int> &imageScore,
                            vector<RankedScore> &rankedScore);

    void computeLikelihood(int imgNum, vector<RankedScore> rankedScore, vector<double> likelihood);

    int nLabel_;

    cv::flann::Index flannIndex_;
    cv::Mat indexedDescriptors_;
    cv::Mat notIndexedDescriptors_;
    vector<int> notIndexedWordIds_;

};

/*
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
*/
/*
template <class T>
void RTAB_incremental_vocabulary<T>::search_descriptors(const cv::Mat & query_data,
                                                        cv::Mat & indices,
                                                        cv::Mat & dists,
                                                        const int knn
                                                        ) const
{
    assert(query_data.type() == CV_32F || query_data.type() == CV_64F);
    if (query_data.type() == CV_32F)
    {
        assert(sizeof(T) == sizeof(float));
    }
    if (query_data.type() == CV_64F)
    {
        assert(sizeof(T) == sizeof(double));
    }

    flann::Matrix<T> query_data_wrap((T *)query_data, query_data.rows, query_data.cols);  ///？？？
    index_.knnSearch(query, indices, dists, knn, flann::SearchParams(128));

}*/



template <class T>
cvflann::flann_distance_t RTAB_incremental_vocabulary<T>::getFlannDistanceType()
{
	cvflann::flann_distance_t distance = cvflann::FLANN_DIST_L2;

	return distance;
}

template <class T>
cv::flann::IndexParams * RTAB_incremental_vocabulary<T>::createFlannIndexParams(int index)
{
	cv::flann::IndexParams * params = 0;

    switch(index)
    {
        case 0:
            params = new cv::flann::LinearIndexParams();
            cout<<"flann linearIndexParams"<<endl;
            break;
        case 1:
            params = new cv::flann::KDTreeIndexParams();
            cout<<"flann randomized kdtrees params"<<endl;
            break;
        case 2:
            params = new cv::flann::KMeansIndexParams();
            cout<<"flann hierarchical k-means tree params"<<endl;
            break;
        case 3:
            params = new cv::flann::CompositeIndexParams();
            cout<<"flann combination of randomized kd-trees and hierarchical k-means tree"<<endl;
            break;
        /*case 4:
            params = new cv::flann::LshIndexParams();
            cout<<"flann LSH Index params"<<endl;
            break;*/
        case 5:
            params = new cv::flann::AutotunedIndexParams();
            cout<<"flann AutotunedIndexParams"<<endl;
            break;
        default:
            break;
    }

	if(!params)
	{
		printf("ERROR: NN strategy not found !? Using default KDTRee...\n");
		params = new cv::flann::KDTreeIndexParams();
	}
	return params ;
}

template <class T>
int RTAB_incremental_vocabulary<T>::clustering(const cv::Mat& features, cv::Mat& centers)
{

    assert(features.type()==CV_32F);
    assert(centers.type()==CV_32F);
    int number_of_clusters;
    assert(features.isContinuous());

    cvflann::Matrix<float> featuresFlann((float*)features.data, features.rows, features.cols);
    cvflann::Matrix<float> centersFlann((float*)centers.data, centers.rows, centers.cols);
    int branching=10;
    int iterations=15;
    cvflann::flann_centers_init_t centers_init= cvflann::FLANN_CENTERS_GONZALES;
    float cb_index=0.2f;

    cvflann::KMeansIndexParams  k_params=cvflann::KMeansIndexParams(branching,iterations,centers_init, cb_index);

    //cv::flann::KMeansIndexParams k_params(10, 1000, cvflann::FLANN_CENTERS_KMEANSPP,0.01);

    number_of_clusters= cvflann::hierarchicalClustering<cvflann::L2<float> >(featuresFlann,centersFlann,k_params);

    return number_of_clusters;

}

template <class T>
void RTAB_incremental_vocabulary<T>::create_and_update_vocab()
{
	if(!notIndexedDescriptors_.empty())
	{
		assert(indexedDescriptors_.cols == notIndexedDescriptors_.cols &&
				 indexedDescriptors_.type() == notIndexedDescriptors_.type());

		//concatenate descriptors
		indexedDescriptors_.push_back(notIndexedDescriptors_);  //插入新进来的图片的IndexedDescriptors

		notIndexedDescriptors_ = cv::Mat();
		notIndexedWordIds_.clear(); //把新加进来的notIndexedwords清空
	}

	if(!indexedDescriptors_.empty())
	{
		cv::flann::IndexParams * params = createFlannIndexParams(1);
		flannIndex_.build(indexedDescriptors_, *params, getFlannDistanceType());
		delete params;
	}
}


template <class T>
void RTAB_incremental_vocabulary<T>::search_descriptors(const cv::Mat  query_descriptors, cv::Mat & indices, cv::Mat & dists, int knn)
{
	assert(notIndexedDescriptors_.empty() && notIndexedWordIds_.size() == 0);

	if(!indexedDescriptors_.empty())
	{
		assert(query_descriptors.type() == indexedDescriptors_.type() && query_descriptors.cols == indexedDescriptors_.cols);


		flannIndex_.knnSearch(query_descriptors, indices, dists, knn, cv::flann::SearchParams(128));

		if( dists.type() == CV_32S )
		{
			cv::Mat temp;
			dists.convertTo(temp, CV_32F);
			dists = temp;
		}
	}
}

template <class T>
void RTAB_incremental_vocabulary<T>::search_image(const cv::Mat & queryDescriptors, cv::Mat & indices, cv::Mat & dists, int k, int imgNum, vector<unsigned int> labels, vector<int> images, multimap<int, int> &imageScore, vector<RankedScore> &rankedScore)
{
  //  clock_t begin2 = clock();
    this->search_descriptors(queryDescriptors, indices, dists, k);
    //clock_t end2 = clock();
    //double query_time = double(end2 - begin2) / CLOCKS_PER_SEC;
    cout.precision(5);
   // cout<<"query time "<<query_time<<endl;

    std::vector<int> indicesVec(indices.rows*indices.cols);

    if (indices.isContinuous())
    {
        indicesVec.assign((int*)indices.datastart, (int*)indices.dataend);
    }

    cout<<"indicesVec.size() "<<indicesVec.size()<<endl;

    vector<int> imageLabelsVec;

    /// Process Nearest Neighbor Distance Ratio
    float nndRatio = 0.8;

    for(int i=0; i<indicesVec.size(); i++)
    {
       // if(dists.at<float>(i,0)<nndRatio*dists.at<float>(i,1))
       // {
           cout<<"indicesVec["<<i<<"] "<<indicesVec[i]<<"  image labels "<<labels[indicesVec[i]]<<endl;
            imageLabelsVec.push_back(labels[indicesVec[i]]);
        //}
    }



    for(int i=0; i<imgNum; i++)
    {
        imageScore.insert(pair<int,int>(i+1,0));
    }

    cout<<imageScore.size()<<endl;

    int score=0;

    set<int> numOfdifferentLabels;
    for(int i=0; i<imageLabelsVec.size(); i++)
    {
            numOfdifferentLabels.insert(imageLabelsVec[i]);
    }

    for(auto ite=imageScore.begin(); ite!=imageScore.end(); ite++)
    {
        for(int i=0; i<imageLabelsVec.size(); i++)
        {
            if((*ite).first==imageLabelsVec[i])
            {
                (*ite).second++;
            }

        }

        cout<<(*ite).first<<" "<<(*ite).second<<endl;

        ///normalize
        float normalized_score=(float)(*ite).second/(float)imageLabelsVec.size();

        RankedScore singleRankedScore;

       // (*ite).second=normalized_score;
       // cout.precision(5);
        //cout<<(*ite).first<<" "<<(*ite).second<<" "<<normalized_score<<endl;
        singleRankedScore.imageIndex=(*ite).first;

        /*
        ///tf-idf weighting scheme
        double N=imgNum;
        double nw= numOfdifferentLabels.size();///the number of images containing descriptor w
        double nwi=1;    ///the number of occurrences of descriptor w in Ii
        double ni=queryDescriptors.rows; ///ni the total number of words in Ii.


        double weighting = (nwi/ni)*log(N/nw);

        cout.precision(5);
        cout<<"N "<<N<<"nw "<<nw<<"nwi "<<nwi<<"ni "<<ni<<endl;
        cout<<"weighting "<<weighting<<endl;
        */
        singleRankedScore.imageScore=normalized_score;

        rankedScore.push_back(singleRankedScore);
    }


        sort(rankedScore.begin(),rankedScore.end(),by_number());

        cout.precision(10);
        for(int i=0; i<rankedScore.size();i++)
        {
            cout<<rankedScore[i].imageIndex<<" "<<rankedScore[i].imageScore<<endl;
        }
}

template <class T>
void RTAB_incremental_vocabulary<T>::computeLikelihood(int imgNum, vector<RankedScore> rankedScore, vector<double> likelihood)
{
    int imageCount=0;  //count the images which has the non-zero likelihood
    for(int i=0; i<rankedScore.size(); i++)
    {
        if(rankedScore[i].imageScore!=0)
        {
            imageCount++;
        }
    }
    cout<<"The number of images which have non-zero likelihood is "<<imageCount<<endl;
    double mean=1/(double)imageCount;
    cout<<"mean "<<mean<<endl;
    double standardDeviation;
    double variance = 0;
    for(int i = 0; i < rankedScore.size(); i++)
    {
        if(rankedScore[i].imageScore!=0)
        {
            variance += (rankedScore[i].imageScore - mean) * (rankedScore[i].imageScore - mean) ;
        }
    }

    standardDeviation=sqrt(variance / rankedScore.size());

    cout.precision(5);

    cout<<"standardDeviation "<<standardDeviation<<endl;

    likelihood.resize(rankedScore.size());
    for(int i=0; i<rankedScore.size(); i++)
    {
        if(rankedScore[i].imageScore!=0)
        {
            likelihood[i]=(rankedScore[i].imageScore-standardDeviation)/mean;
        }
        else
        {
            likelihood[i]=0;
        }
        cout<<"image "<<rankedScore[i].imageIndex<<" imageLikelihood "<<likelihood[i]<<endl;
    }

}

typedef RTAB_incremental_vocabulary<float> vocabulary32;
typedef RTAB_incremental_vocabulary<double> vocabulary64;



#endif /* RTAB_incremental_vocabulary_cpp */
