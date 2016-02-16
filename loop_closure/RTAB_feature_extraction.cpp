#include "RTAB_feature_extraction.hpp"

#if CV_MAJOR_VERSION == 2

RTAB_feature_extraction::RTAB_feature_extraction()
{


}

bool RTAB_feature_extraction::RTAB_feature_extraction_exe(Mat img, Mat &descriptors)
{
     //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector(minHessian);

    SurfDescriptorExtractor extractor;

    vector<KeyPoint> keyPoints;

    detector.detect(img, keyPoints);

    extractor.compute(img, keyPoints, descriptors);

    if(descriptors.rows>300)
    {
        cout<<"has enough descriptors "<<descriptors.size()<<endl;
        return true;
    }
    else
    {
        cout<<"Not Enough descriptors, Only has "<<descriptors.size()<<endl;
        return false;
    }


}


RTAB_feature_extraction::~RTAB_feature_extraction()
{
    //dtor
}

#endif

