#ifndef RTAB_FEATURE_EXTRACTION_H
#define RTAB_FEATURE_EXTRACTION_H

#include <cassert>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp> // for homography

using namespace std;
using namespace cv;

class RTAB_feature_extraction
{
    public:
        RTAB_feature_extraction();
        bool RTAB_feature_extraction_exe(Mat img, Mat &descriptors);
        virtual ~RTAB_feature_extraction();
    protected:
    private:
};

#endif // RTAB_FEATURE_EXTRACTION_H

