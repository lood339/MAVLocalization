/** @Main KVLD algorithm implementation
 ** @Containing scale image pyramid, VLD structure and KVLD algorithm
 ** @author Zhe Liu
 **/

/*
Copyright (C) 2011-12 Zhe Liu and Pierre Moulon.
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).
*/
#include "kvld.h"
#include <functional>
#include <numeric>
#include <map>


ImageScale::ImageScale(const cv::Mat& I, double r){
  IntegralImages inter(I);
  radius_size=r;
  step=sqrt(2.0);
  int size= std::max(I.cols,I.rows);

  int number= int(log(size/r)/log(2.0))+1;
  angles.resize(number);
  magnitudes.resize(number);
  ratios.resize(number);

  GradAndNorm(I,angles[0],magnitudes[0]);
  ratios[0]=1;

//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
  for (int i=1; i<number; i++){
	  cv::Mat I2;
    double ratio=1*pow(step,i);  
   
	I2 = cv::Mat(int(I.rows / ratio), int(I.cols / ratio), CV_32FC1);
	angles[i] = cv::Mat(int(I.rows / ratio), int(I.cols / ratio), CV_32FC1);
	magnitudes[i] = cv::Mat(int(I.rows / ratio), int(I.cols / ratio), CV_32FC1);

    for (int colI=0;colI<I2.cols; colI++){
      for (int rowI=0;rowI<I2.rows;rowI++){
        I2.at<float>(rowI,colI)=inter(double(colI+0.5)*ratio,double(rowI+0.5)*ratio,ratio);
      }
    }

    GradAndNorm(I2,angles[i],magnitudes[i]);
    ratios[i]=ratio;
  }
}

void ImageScale::GradAndNorm(const cv::Mat& I, cv::Mat& angle, cv::Mat& m){
	angle = cv::Mat(I.rows, I.cols, CV_32FC1);
	m = cv::Mat(I.rows, I.cols, CV_32FC1);
		angle = cv::Mat::zeros(angle.rows, angle.cols, angle.type());
		m = cv::Mat::zeros(m.rows, m.cols, m.type());

//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
		for (int x=1;x<I.cols-1;x++)
			for (int y=1;y<I.rows-1;y++){
				float gx = I.at<float>(y, x + 1) - I.at<float>(y, x - 1);
				float gy = I.at<float>(y + 1, x) - I.at<float>(y - 1, x);

				if (!anglefrom(gx, gy, angle.at<float>(y, x)))
					angle.at<float>(y, x) = -1;
				m.at<float>(y, x) = sqrt(gx*gx + gy*gy);
			}
	}

int ImageScale::getIndex(const double r)const{
		const double step=sqrt(2.0);

		if (r<=radius_size) return 0;
		else{
			double range_low=radius_size;
			int index=0;
			while (r>range_low*step){
				index++;
				range_low*=step;
			}
			return std::min(int(angles.size()-1), index);
		}
	}


template<typename T>
VLD::VLD(const ImageScale& series, T const& P1, T const& P2) : contrast(0.0) {
	//============== initializing============//
	principleAngle.fill(0);
	descriptor.fill(0);
	weight.fill(0);

	begin_point[0]=P1.cvKeyPoint.pt.x;
	begin_point[1] = P1.cvKeyPoint.pt.y;
	end_point[0] = P2.cvKeyPoint.pt.x;
	end_point[1] = P2.cvKeyPoint.pt.y;

	float dy= float(end_point[1]- begin_point[1]), dx= float(end_point[0]- begin_point[0]);
	  distance=sqrt(dy*dy+dx*dx);

	if (distance==0)
		std::cerr<<"Two SIFT points have the same coordinate"<<std::endl;

	float radius=std::max(distance/float(dimension+1), 2.0f);//at least 2

	double mainAngle= get_orientation();//absolute angle

	int image_index=series.getIndex(radius);

	const cv::Mat& ang = series.angles[image_index];
	const cv::Mat& m   = series.magnitudes[image_index];
	double ratio=series.ratios[image_index];
  
 // std::cout<<std::endl<<"index of image "<<radius<<" "<<image_index<<" "<<ratio<<std::endl;
	
  int w=m.cols ,h=m.rows;
	float r=float(radius/ratio);//series.radius_size;
	float sigma2=r*r;
	//======Computing the descriptors=====//

	for (int i=0;i<dimension; i++){
		double statistic[binNum];
    std::fill_n(statistic, binNum, 0.0);

		float xi= float(begin_point[0]+ float(i+1)/(dimension+1)*(dx));
		float yi= float(begin_point[1]+ float(i+1)/(dimension+1)*(dy));
		yi/=float(ratio);
		xi/=float(ratio);
		
    for (int y=int(yi-r);y<=int(yi+r+0.5);y++){
			for (int x=int(xi-r);x<=int(xi+r+0.5);x++){
				float d=point_distance(xi,yi,float(x),float(y));
				if (d<=r && inside(w,h,x,y,1)){
					//================angle and magnitude==========================//
					double angle;
					if (ang.at<float>(y,x)>=0)
						angle = ang.at<float>(y, x) - mainAngle;//relative angle
					else angle=0.0;

					//cout<<angle<<endl;
					while (angle<0)
						angle +=2*PI;
					while (angle>=2*PI)
						angle -=2*PI;
					
					//===============principle angle==============================//
					int index=int(angle*binNum/(2*PI)+0.5);

					double Gweight = exp(-d*d / 4.5 / sigma2)*(m.at<float>(y, x));
          // std::cout<<"in number "<<image_index<<" "<<x<<" "<<y<<" "<<m(y,x)<<std::endl;
					if (index<binNum)
						statistic[index]+=Gweight;
					else // possible since the 0.5
						statistic[0]+=Gweight;

					//==============the descriptor===============================//
					int index2=int(angle*subdirection/(2*PI)+0.5);
					assert(index2>=0 && index2<=subdirection);

					if (index2<subdirection)
						descriptor[subdirection*i+index2]+=Gweight;
					else descriptor[subdirection*i]+=Gweight;// possible since the 0.5
				}
			}
		}
		//=====================find the biggest angle of ith SIFT==================//
		int index,second_index;
		max(statistic,weight[i],binNum,index,second_index);
		principleAngle[i]=index;
	}

  normalize_weight(descriptor);
 
  contrast= std::accumulate(weight.begin(), weight.end(), 0.0);
	contrast/=distance/ratio;
	normalize_weight(weight);
}


float KVLD(const cv::Mat& I1,const cv::Mat& I2,
	std::vector<VLDKeyPoint>& F1, std::vector<VLDKeyPoint>& F2, const std::vector<cv::DMatch>& matches,
	std::vector<cv::DMatch>& matchesFiltered, std::vector<double>& score, Matrixf& E, std::vector<bool>& valide, KvldParameters& kvldParameters){
		matchesFiltered.clear();
		score.clear();

		ImageScale Chaine1(I1);
		ImageScale Chaine2(I2);

		std::cout<<"Image scale-space complete..."<<std::endl;

		float range1=getRange(I1,std::min(F1.size(),matches.size()),kvldParameters.inlierRate,kvldParameters.rang_ratio);
		float range2=getRange(I2,std::min(F2.size(),matches.size()),kvldParameters.inlierRate,kvldParameters.rang_ratio);

		size_t size=matches.size();

		////================distance map construction, for use of selecting neighbors===============//
		//std::cout<<"computing distance maps"<<std::endl;
		//libNumerics::matrix<float> dist1=libNumerics::matrix<float>::zeros(F1.size(), F1.size());
		//libNumerics::matrix<float> dist2=libNumerics::matrix<float>::zeros(F2.size(), F2.size());

		//  for (int a1=0; a1<F1.size();++a1)
		//    for (int a2=0; a2<F1.size();++a2)
		//      dist1(a1,a2)=point_distance(F1[a1],F1[a2]);

		//  for (int b1=0; b1<F2.size();++b1)
		//    for (int b2=0; b2<F2.size();++b2)
		//      dist2(b1,b2)=point_distance(F2[b1],F2[b2]);

		fill(valide.begin(),valide.end(), true);
		std::vector<double> scoretable(size, 0);
		std::vector<size_t> result(size, 0);

		//============main iteration for match verification==========//
		std::cout<<"main iteration";
		bool change=true, initial=true;

		while(change){
			std::cout<<".";
			change=false;

			fill(scoretable.begin(), scoretable.end(), 0.0);
			fill(result.begin(), result.end(), 0);
			//========substep 1: search for each match its neighbors and verify if they are gvld-consistent ============//
			for (int it1=0; it1<size-1;it1++){
				if (valide[it1]){
					size_t a1=matches[it1].queryIdx, b1=matches[it1].trainIdx;

					for (int it2=it1+1; it2<size;it2++){
						if (valide[it2]){
							size_t a2=matches[it2].queryIdx, b2=matches[it2].trainIdx;
							float dist1=point_distance(F1[a1],F1[a2]);
							float dist2=point_distance(F2[b1],F2[b2]);
							if ( dist1>min_dist && dist2>min_dist
								&& (dist1<range1 || dist2<range2))
							{

									if(E.at<float>(it1,it2)==-1) //update E if unknow
									{
										E.at<float>(it1, it2) = -2; E.at<float>(it2, it1) = -2;

										if(!kvldParameters.geometry || consistent(F1[a1],F1[a2],F2[b1],F2[b2])<distance_thres)
										{
											VLD vld1(Chaine1,F1[a1],F1[a2]);
											VLD vld2(Chaine2,F2[b1],F2[b2]);
											//vld1.test();
											double error=vld1.difference(vld2);
											//std::cout<<std::endl<<it1<<" "<<it2<<" "<<dist1(a1,a2)<<" "<< dist2(b1,b2)<<" "<<error<<std::endl;
											if (error<juge)
											{
												E.at<float>(it1, it2) = (float)error;
												E.at<float>(it2, it1) = (float)error;
												//std::cout<<E(it2,it1)<<std::endl;
											}
										}
									}

									if (E.at<float>(it1, it2) >= 0)
									{
										result[it1]+=1;
										result[it2]+=1;
										scoretable[it1] += double(E.at<float>(it1, it2));
										scoretable[it2] += double(E.at<float>(it1, it2));
										if (result[it1]>=max_connection)
											break;
									}
							}
						}
					}
				}
			}

			//========substep 2: remove false matches by K gvld-consistency criteria ============//
			for (int it=0; it<size;it++){
				if (valide[it] && result[it]<kvldParameters.K)  {valide[it]=false;change=true;}
			}
			//========substep 3: remove multiple matches to a same point by keeping the one with the best average gvld-consistency score ============//
			if(uniqueMatch){
				for (int it1=0; it1<size-1;it1++){
					if (valide[it1]){
						size_t a1=matches[it1].queryIdx, b1=matches[it1].trainIdx;

						for (int it2=it1+1; it2<size;it2++)
							if (valide[it2]){
								size_t a2=matches[it2].queryIdx, b2=matches[it2].trainIdx;

								if(a1==a2||b1==b2
									|| (F1[a1].cvKeyPoint.pt.x == F1[a2].cvKeyPoint.pt.x &&
									F1[a1].cvKeyPoint.pt.y == F1[a2].cvKeyPoint.pt.y &&
									(F2[b1].cvKeyPoint.pt.x != F2[b2].cvKeyPoint.pt.x ||
									F2[b1].cvKeyPoint.pt.y != F2[b2].cvKeyPoint.pt.y))
									|| ((F1[a1].cvKeyPoint.pt.x != F1[a2].cvKeyPoint.pt.x ||
									F1[a1].cvKeyPoint.pt.y != F1[a2].cvKeyPoint.pt.y) &&
									F2[b1].cvKeyPoint.pt.x == F2[b2].cvKeyPoint.pt.x &&
									F2[b1].cvKeyPoint.pt.y == F2[b2].cvKeyPoint.pt.y)
									){
										//cardinal comparison
										if(result[it1]>result[it2]){
											valide[it2]=false;change=true;
										}else if(result[it1]<result[it2]){
											valide[it1]=false;change=true;

										}else if(result[it1]==result[it2]){
											//score comparison
											if (scoretable[it1]>scoretable[it2]){
												valide[it1]=false;change=true;
											}else if (scoretable[it1]<scoretable[it2]){
												valide[it2]=false;change=true;
											}
										}
								}
							}
					}
				}
			}
			//========substep 4: if geometric verification is set, re-score matches by geometric-consistency, and remove poorly scored ones ============================//
			if (kvldParameters.geometry){
				scoretable.resize(size,0);

				std::vector<bool> switching(size,false);

				for (int it1=0; it1<size;it1++){
					if (valide[it1]) {
						size_t a1=matches[it1].queryIdx, b1=matches[it1].trainIdx;
						float index=0.0f;
						int good_index=0;
						for (int it2=0; it2<size;it2++){
							if (it1!=it2 && valide[it2]){
								size_t a2=matches[it2].queryIdx, b2=matches[it2].trainIdx;
								float dist1=point_distance(F1[a1],F1[a2]);
								float dist2=point_distance(F2[b1],F2[b2]);
								if ((dist1<range1 || dist2<range2)
									&& (dist1>min_dist && dist2>min_dist)
									){
										float d=consistent(F1[a1],F1[a2],F2[b1],F2[b2]);
										scoretable[it1]+=d;
										index+=1;
										if (d<distance_thres)
											good_index++;
								}
							}
						}
						scoretable[it1]/=index;
						if (good_index<0.3f*float(index) && scoretable[it1]>1.2){switching[it1]=true;change=true;}
					}
				}
				for (int it1=0; it1<size;it1++){
					if (switching[it1])
						valide[it1]=false;
				}
			}
		}
		std::cout<<std::endl;

		//=============== generating output list ===================//
		for (int it=0; it<size;it++){
			if (valide[it]){
				matchesFiltered.push_back(matches[it]);
				score.push_back(scoretable[it]);
			}
		}
		return float(matchesFiltered.size())/matches.size();
}

void writeResult(const std::string output,const std::vector<VLDKeyPoint>& F1,const std::vector<VLDKeyPoint>& F2,const std::vector<cv::DMatch>& matches,
  const std::vector<cv::DMatch>& matchesFiltered,const std::vector<double>& score){
//========features
  //  std::ofstream feature1((output+"Detectors1.txt"));
  //  if (!feature1.is_open())
  //    std::cout<<"error while writing Features1.txt"<<std::endl;
  //  
  //  feature1<<F1.size()<<std::endl;
  //  for (auto it=F1.begin(); it!=F1.end();it++){
		//it->writeDetector(feature1);
  //  }
  //  feature1.close();

  //  std::ofstream feature2((output+"Detectors2.txt"));
  //  if (!feature2.is_open())
  //    std::cout<<"error while writing Features2.txt"<<std::endl;
  //  feature2<<F2.size()<<std::endl;
  //  for (auto it=F2.begin(); it!=F2.end();it++){
		//it->writeDetector(feature2);
  //  }
  //  feature2.close();

//========matches
    //std::ofstream initialmatches((output+"initial_matches.txt"));
    //if (!initialmatches.is_open())
    //  std::cout<<"error while writing initial_matches.txt"<<std::endl;
    //initialmatches<<matches.size()<<std::endl;
    //for (auto it=matches.begin(); it!=matches.end();it++){
    //    initialmatches<<it->queryIdx<<" "<<it->trainIdx<<std::endl;
    //    
    //}
    //initialmatches.close();

//==========kvld filtered matches
    std::ofstream filteredmatches((output+"kvld_matches.txt"));
    if (!filteredmatches.is_open())
      std::cout<<"error while writing kvld_filtered_matches.txt"<<std::endl;

    filteredmatches<<matchesFiltered.size()<<std::endl;
    for (auto it=matchesFiltered.begin(); it!=matchesFiltered.end();it++){
        filteredmatches<<it->queryIdx<<" "<<it->trainIdx<<std::endl;
        
    }
   filteredmatches.close();

//====== KVLD score of matches
   std::ofstream kvldScore((output+"kvld_matches_score.txt"));
    if (!kvldScore.is_open())
      std::cout<<"error while writing kvld_matches_score.txt"<<std::endl;
	
    for (std::vector<double>::const_iterator it=score.begin(); it!=score.end();it++){
        kvldScore<<*it<<std::endl;   
    }
   kvldScore.close();
}
