//
//  RTAB_parameter.h
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-02-13.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef RTAB_parameter_h
#define RTAB_parameter_h


// parameter in RTAB system
// Table 1
class RTAB_parameter
{
public:
    static const int T_stm_ = 30;
    constexpr static const double T_similarity_ = 0.2;
    constexpr static const double T_recent_ = 0.2;
    
    constexpr static const double surf_T_nndr_ = 0.8;
    static const int surf_T_max_feature_ = 400;  // maxinum number of feature
    static const int surf_T_min_feature_ = 50;  // mininum number of feature
    constexpr static const double surf_T_bad_ = 0.25;
    
    /*
    RTAB_parameter()
    {
        T_stm_ = 30;
        T_similarity_ = 0.2;
        T_recent_ = 0.2;
        
        surf_T_nndr_ = 0.8;
        surf_T_max_feature_ = 400;
        surf_T_min_feature_ = 50;
        surf_T_bad_ = 0.25;
    }
     */
    
};


#endif /* RTAB_parameter_h */
