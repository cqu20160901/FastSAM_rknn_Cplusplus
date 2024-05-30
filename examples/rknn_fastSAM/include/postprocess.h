#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

#include <opencv2/highgui.hpp>

typedef signed char int8_t;
typedef unsigned int uint32_t;

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
    float mask[32];
} DetectRect;

// FastSAM
class GetResultRectFastSAM
{
public:
    GetResultRectFastSAM();

    ~GetResultRectFastSAM();

    int GenerateMeshGrid();

    int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects, cv::Mat &SegMask);

    float sigmoid(float x);

private:
    std::vector<float> MeshGrid;

    int ClassNum = 1;
    int HeadNum = 3;

    int InputWidth = 640;
    int InputHeight = 352;
    int Strides[3] = {8, 16, 32};
    int MapSize[3][2] = {{44, 80}, {22, 40}, {11, 20}};

    int MaskNum = 32;
    int SegWidth = 160;
    int SegHeight = 88;

    float NmsThresh = 0.5;
    float ObjectThresh = 0.25;

    std::vector<float> RegDfl;
    float RegDeq[26] = {0};
    int DFLNum = 26;
};

#endif
