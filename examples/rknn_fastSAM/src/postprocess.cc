#include "postprocess.h"
#include <math.h>

#include <cstdlib>
#include <ctime>


#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline cv::Vec3b GenerateRandomColor()
{
    cv::RNG Rng(cv::getTickCount());
    int Blue = Rng.uniform(0, 256);
    int Green = Rng.uniform(0, 256);
    int Red = Rng.uniform(0, 256);

    return cv::Vec3b(Blue, Green, Red);
}


static inline float fast_exp(float x)
{
    // return exp(x);
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}

static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
{
    float Inter = 0;
    float Total = 0;
    float XMin = 0;
    float YMin = 0;
    float XMax = 0;
    float YMax = 0;
    float Area1 = 0;
    float Area2 = 0;
    float InterWidth = 0;
    float InterHeight = 0;

    XMin = ZQ_MAX(XMin1, XMin2);
    YMin = ZQ_MAX(YMin1, YMin2);
    XMax = ZQ_MIN(XMax1, XMax2);
    YMax = ZQ_MIN(YMax1, YMax2);

    InterWidth = XMax - XMin;
    InterHeight = YMax - YMin;

    InterWidth = (InterWidth >= 0) ? InterWidth : 0;
    InterHeight = (InterHeight >= 0) ? InterHeight : 0;

    Inter = InterWidth * InterHeight;

    Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
    Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

    Total = Area1 + Area2 - Inter;

    return float(Inter) / float(Total);
}

static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

/****** FastSAM ****/
GetResultRectFastSAM::GetResultRectFastSAM()
{
}

GetResultRectFastSAM::~GetResultRectFastSAM()
{
}

float GetResultRectFastSAM::sigmoid(float x)
{
    return 1 / (1 + fast_exp(-x));
}

int GetResultRectFastSAM::GenerateMeshGrid()
{
    int ret = 0;
    if (HeadNum == 0)
    {
        printf("=== FastSAM MeshGrid  Generate failed! \n");
    }

    for (int index = 0; index < HeadNum; index++)
    {
        for (int i = 0; i < MapSize[index][0]; i++)
        {
            for (int j = 0; j < MapSize[index][1]; j++)
            {
                MeshGrid.push_back(float(j + 0.5));
                MeshGrid.push_back(float(i + 0.5));
            }
        }
    }

    printf("=== FastSAM MeshGrid  Generate success! \n");

    return ret;
}

int GetResultRectFastSAM::GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects, cv::Mat &SegMask)
{
    int ret = 0;
    if (MeshGrid.empty())
    {
        ret = GenerateMeshGrid();
    }

    int gridIndex = -2;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
    float cls_val = 0;
    float cls_max = 0;
    int cls_index = 0;

    int quant_zp_cls = 0, quant_zp_reg = 0, quant_zp_msk, quant_zp_seg = 0;
    float quant_scale_cls = 0, quant_scale_reg = 0, quant_scale_msk = 0, quant_scale_seg = 0;

    DetectRect temp;
    std::vector<DetectRect> detectRects;

    for (int index = 0; index < HeadNum; index++)
    {
        int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
        int8_t *cls = (int8_t *)pBlob[index * 2 + 1];
        int8_t *msk = (int8_t *)pBlob[HeadNum * 2 + index];

        quant_zp_reg = qnt_zp[index * 2 + 0];
        quant_zp_cls = qnt_zp[index * 2 + 1];
        quant_zp_msk = qnt_zp[HeadNum * 2 + index];

        quant_scale_reg = qnt_scale[index * 2 + 0];
        quant_scale_cls = qnt_scale[index * 2 + 1];
        quant_scale_msk = qnt_scale[HeadNum * 2 + index];

        float sfsum = 0;
        float locval = 0;
        float locvaltemp = 0;

        for (int h = 0; h < MapSize[index][0]; h++)
        {
            for (int w = 0; w < MapSize[index][1]; w++)
            {
                gridIndex += 2;

                if (1 == ClassNum)
                {
                    cls_max = sigmoid(DeQnt2F32(cls[0 * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w], quant_zp_cls, quant_scale_cls));
                    cls_index = 0;
                }
                else
                {
                    for (int cl = 0; cl < ClassNum; cl++)
                    {
                        cls_val = cls[cl * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w];

                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }
                    cls_max = sigmoid(DeQnt2F32(cls_max, quant_zp_cls, quant_scale_cls));
                }

                if (cls_max > ObjectThresh)
                {
                    RegDfl.clear();
                    for (int lc = 0; lc < 4; lc++)
                    {
                        sfsum = 0;
                        locval = 0;
                        for (int df = 0; df < DFLNum; df++)
                        {
                            locvaltemp = exp(DeQnt2F32(reg[((lc * DFLNum) + df) * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w], quant_zp_reg, quant_scale_reg));
                            RegDeq[df] = locvaltemp;
                            sfsum += locvaltemp;
                        }
                        for (int df = 0; df < DFLNum; df++)
                        {
                            locvaltemp = RegDeq[df] / sfsum;
                            locval += locvaltemp * df;
                        }

                        RegDfl.push_back(locval);
                    }

                    xmin = (MeshGrid[gridIndex + 0] - RegDfl[0]) * Strides[index];
                    ymin = (MeshGrid[gridIndex + 1] - RegDfl[1]) * Strides[index];
                    xmax = (MeshGrid[gridIndex + 0] + RegDfl[2]) * Strides[index];
                    ymax = (MeshGrid[gridIndex + 1] + RegDfl[3]) * Strides[index];

                    xmin = xmin > 0 ? xmin : 0;
                    ymin = ymin > 0 ? ymin : 0;
                    xmax = xmax < InputWidth ? xmax : InputWidth;
                    ymax = ymax < InputHeight ? ymax : InputHeight;

                    if (xmin >= 0 && ymin >= 0 && xmax <= InputWidth && ymax <= InputHeight)
                    {
                        temp.xmin = xmin / InputWidth;
                        temp.ymin = ymin / InputHeight;
                        temp.xmax = xmax / InputWidth;
                        temp.ymax = ymax / InputHeight;
                        temp.classId = cls_index;
                        temp.score = cls_max;

                        for (int ms = 0; ms < MaskNum; ms++)
                        {
                            temp.mask[ms] = DeQnt2F32(msk[ms * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w], quant_zp_msk, quant_scale_msk);
                        }
                        detectRects.push_back(temp);
                    }
                }
            }
        }
    }

    std::sort(detectRects.begin(), detectRects.end(), [](DetectRect &Rect1, DetectRect &Rect2) -> bool
              { return (Rect1.score > Rect2.score); });

    for (int i = 0; i < detectRects.size(); ++i)
    {
        float xmin1 = detectRects[i].xmin;
        float ymin1 = detectRects[i].ymin;
        float xmax1 = detectRects[i].xmax;
        float ymax1 = detectRects[i].ymax;
        int classId = detectRects[i].classId;
        float score = detectRects[i].score;

        if (classId != -1)
        {
            // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1的格式存放在vector<float>中
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin1));
            DetectiontRects.push_back(float(ymin1));
            DetectiontRects.push_back(float(xmax1));
            DetectiontRects.push_back(float(ymax1));

            for (int j = i + 1; j < detectRects.size(); ++j)
            {
                float xmin2 = detectRects[j].xmin;
                float ymin2 = detectRects[j].ymin;
                float xmax2 = detectRects[j].xmax;
                float ymax2 = detectRects[j].ymax;
                float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                if (iou > NmsThresh)
                {
                    detectRects[j].classId = -1;
                }
            }
        }
    }

    int8_t *seg = (int8_t *)pBlob[9];
    quant_zp_seg = qnt_zp[9];
    quant_scale_seg = qnt_scale[9];

    int left = 0, top = 0, right = 0, bottom = 0;
    float SegSum = 0;

    for (int i = 0; i < detectRects.size(); ++i)
    {
        if (-1 != detectRects[i].classId)
        {
            left = int(detectRects[i].xmin * SegWidth + 0.5);
            top = int(detectRects[i].ymin * SegHeight + 0.5);
            right = int(detectRects[i].xmax * SegWidth + 0.5);
            bottom = int(detectRects[i].ymax * SegHeight + 0.5);

            cv::Vec3b Colors = GenerateRandomColor();
            for (int h = top; h < bottom; ++h)
            {
                for (int w = left; w < right; ++w)
                {
                    SegSum = 0;
                    for (int s = 0; s < MaskNum; ++s)
                    {
                        SegSum += detectRects[i].mask[s] * DeQnt2F32(seg[s * SegWidth * SegHeight + h * SegWidth + w], quant_zp_seg, quant_scale_seg);
                    }

                    if (1 / (1 + exp(-SegSum)) > 0.5)
                    {
                        SegMask.at<cv::Vec3b>(h, w) = Colors;
                    }
                }
            }
        }
    }

    return ret;
}
