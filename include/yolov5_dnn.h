#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
using namespace cv;
using namespace std;

struct DetectResult {
    int classId;
    float score;
    cv::Rect box;
};

class YOLOv5Detector {
public:
    void initConfig(std::string onnxpath, int iw, int ih, float threshold);
    void loadLabels(vector<string> &classNamesVec,string label_file);
    void detect(cv::Mat & frame, std::vector<DetectResult> &result);
    void useCUDA();
private:
    int input_w = 640;
    int input_h = 640;
    cv::dnn::Net net;
    float threshold_score = 0.25;   //预测概率最高的类别的分值的阈值   或 置信度
    float nms_threshold = 0.45;
    bool USE_CUDA = false;
    // vector<pair<int,Point>> targets;
};