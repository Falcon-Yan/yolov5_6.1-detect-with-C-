#include "yolov5_dnn.h"

void YOLOv5Detector::initConfig(std::string onnxpath, int iw, int ih, float threshold)
{
    this->input_w = iw;
    this->input_h = ih;
    this->threshold_score = threshold;
    this->net = cv::dnn::readNetFromONNX(onnxpath);
}

void YOLOv5Detector::detect(cv::Mat &frame, std::vector<DetectResult> &results)
{
    // 图象预处理 - 格式化操作
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));   //转为正方形

    float x_factor = image.cols / (float)input_w;    //计算缩放比例 
    float y_factor = image.rows / (float)input_h;

    // 推理
    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    cv::Mat preds = this->net.forward();

    // 后处理, 1x25200x（5+类别数）
    // std::cout << "rows: "<< preds.size[1]<< " data: " << preds.size[2] << std::endl;
    cv::Mat det_output(preds.size[1], preds.size[2], CV_32F, preds.ptr<float>());
    float confidence_threshold = 0.5;
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < det_output.rows; i++)
    {
        float confidence = det_output.at<float>(i, 4);
        if (confidence < 0.45)
        {
            continue;
        }
        cv::Mat classes_scores = det_output.row(i).colRange(5, det_output.cols);
        cv::Point classIdPoint;
        double score;    //预测概率最高的类别的分值
        // cout<<score<<endl;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

        // 置信度 0～1之间
        if (score > this->threshold_score)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
        }
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, threshold_score, nms_threshold, indexes); //  score_threshold 0.25      nms_threshold :0.45
                                                                                    //&bboxes, const std::vector<float> &scores, const float score_threshold, const float nms_threshold,
    for (size_t i = 0; i < indexes.size(); i++)
    {
        DetectResult dr;
        int index = indexes[i];    
        int idx = classIds[index];
        dr.box = boxes[index];        //NMS处理后的预测锚框
        dr.classId = idx;                     //NMS处理后的预测classId
        dr.score = confidences[index];   //NMS处理后的预测score 
        cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                      cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
        results.push_back(dr);
    }

    std::ostringstream ss;
    std::vector<double> layersTimings;
    double freq = cv::getTickFrequency() / 1000.0;    // 用于返回CPU的频率 次/s 除以1000，得到次/ms
    double time = net.getPerfProfile(layersTimings) / freq;       //计算模型的推理时间
    ss << "FPS: " << 1000 / time << " ; time : " << time << " ms";
    putText(frame, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

void YOLOv5Detector::loadLabels(vector<string> &classNamesVec, string label_file)
{
    ifstream classNamesFile(label_file);
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
        {
            classNamesVec.push_back(className);
        }
    }
    else
    {
        cout << "打开文件失败" << endl;
    }
    cout << "导入类别数：" << classNamesVec.size() << endl;
    classNamesFile.close();
}

void YOLOv5Detector::useCUDA()
{
    USE_CUDA = true;
    if (this->USE_CUDA)
    {
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    }
}