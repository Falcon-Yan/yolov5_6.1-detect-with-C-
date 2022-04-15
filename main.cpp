// dnn_yolov4.cpp : 定义控制台应用程序的入口点。
//
//  #include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include "yolov5_dnn.h"

using namespace cv;
using namespace std;

YOLOv5Detector *detector = new YOLOv5Detector();
vector<string> classNamesVec;
vector<DetectResult> results;
//测试模型
string label_file = "/home/yan/workspace/OpenCVprojects/YOLOv5_6.1/file/coco.names"; // 类别标签文件    发现最好不要用相对路径 相对路径从当强shell目录算起
string model = "/home/yan/workspace/OpenCVprojects/YOLOv5_6.1/file/yolov5s.onnx";
// 消毒目标识别
// String label_file = "/home/yan/workspace/OpenCVprojects/YOLOv5_6.1/file/label.names"; // 类别标签文件
// String model ="/home/yan/workspace/OpenCVprojects/YOLOv5_6.1/file/detect_button.onnx";

int main(int argc, char *argv[])
{
	//加载类别
	detector->loadLabels(classNamesVec, label_file);
	detector->initConfig(model, 640, 640, 0.8f);
	detector->useCUDA();
	VideoCapture cap;
	cap.open(0);
	Mat frame;
	while (true)
	{
		cap >> frame;
		detector->detect(frame, results);
		for (DetectResult dr : results)
		{
			cv::Rect box = dr.box;
			cv::putText(frame, classNamesVec[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));

			Point pos(dr.box.x+0.5*dr.box.width,dr.box.y+0.5*dr.box.height);
			cout<<"类别："<<classNamesVec[dr.classId]<<"   坐标："<<pos<<endl;
		}
		cv::imshow("YOLOv5-6.1 + OpenCV DNN", frame);
		char c = cv::waitKey(33);
		if (c == 27)
		{ // ESC 退出
			break;
		}
		// reset for next frame
		results.clear();
	}
	return 0;
}
