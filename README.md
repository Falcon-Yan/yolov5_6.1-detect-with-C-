# yolov5_6.1-detect-with-C-
将yolov5_6.1 训练的模型用openCV4.5.3 DNN部署到C++ ,实现实时目标检测。
由于上传文件大小限制，yolov5s. ，文件的获取方法：
1.去其他仓库下载，
2.先下载yolov5s.pth ，再转为onnx格式。此处用的是yolov5原文件夹中的/export.py脚本。
$ python path/to/export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...
