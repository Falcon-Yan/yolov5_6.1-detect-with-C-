cmake_minimum_required(VERSION 3.0)       #设置最小cmake版本

project(myproject)                                                   #设置项目名称

add_compile_options(-g -Wall -std=c++11)    #增加编译参数

set(CMAKE_BUILD_TYPE Debug)                         #设置为可调试的Debug模式

include_directories(${CMAKE_SOURCE_DIR}/include)                               #指定头文件路径

set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)                        #指定库文件存放路径lib

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)            #指定可执行文件存放路径

aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC)                          #定义一个SRC变量，值为./src文件夹下所有的源代码文件

add_library(yolov5 SHARED ${SRC})                          #生成动态链接库libmyfunc.so文件，存入lib文件夹    自己写了函数时使用这条
  
link_directories(${CMAKE_SOURCE_DIR}/lib)              #将该路径添加为库文件搜索路径

######################   openCV 3.4.6   ##########################################################

#添加OPENCV库
#指定OpenCV版本，代码如下
# find_package(OpenCV 3.4.6 REQUIRED)                                                                                     #使用时打开注释
#如果不需要指定OpenCV版本，代码如下
# find_package(OpenCV REQUIRED)

#添加OpenCV头文件
# include_directories(${OpenCV_INCLUDE_DIRS})                                                                  #使用时打开注释

#显示OpenCV_INCLUDE_DIRS的值
# message(${OpenCV_INCLUDE_DIRS})                                                                                          #使用时打开注释

# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 ）
# add_executable(main.out main.cpp)                      #编译成可执行文件main.out 存入bin文件夹                   #使用时打开注释

# 将库文件链接到可执行程序上
# target_link_libraries( main.out  ${OpenCV_LIBS})
###############################################################################           #使用时打开注释
# target_link_libraries(main.out myfunc)                #如果自己写了子函数，生成了库需要链接,就使用这条代码，lib文件夹中的动态链接库libmyfunc.so


###############################     关于切换openCV 版本的方法     ##############################
# 1.使用3.4.6时可以把 openCV 4.5.3  部分全注释掉，然后打开openCV 3.4.6  的注释
# 2.更通用的方法是使用
# https://blog.csdn.net/weixin_39956356/article/details/102643415?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-3&spm=1001.2101.3001.4242.2
# 中的方法：pkg-config
# PKG_SEARCH_MODULE(PKG_OPENCV REQUIRED opencv)	使用openCV 3.4.6
# PKG_SEARCH_MODULE(PKG_OPENCV REQUIRED opencv4)	使用openCV 4.5.3

######################   openCV 4.5.3   ##########################################################
# 非root下需要自己加入PKG_CONFIG_PATH，不然报错!!!
#SET(ENV{PKG_CONFIG_PATH} /home/yan/lib/opencv-4.5.3/build/installed/lib/pkgconfig)                                                 
FIND_PACKAGE(PkgConfig REQUIRED)				#PkgConfig名字是固定的，代表准备加入pkg-config模块，即查找/usr/bin/pkg-config
												#eg: -- Found PkgConfig: /usr/bin/pkg-config (found version "0.29.1") 
PKG_SEARCH_MODULE(PKG_OPENCV REQUIRED opencv4)	#通过执行pkg-config程序，并指定我需要的模块是opencv,注意opencv名字固定，是源于安装OpenCV生成的opencv.pc，PKG_OPENCV是前缀(观察下面)，
												#eg: -- Checking for one of the modules 'opencv'
#PKG_OPENCV相关头文件/库的输出
MESSAGE(STATUS "PKG_OPENCV_FOUND:          ${PKG_OPENCV_FOUND}")			#有没有找到		    eg: -- PKG_OPENCV_FOUND:        1
MESSAGE(STATUS "PKG_CONFIG_EXECUTABLE:     ${PKG_CONFIG_EXECUTABLE}")		#pkg-config脚本路径	 eg:  /usr/bin/pkg-config
MESSAGE(STATUS "PKG_CONFIG_VERSION_STRING: ${PKG_CONFIG_VERSION_STRING}")	#pkg-config版本号	  相当于执行: pkg-config --version

MESSAGE(STATUS "PKG_OPENCV_VERSION:        ${PKG_OPENCV_VERSION}")			#opencv的版本		  相当于执行: pkg-config opencv --modversion
MESSAGE(STATUS "PKG_OPENCV_PREFIX:         ${PKG_OPENCV_PREFIX}")			#opencv的安装路径PREFIX

MESSAGE(STATUS "PKG_OPENCV_INCLUDE_DIRS:   ${PKG_OPENCV_INCLUDE_DIRS}")		#头文件路径		    include路径，多个，分开
MESSAGE(STATUS "PKG_OPENCV_LIBRARY_DIRS:   ${PKG_OPENCV_LIBRARY_DIRS}")		#动态库路径		    lib路径，多个，分开
MESSAGE(STATUS "PKG_OPENCV_LIBRARIES:      ${PKG_OPENCV_LIBRARIES}")		#有那些动态库?        相当于执行: pkg-config opencv --libs-only-l，只列出库名字
MESSAGE(STATUS "PKG_OPENCV_LDFLAGS:        ${PKG_OPENCV_LDFLAGS}")			#所有需要链接的标志	  相当于执行: pkg-config opencv --libs，不仅有库名字，还有其路径
MESSAGE(STATUS "PKG_OPENCV_LDFLAGS_OTHER:  ${PKG_OPENCV_LDFLAGS_OTHER}")	#其他需要链接的标志	  相当于执行: pkg-config opencv --libs-only-other
MESSAGE(STATUS "PKG_OPENCV_CFLAGS:         ${PKG_OPENCV_CFLAGS}")			#cflag可以指头文件    相当于执行: pkg-config opencv --cflags，多个，分开
MESSAGE(STATUS "PKG_OPENCV_CFLAGS_OTHER:   ${PKG_OPENCV_CFLAGS_OTHER}")		#其他cflag标志		 相当于执行: pkg-config opencv --libs-only-other

# 添加三方opencv的头文件路径
INCLUDE_DIRECTORIES(${PKG_OPENCV_INCLUDE_DIRS})

# 把生成的 myfunc 库和所需的其它库链接起来
TARGET_LINK_LIBRARIES ( yolov5 ${PKG_OPENCV_LDFLAGS})

# 指定生成目标
ADD_EXECUTABLE(main.out main.cpp)
# 为指定的bin文件添加三方链接库
TARGET_LINK_LIBRARIES(main.out ${PKG_OPENCV_LDFLAGS})

TARGET_LINK_LIBRARIES(main.out yolov5)                #如果自己写了子函数，生成了库需要链接,就使用这条代码，lib文件夹中的动态链接库libmyfunc.so