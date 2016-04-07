#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include <cstdio>    
#include <cstdlib>    
#include <Windows.h>   
#include<math.h>
#include"lbp.h"

using namespace std;   
using namespace cv;

int main()
{
	string VideoPath="girl.flv";//视频的地址
	string TemplateFacePath="girl.png";//模板图像的地址
	FaceRecognitionHaarLBP(VideoPath.c_str(),TemplateFacePath.c_str());

	return 0;
}