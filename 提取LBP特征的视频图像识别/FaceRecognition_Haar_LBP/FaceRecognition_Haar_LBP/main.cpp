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
	string VideoPath="girl.flv";//��Ƶ�ĵ�ַ
	string TemplateFacePath="girl.png";//ģ��ͼ��ĵ�ַ
	FaceRecognitionHaarLBP(VideoPath.c_str(),TemplateFacePath.c_str());

	return 0;
}