#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/features2d.hpp>
using namespace cv;
using namespace std;
int main(int argc, char **argv) {
	CommandLineParser parser(argc, argv, "{help h||}{ @image | ../data/baboon.jpg | }");
	Mat src;
	string filename = parser.get<string>("@image");
	if ((src = imread(filename, IMREAD_COLOR)).empty())
	{
		cout << "Couldn't load image";
		return -1;
	}
	Mat gray, thresh, smooth,morph ;
	cvtColor(src, src, COLOR_BGRA2BGR);
	cout << "Number of input channels=" << src.channels();
	bilateralFilter(src, smooth, 9, 30, 30);
	cvtColor(smooth, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(5, 3),1.5);
	adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY
		, 11, 2);
	threshold(gray, thresh, 50, 255, THRESH_BINARY+THRESH_OTSU);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(25, 9));
	morphologyEx(thresh, morph, MORPH_OPEN, kernel);
	//morphologyEx(thresh, thresh, MORPH_OPEN, kernel);
	//
	Ptr<MSER> ms = MSER::create();
	vector<vector<Point> > regions;
	vector<Rect> mser_boundingbox;
	ms->detectRegions(thresh, regions, mser_boundingbox);

	for (int i = 0; i < regions.size(); i++)
	{
		
		rectangle(src, mser_boundingbox[i], CV_RGB(0, 255, 0));
	}
	imshow("processed_image", morph);
	imshow("original", src);
	//imshow("thresh", thresh);
	moveWindow("original", 10, 50);
	moveWindow("thresh", 30, 70);
	waitKey(0);
	return 0;
	
}
