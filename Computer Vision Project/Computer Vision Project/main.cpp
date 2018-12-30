#include<iostream>

#include<opencv2/core.hpp>

#include<opencv2/imgcodecs.hpp>

#include<opencv2/highgui.hpp>

#include<opencv2/imgproc.hpp>

#include<opencv2/features2d.hpp>

using namespace cv;

using namespace std;

vector<Mat> find_descriptors(vector<Mat> image_vector);
int best_template_match(Mat feature_descriptor, vector<Mat> template_vector_descriptors);
const char *getTextForEnum(int enumVal);
bool compare_rect(const Rect & a, const Rect & b);

int main(int argc, char **argv) {

	enum Plate_numbers 
	{
		A,
		B,
		C,
		D,
		E,
		F,
		G,
		H,
		I,
		J,
		K,
		L,
		M,
		N,
		O,
		P,
		Q,
		R,
		S,
		T,
		U,
		V,
		W,
		X,
		Y,
		Z,
		One,
		Two,
		Three,
		Four,
		Five,
		Six,
		Seven,
		Eight,
		Nine,
		Zero
	};

	//Get image to parse as command line argument
	CommandLineParser parser(argc, argv, "{help h||}{ @image | ../data/baboon.jpg | }");

	//Assign image to Mat 
	Mat src;
	string filename = parser.get<string>("@image");
	if ((src = imread(filename, IMREAD_COLOR)).empty())

	{

		cout << "Couldn't load image";

		return -1;

	}

	//Get rid of alphas in image
	Mat gray, thresh, smooth, morph;
	cvtColor(src, src, COLOR_BGRA2BGR);

	//Create grayscale image and blur for feature
	bilateralFilter(src, smooth, 9, 30, 30);
	cvtColor(smooth, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(5, 3), 1.5);
	adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

	//Find important features in image
	threshold(gray, thresh, 50, 255, THRESH_BINARY + THRESH_OTSU);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(25, 9));
	//morphologyEx(thresh, morph, MORPH_CLOSE, kernel);
	morphologyEx(thresh, morph, MORPH_OPEN, kernel);

	//
	Ptr<MSER> ms = MSER::create();
	vector<vector<Point> > regions;
	vector<Rect> mser_boundingbox;
	vector<Mat> license_characters;

	//Load template images into array
	vector<String> fn;
	glob("C:/Users/barkas2/Desktop/Program 3/Program 3/Program 3/characters/*.jpg", fn, false);

	vector<Mat> template_vector;
	size_t count = fn.size();
	for(size_t i = 0; i < count; i++)
	{
		template_vector.push_back(imread(fn[i]));
	}

	//Find regions that could contain license plate
	ms->detectRegions(thresh, regions, mser_boundingbox);
		
	//Filter out all boundingboxes outside of ratio threshold to get license numbers
	for(auto it = mser_boundingbox.begin(); it != mser_boundingbox.end();)
	{
		auto it2 = regions.begin();
		//Find ratio of each bounding box
		double ratio = ((double)it->width) / ((double)it->height);

		//If bounding box is outside threshold, excise it from vector
		if (ratio < .2 || ratio > .6)
		{
			it = mser_boundingbox.erase(it);
			it2 = regions.erase(it2);			
		} 
		else
		{
			++it;
			++it2;
		}
	}

	//For each bounding box, extract image into a new MAT
	for (int i = 0; i < mser_boundingbox.size(); i++)
	{
		int x_pos = mser_boundingbox[i].x;
		int y_pos = mser_boundingbox[i].y;
		int width = mser_boundingbox[i].width;
		int height = mser_boundingbox[i].height;

		if (x_pos - 5 >= 0)
			x_pos -= 5;
		if (y_pos - 5 >= 0)
			y_pos -= 5;
		if (x_pos + width + 10 < src.cols)
		{
			width += 10;
		}
		if (y_pos + height + 10 < src.rows)
		{
			height += 10;
		}
		license_characters.push_back(src
		(Rect(x_pos,
			y_pos,
			width,
			height)));
	}


	//perform edge detection on extracted features
	for (int i = 0; i < mser_boundingbox.size(); i++)
	{
		string output = to_string(i);
		imshow(output, license_characters[i]);
	}
	

	//Perform keypoint extraction for each feature-template combination
	//Create a vector of mats for descriptors of keypoints for each template image (one Mat per image)
	vector<Mat> template_descriptors_vector = find_descriptors(template_vector);

	//Create mats for descriptors of keypoints for each extracted feature
	vector<Mat> feature_descriptors_vector = find_descriptors(license_characters);

	//Print out the characters of the license plate in the correct order
	vector<const char *> characters;
	int best_match =0;
	for (int i = 0; i < license_characters.size(); i++)
	{
		best_match = best_template_match(feature_descriptors_vector[i], template_descriptors_vector);
		const char * temp_plate_text = getTextForEnum(best_match);
		characters.push_back(temp_plate_text);		
		best_match = 0;
	}

	int largest_x = 0;
	for (int i = 0; i < mser_boundingbox.size(); i++)
	{
		if (mser_boundingbox[i].x > largest_x) largest_x = mser_boundingbox[i].x;
	}
	int smallest_x = largest_x;
	
	int index = 0;
	for (int i = 0; i < mser_boundingbox.size(); i++)
	{
		if (mser_boundingbox[i].x < smallest_x)
		{
			smallest_x = mser_boundingbox[i].x;
			index = i;
		}
	}
	cout << "License plate read: " << characters[index];

	while (smallest_x < largest_x)
	{
		int next_smallest = largest_x;
		for (int i = 0; i < mser_boundingbox.size(); i++)
		{
			if (mser_boundingbox[i].x > smallest_x && mser_boundingbox[i].x < next_smallest)
			{
				next_smallest = mser_boundingbox[i].x;
				index = i;
			}
		}
		cout << characters[index];
		smallest_x = next_smallest;
	} 

	imshow("processed_image", morph);

	imshow("original", src);

	moveWindow("original", 10, 50);

	moveWindow("thresh", 30, 70);

	waitKey(0);

	return 0;

}

//Precondition: Need to have found keypoints for both images and have them stored in a vector.
//template_vector_descriptors stores the descriptors for all templates. feature_descriptor stores
//the descriptors for the current feature you are checking in original image.
//Postcondition: Returns the index of the template that has the best match
int best_template_match(Mat feature_descriptor, vector<Mat> template_vector_descriptors)
{
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	vector<vector<DMatch>> match_vector;
	double average_dist;
	double best_average = 100;
	int index_of_best = 0;

	for (int i = 0; i < template_vector_descriptors.size(); i++)
	{
		matcher.match(feature_descriptor, template_vector_descriptors[i], matches);
		match_vector.push_back(matches);
	}

	for (int i = 0; i < template_vector_descriptors.size(); i++)
	{
		average_dist = 0;
		for (int j = 0; j < match_vector[i].size(); j++)
		{
			average_dist += abs(match_vector[i][j].distance);
		}
		average_dist /= match_vector[i].size();
		

		if (average_dist < best_average)
		{
			best_average = average_dist;
			index_of_best = i;
		}
	}

	return index_of_best;
}

//precondition: Must have an empty vector of Mats to store the descriptors for each image in the image vector
//postcondition: Returns a vector of vectors containing of keypoints for all images in the image vector.
vector<Mat> find_descriptors(vector<Mat> image_vector)
{
	Ptr<KAZE> kz = KAZE::create();
	Mat template_descriptors;				//single template descriptors
	vector<Mat> template_descriptors_vector;
	vector<KeyPoint> template_keypoints;		//one templates keypoints
	//vector<vector<KeyPoint>> template_keypoints_vector;	//all template keypoints

	//Iterate through templates to find keypoints for each image
	for (int i = 0; i < image_vector.size(); i++)
	{
		kz->detectAndCompute(image_vector[i], noArray(), template_keypoints, template_descriptors);
		//template_keypoints_vector.push_back(template_keypoints);
		template_descriptors_vector.push_back(template_descriptors);
	}
	return template_descriptors_vector;
	//return template_keypoints_vector;
}

const char* getTextForEnum(int enumVal)
{
	static const char *EnumStrings[] = {
		"A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
		"K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
		"U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", 
		"4", "5", "6", "7", "8", "9"
	};
	return EnumStrings[enumVal];
}

/*bool compare_rect(const Rect & a, const Rect & b)
{
	return a.x < b.x;
}
*/
