#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stack>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dict.hpp>

using namespace cv;
using namespace std;

Stitcher::Mode mode = Stitcher::PANORAMA;
RNG rng(12345);

Mat blurIM(Mat &source,const int &koof, const double &sigma) {
	Mat result = source;
	vector<double> weights(koof*koof,0);
	int pointx = -koof;
	int pointy =  koof;
	// calculating 1/(2sigma^2)
	double gausK = 1.00000 / (2.0000 * pow(sigma,2));
	// calculating 1/(2pi*sigma^2)
	double gausK_ = gausK / 3.14;
	double summ = 0;
	// calculating the gausian kernel wih readius koof
	for (int row = 0; row < koof; row++) {
		for (int col = 0; col < koof; col++) {
			weights[row*koof + col] = gausK_ * exp(-(pow(pointx, 2) + pow(pointy, 2))*gausK);
			summ += weights[row * koof + col];
			pointx += 1;
		}
		pointx = -koof;
		pointy -= 1;
	}
	for (int row = 0; row < koof; row++) {
		for (int col = 0; col < koof; col++) {
			weights[row * koof + col] /= summ;
		}
	}
	// bluring image
	std::cout << "Imsize:" << source.rows-koof << " / " << source.cols <<endl;
	for (int row = 0; row < source.rows; row++) {
		for (int col = 0; col < source.cols; col++) {
			// Mat(x,y) .* Gausian kernel (x,y)
			pointx = col-koof;
			pointy = row+koof;
			//std::cout << row-koof << " " << col+koof << endl;

			double Rsumm = 0, Gsumm = 0, Bsumm = 0;
			
			for (int row_ = 0; row_ < koof; row_++) {
				for (int col_ = 0; col_ < koof; col_++) {
					int pointy_ = pointy;
					if (pointy >= source.rows) {
						pointy = source.rows - (pointy - (source.rows - 1));
					}
					// B channel pixel revalue
					Bsumm += weights[row_ * koof + col_] * source.at<Vec3b>(abs(pointy), abs(pointx))[0];
					// G channel pixel revalue
					Gsumm += weights[row_ * koof + col_] * source.at<Vec3b>(abs(pointy), abs(pointx))[1];
					// R channel pixel revalue
					Rsumm += weights[row_ * koof + col_] * source.at<Vec3b>(abs(pointy), abs(pointx))[2];           
					if (pointx >= source.cols-1)pointx = pointx--;
					else pointx++;
					pointy = pointy_;
				}
			   pointy--;
			   pointx = col - koof;
			}
			result.at<Vec3b>(row, col)[0] = Bsumm;
			result.at<Vec3b>(row, col)[1] = Gsumm;
			result.at<Vec3b>(row, col)[2] = Rsumm;
		}
	}

	//namedWindow("Blured");
	//imshow("Blured", result);
	//imwrite("Blured_.jpg", result);
	//waitKey(0);
	return result;
}

//Gradient function
void Sobel(Mat &source) {
	Mat resultx = source;
	Mat resulty = source;
	Mat resultSumm = source;
	for (int row = 0; row < source.rows; row++) {
		for (int col = 0; col < source.cols-1; col++) {
		   //Finding derivatives
			resultx.at<uchar>(row, col) = (source.at<uchar>(row, col + 1) - source.at<uchar>(row, col));          
		}
	}
	namedWindow("X");
	imshow("X", resultx);
	imwrite("X.jpg", resultx);
	waitKey(0);
	//Y dirivative
	for (int row = 0; row < source.rows - 1; row++) {
		for (int col = 0; col < source.cols; col++) {
			resulty.at<uchar>(row, col) = (source.at<uchar>(row + 1, col) - source.at<uchar>(row, col));
	
		}
	}
	namedWindow("Y");
	imshow("Y", resulty);
	imwrite("Y.jpg", resulty );
	waitKey(0);
	//Mixing derivatives
	for (int row = 0; row < source.rows - 1; row++) {
		for (int col = 0; col < source.cols; col++) {
			resultSumm.at<uchar>(row, col) = (resultx.at<uchar>(row , col) + source.at<uchar>(row, col)) ;
		}
	}
	namedWindow("Summ");
	imshow("Z", resultSumm);
	imwrite("Summed.jpg", resultSumm);
	waitKey(0);
	return;
}
int stitch_(Mat &im1, Mat &im2) {

	Ptr<ORB> detector = cv::ORB::create();
	vector<KeyPoint> kp1, kp2;
	Mat des1, des2;
	detector->detectAndCompute(im1,Mat(),kp1,des1);
	detector->detectAndCompute(im2,Mat(),kp2,des2);
	Mat out;
	Mat im1_ = im1;
	Mat im2_ = im2;
	drawKeypoints(im1,kp1, im1_);
	drawKeypoints(im2, kp2, im2_);
	namedWindow("out1");
	namedWindow("out2");
	imshow("out1",im1_);
	imshow("out2", im2_);
	waitKey(0);
	vector<cv::DMatch > matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	matcher->match(des1, des2, matches);
	vector<Point2d> good_points1, good_points2;
	double max_dist = 0; double min_dist = 100;
	for (const auto& m : matches) {
		double dist = m.distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	for (const auto& m : matches)
	{
		if (m.distance <= 1.7*min_dist)
		{
			good_points1.push_back(kp1.at(m.queryIdx).pt);
			good_points2.push_back(kp2.at(m.trainIdx).pt);
		}
	}
	Rect croppImg1(0, 0, im1.cols, im1.rows);
	Rect croppImg2(0, 0, im2.cols, im2.rows);
	int imgWidth = im1.cols;
	int movementDirection = 0;
	for (int i = 0; i < good_points1.size(); ++i)
	{
		if (good_points1[i].x < imgWidth)
		{
			croppImg1.width = good_points1.at(i).x;
			croppImg2.x = good_points2[i].x;
			croppImg2.width = im2.cols - croppImg2.x;
			movementDirection = good_points1[i].y - good_points2[i].y;
			imgWidth = good_points1[i].x;
		}
	}
	im1 = im1(croppImg1);
	im2 = im2(croppImg2);
	int maxHeight = im1.rows > im2.rows ? im1.rows : im2.rows;
	int maxWidth = im1.cols + im2.cols;
	Mat result = Mat::zeros(cv::Size(maxWidth, maxHeight + abs(movementDirection)), CV_8UC3);
	if (movementDirection > 0)
	{
		cv::Mat half1(result,Rect(0, 0, im1.cols, im1.rows));
		im1.copyTo(half1);
		cv::Mat half2(result, Rect(im1.cols, abs(movementDirection), im2.cols, im2.rows));
		im2.copyTo(half2);
	}
	else
	{
		Mat half1(result, Rect(0, abs(movementDirection), im1.cols, im1.rows));
		im1.copyTo(half1);
		Mat half2(result, Rect(im1.cols, 0, im2.cols, im2.rows));
		im2.copyTo(half2);
	}
	imshow("Stitched Image", result);

	waitKey(0);
	return 1;
}

void rotatePoint(Point& p,const  Point& center, double angle) {
	Point new_p;

	new_p.x = cos(angle) * (p.x - center.x) - sin(angle) * (p.y - center.y) + center.x;
	new_p.y = sin(angle) * (p.x - center.x) + cos(angle) * (p.y - center.y) + center.y;
	p = new_p;
};



void contours(Mat &input){
	Mat canny ,threshold_=input;
	//Creating a countur of the image
	Canny(input,canny,50,140);
	blur(canny, canny, Size(2,2));
	namedWindow("Result");
	imshow("Result", canny);
	threshold(threshold_,threshold_,100,255,THRESH_BINARY);
	namedWindow("Threshold");
	imshow("Threshold", threshold_);
	waitKey(0);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > shapes;
	findContours(threshold_, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//Approximation wtih shapes

	/*Mat angles = Mat::zeros(threshold_.size(), CV_32FC1);
	cornerHarris(threshold_,angles,10,3,0.1);
	normalize(angles, angles,0,255, NORM_MINMAX, CV_32FC1, Mat());

	cout << "AAA";*/
	cout << "AAA";
	cvtColor(threshold_, threshold_, COLOR_GRAY2BGR);
	vector<vector<Point>> boxes_;
	//Converting contours into shapes
	int sh = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		vector<Point> shape;
		approxPolyDP(contours[i], shape, 5, true);
		shapes.push_back(shape);
		vector<Point> box;
		Point2f box_p[4];
		minAreaRect(shape).points(box_p);
		for (int j = 0; j < 4; j++) {
			box.push_back(box_p[j]);
		}
		boxes_.push_back(box);
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(threshold_, shapes, i, color, 2, LINE_8);
		drawContours(threshold_, boxes_, i, color, 2, LINE_8);
		for (int j = 0; j < shapes[i].size(); j++) {
			circle(threshold_,shapes[i][j],2,Scalar(256,0,256),2);
		}
	}

	cout << "AAA";
	imshow("Threshold", threshold_);
	waitKey(0);
	//taking the first shape 

	auto shapes_ = shapes;
	sort(shapes.begin(), shapes.end(), [](const vector<Point> shape1, const vector<Point> shape2)->bool {
		return contourArea(shape1) > contourArea(shape2);
		});
	double len_ = -1;
	vector<RotatedRect> boxes;
	for (int i =0; i < shapes.size();i++) {
		auto box = minAreaRect(shapes[i]);
		boxes.push_back(box);
		Point2f core_point[4];
		box.points(core_point);
		int min_x = 0;
		int min_y = 0;
		double angle = box.angle;
		if (angle > 45) {
			angle = 90 - angle;
			angle = -angle;
		}
		cout << "Angle " << angle<< endl;
		for (int j = 0; j < shapes[i].size(); j++) {
			rotatePoint(shapes[i][j], core_point[0], -angle * CV_PI / 180);
		}
		box = minAreaRect(shapes[i]);
		box.points(core_point);
		Point2f corner = core_point[1];
		for (int j = 0; j < shapes[i].size(); j++) {
			shapes[i][j].x -= corner.x;
			shapes[i][j].y -= corner.y;
			if (shapes[i][j].x < 0 && shapes[i][j].x < min_x)min_x = shapes[i][j].x;
			if (shapes[i][j].y < 0 && shapes[i][j].y < min_y)min_y = shapes[i][j].y;
			cout << "new x " << shapes[i][j].x << endl;
			cout << "new y " << shapes[i][j].y << endl;
		}
		for (int j = 0; j < shapes[i].size(); j++) {
			shapes[i][j].x -= min_x;
			shapes[i][j].y -= min_y;
		}
		auto box_ = boundingRect(shapes[i]);
		Mat show;
		show = Mat::zeros(Size2d(1000,1000),CV_32FC1);
		Scalar color = Scalar(256, 256, 256);
		drawContours(show, shapes, i, color, 2, LINE_8);
		namedWindow("shape" + to_string(i));
		imshow("shape"+to_string(i),show);
		waitKey(0);
	}
	Mat show;
	show = Mat::zeros(Size2d(1000, 1000), CV_32FC1);
	vector<int> matched;
	vector<vector<double>> assembly_algorythm;
	vector<Point> core = shapes[0];
	auto core_moment = moments(core); 
	while (shapes.size()!=1) {
		int matched_shape = 0;
		int corner_matches = 0;
		double rect = 10000000;
		int matches =  0;
		int core_point = 0;
		int matched_point = 0;
		double angle = 0;
		auto core_moments = moments(core);
		Point core_center = { int(core_moments.m10 / core_moments.m00),int(core_moments.m01 / core_moments.m00) };
		double core_area = contourArea(core);
		for (int i = 1; i < shapes.size(); i++) {
			for (int j = 0; j < core.size(); j++) {		
				for (double a = 0; a < CV_PI * 2; a += CV_PI / 100) {
					cout << "angle" << a << endl;
					for (int k = 0; k < shapes[i].size(); k++) {
					auto shape_ = shapes[i];
					Moments moments_ = moments(shape_);
					Point center = { int(moments_.m10 / moments_.m00),int(moments_.m01 / moments_.m00) };
						for (int I = 0; I < shape_.size(); I++) {
							rotatePoint(shape_[I], center, a);
						}
						
						Point2f move_vec = core[j] - shape_[k];
						for (int I = 0; I < shape_.size(); I++) {
							shape_[I].x += move_vec.x;
							shape_[I].y += move_vec.y;
						}
						moments_ = moments(shape_);
						center = { int(moments_.m10 / moments_.m00),int(moments_.m01 / moments_.m00) };
						bool intersect = false;
						Mat img1 = Mat::zeros(Size2d(1000, 1000), CV_8UC1);
						Mat img2 = Mat::zeros(Size2d(1000, 1000), CV_8UC1);
						fillPoly(img1, vector<vector<Point>>() = { core }, Scalar(256));
						fillPoly(img2, vector<vector<Point>>() = { shape_ }, Scalar(256));
						Mat intersection;
						bitwise_and(img1, img2, intersection);
						vector<vector<Point>> intersection_contour;
						double intersection_area=0;
						findContours(intersection,intersection_contour,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
						for (int L = 0; L < intersection_contour.size(); L++) {
							intersection_area += contourArea(intersection_contour[L]);
						}
						if (intersection_area > 200)intersect = true;
						/*for (int I = 0; I < shape_.size(); I++) {
							if ( pointPolygonTest(core, shape_[I], true) > 2)intersect = true;
						}/*
						for (int J = 0; J <core.size(); J++) {
							if (pointPolygonTest(shape_, core[J], true) > 0 && pointPolygonTest(shape_, core[J], true) > 3)intersect = true;
						}*/
					
						if (!intersect ) {

							int current_matches = 0;
							for (int I = 0; I < shape_.size(); I++) {
								for (int J = 0; J < core.size(); J++) {
									if (norm(Mat(shape_[I]) - Mat(core[J])) <= 2) {
										current_matches++;
									}
								}	
							}	
							if (current_matches > matches) {
								show = Mat::zeros(Size2d(1000, 1000), CV_32FC1);
								drawContours(show, vector<vector<Point>>() = { shape_
									}, -1, Scalar(256, 0, 256), 2, LINE_8);
								drawContours(show, vector<vector<Point>>() = { core }, 0, Scalar(256), 2, LINE_8);
								fillPoly(show,vector<vector<Point>>() = { shape_
									},Scalar(256));
								cout << i << " " << j << " " << a << "  "<< k << " " << current_matches << endl;
								namedWindow("test1");
								imshow("test1", show);

								waitKey(1);
								
								matches = current_matches;
								matched_shape = i;
								core_point = j;

								matched_point = k;
								angle = a;
							}
							else if (current_matches == matches) {
								corner_matches = 0;
							}
						}
					};
				}
			}
		}

		cout << "Matched shape " << matched_shape << "by point " << matched_point << "to point"<< core_point <<" angle " << angle << " matches" << matches << endl;
		Mat drawing_ = Mat::zeros(Size2d(1000, 1000), CV_8UC1);
		cvtColor(drawing_, drawing_, COLOR_GRAY2BGR);
		drawContours(drawing_,shapes, matched_shape, Scalar(256, 0, 256), 2, LINE_8);
		namedWindow("test2");
		imshow("test2", drawing_);
		waitKey(1);
		vector<double> assembly = {double(matched_shape),double(core_point),angle};
		assembly_algorythm.push_back(assembly);
		Point move_vec_ = core[core_point] - shapes[matched_shape][matched_point];
		int min_x = 0;
		int min_y = 0;
		Moments moments_ = moments(shapes[matched_shape]);
		Point center = { int(moments_.m10 / moments_.m00),int(moments_.m01 / moments_.m00) };
		for (int I = 0; I < shapes[matched_shape].size(); I++) {
			rotatePoint(shapes[matched_shape][I], center, angle);
		}
		drawing_ = Mat::zeros(Size2d(1000, 1000), CV_8UC1);
		cvtColor(drawing_, drawing_, COLOR_GRAY2BGR);
		drawContours(drawing_, shapes, matched_shape, Scalar(0, 256, 256), 2, LINE_8);
		namedWindow("test2");
		imshow("test2", drawing_);
		waitKey(1);
		move_vec_ = core[core_point] - shapes[matched_shape][matched_point];
		for (int I = 0; I < shapes[matched_shape].size(); I++) {
			shapes[matched_shape][I] += move_vec_;
			if (shapes[matched_shape][I].x < min_x) min_x = shapes[matched_shape][I].x;
			if (shapes[matched_shape][I].y < min_y) min_y = shapes[matched_shape][I].y;
		}
		for (int I = 0; I < shapes[matched_shape].size(); I++) {
			shapes[matched_shape][I] -= Point(min_x,min_y);
		}
		for (int J = 0; J < core.size(); J++) {
			core[J] -= Point(min_x, min_y);
		}
		cout << "Matched shape " << matched_shape << "by point " << matched_point << "to point"<< core_point <<" angle " << angle << " matches" << matches << endl;
		Mat drawing = Mat::zeros(Size2d(1000,1000),CV_8UC1);
		vector<vector<Point>> new_core = {core,shapes[matched_shape]};
		fillPoly(drawing,new_core,Scalar(256));
		blur(drawing,drawing,Size(4,4));
		try {
			findContours(drawing, new_core, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		}
		catch (Exception ex) { ex.what(); };
		approxPolyDP(new_core[0],core,2,true);
		vector < vector<Point> >temp ;
		temp.push_back(core);
		cvtColor(drawing, drawing, COLOR_GRAY2BGR);
		drawContours(drawing,temp,-1, Scalar(256, 0, 256),2,LINE_8);
		namedWindow("test");
		imshow("test", drawing);
		waitKey(0);
	
		shapes.erase(next(shapes.begin(), matched_shape));
		shapes.erase(shapes.begin());
		shapes.insert(shapes.begin(),core);
	}
	return;
}

int main()
{
	Mat image = imread("Hatkid.jpg", IMREAD_COLOR);
	//namedWindow("test1");
	//imshow("test1", image);
	//image = blurIM(image, 10, 10);
	//Mat gray;
	//cvtColor(image, gray,COLOR_RGB2GRAY);
	//Sobel(gray);
  Mat im1 = imread("1.jpg");
	Mat im2 = imread("2.jpg");
	resize(im1, im1, im1.size() / 2);
	resize(im2, im2, im2.size() / 2);
	stitch_(im1,im2);
	/*Mat testim = imread("Torn5.jpg",IMREAD_COLOR);
	resize(testim,testim,testim.size()/2,testim.cols/2,testim.rows/2);
	testim = blurIM(testim, 2, 2);
	cout << "blur fail" << endl;
	cvtColor(testim, testim, COLOR_RGB2GRAY);
	contours(testim);*/
	return 0;
}
