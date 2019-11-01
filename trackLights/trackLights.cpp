
//---------------------------------【头文件、命名空间包含部分】-------------------------------
//		描述：包含程序所使用的头文件和命名空间
//-------------------------------------------------------------------------------------------------
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <iostream>
using namespace std;
using namespace cv;

//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define POINT_DIST(p1,p2) std::sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y))

int extractId(String a)
{
	string a_ = a;	
	string src_ = "D:\\theThirdYear\\RM\\task1\\imageDataset\\";
	int pos_a = a_.find(src_);	
	int n = src_.size();
	a_ = a_.erase(pos_a, n);	

	string tail = ".jpg";
	pos_a = a_.find(tail);	
	n = src_.size();
	a_ = a_.erase(pos_a, n);	

	stringstream ss;
	int A;
	ss << a_;
	ss >> A;	

	return A;
}

//---------------------------------【图像文件重编号】------------------------------------------
/*
bool cmp(String a, String b)
{
	if (a.length() != b.length())
		return a.length() < b.length();
	else
	{
		int A = extractId(a);
		int B = extractId(b);
		return A < B;
	}
}

void renameImage()
{
	Mat image;
	String src = "D:\\theThirdYear\\RM\\task1\\imageDataset\\";
	String dst = "D:/theThirdYear/RM/task1/image/";

	vector<String> names;
	char name[10];
	glob(src, names, false);
	sort(names.begin(), names.end(),cmp);

	//for (int i = 0; i < names.size(); i++)
	//{
	// 	cout << names[i] << endl;
	//}
	//waitKey();

	for (int i = 0; i < names.size(); i++)
	{
		image = imread(names[i]);
		if (image.data != NULL)
		{
			sprintf_s(name, "%04d.jpg", i);
			imwrite(dst + name, image);
			cout << name << endl;
		}
		else {
			cout << "end of reading" << endl;
		}

	}
}
*/


//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
const int hmin_max = 360;
//int hmin = 120;
int hmin = 0;

const int hmax_max = 360;
//int hmax = 163;
int hmax = 360;

//int smin = 34;
int smin = 0;
const int smin_max = 255;

int smax = 103;
//int smax = 40;
//int smax = 255;
const int smax_max = 255;

//int vmin = 210;
int vmin = 220;
const int vmin_max = 255;

int vmax = 255;
const int vmax_max = 255;

int thresholds = 10;

int frame = 0;

Mat img;  
Mat hsv_img; 
Mat gray_img;
Mat bin_img;
Mat bin_img_hsv;   
Mat bin_img_rgb;
Mat show_img;

//-----------------------------------【putFrame( )函数】--------------------------------
//		描述：在图像窗口输出帧数
//------------------------------------------------------------------------------------------
Mat putFrame(Mat &image, int frame)
{
	Mat img = image;
	// 在图像窗口输出帧数
	char id[10];
	sprintf_s(id, "%04d", frame++);
	string text = "Frame: ";
	text = text + id;
	// 设置文本参数
	int font_face = FONT_HERSHEY_COMPLEX;
	double font_scale = 1;
	int thickness = 2;
	int baseline;
	// 获取文本框长宽
	Size text_size = getTextSize(text, font_face, font_scale, thickness, &baseline);
	Point origin;
	origin.x = 0;
	origin.y = text_size.height;
	putText(img, text, origin, font_face, font_scale, Scalar(255, 255, 255), thickness, 8, 0);
	return img;
}


//-----------------------------------【on_Trackbar( )函数】--------------------------------
//		描述：响应滑动条的回调函数
//------------------------------------------------------------------------------------------
void on_Trackbar(int, void*)
{
	// 0-144
	//cv::inRange(hsv_img, cv::Scalar(hmin, smin, vmin), cv::Scalar(hmax, 103, 255), bin_img);
	// 145
	cv::inRange(hsv_img, cv::Scalar(hmin, smin, vmin), cv::Scalar(hmax, smax, vmax), bin_img_hsv);
	show_img = putFrame(bin_img_hsv, frame);
	cv::imshow("binary image", show_img);
}

void cannyTrack(int, void*)
{
	Mat result;
	Canny(bin_img_hsv, result, thresholds, thresholds * 3, 3);
	//imshow("edge", result);
}

//------------------------------------【方法一：HSV空间】----------------------------------
//                             描述：用滑动条调节参数到适合的范围
//---------------------------------------------------------------------------------------
void HSVMethod(cv::VideoCapture Sequence)
{
	//创建窗体
	namedWindow("Trackbar", 1);
	namedWindow("binary image", 1);
	//在创建的窗体中创建一个滑动条控件	
	createTrackbar("hmin", "Trackbar", &hmin, hmin_max, on_Trackbar);
	createTrackbar("hmax", "Trackbar", &hmax, hmax_max, on_Trackbar);
	createTrackbar("smin", "Trackbar", &smin, smin_max, on_Trackbar);
	createTrackbar("smax", "Trackbar", &smax, smax_max, on_Trackbar);
	createTrackbar("vmin", "Trackbar", &vmin, vmin_max, on_Trackbar);
	createTrackbar("vmax", "Trackbar", &vmax, vmax_max, on_Trackbar);

	//Canny算子提取边缘
	//namedWindow("edge", 1);
	//createTrackbar("thresholds", "edge", &thresholds, 100, cannyTrack);

//#define SHOW_BIN_IMG
	//int frame = 0;
	while (1)
	{
		Sequence >> img;
		if (img.empty())
		{
			std::cout << "End of Sequence" << std::endl;
			break;
		}

		//cv::waitKey();
		hsv_img = cv::Mat(img.size(), img.type());
		cv::cvtColor(img, hsv_img, CV_BGR2HSV);
		//cv::imshow("hsv", hsv_img);

		//结果在回调函数中显示
		on_Trackbar(hmin, 0);
		on_Trackbar(hmax, 0);
		on_Trackbar(smin, 0);
		on_Trackbar(smax, 0);
		on_Trackbar(vmin, 0);
		on_Trackbar(vmax, 0);

		//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		//cv::dilate(bin_img_hsv, bin_img_hsv, element, cv::Point(-1, -1), 1);

		//cannyTrack(thresholds, 0);

#ifdef SHOW_BIN_IMG
		// 显示二值化图片
		show_img = putFrame(bin_img_hsv, frame);
		cv::imshow("binary image", show_img);
#endif
		frame++;
		cv::waitKey();
	}
}

//---------------------------------------【方法二：RGB空间】-------------------------------
//                                     描述：分别考虑颜色和亮度
//---------------------------------------------------------------------------------------	
void RGBMethod(std::vector<cv::RotatedRect> &lights)
{
//#define SHOW_IMG
//#define SHOW_SUBTRACT_IMG
//#define SHOW_GRAY_IMG
//#define SHOW_COLOR_IMG
//#define SHOW_LIGHT_IMG
//#define SHOW_BIN_IMG
#define	DRAW_LIGHT_RECT

		cv::Mat binary_color_img;
		cv::Mat binary_light_img;
		cv::Mat subtract_img;

		//------------考虑颜色------------
		std::vector<cv::Mat> bgr_channel;
		cv::split(img, bgr_channel);
		cv::subtract(bgr_channel[0], bgr_channel[1], subtract_img);
		//cv::cvtColor(img, hsv_img, CV_BGR2HSV);
		float thresh = 70;
		cv::threshold(subtract_img, binary_color_img, thresh, 255, CV_THRESH_BINARY);

#ifdef SHOW_SUBTRACT_IMG
		// 显示相减结果
		show_img = putFrame(subtract_img, frame);
		cv::imshow("subtract image", show_img);
#endif

#ifdef SHOW_IMG
		// 显示原始图片
		show_img = putFrame(img, frame);
		cv::imshow("image", show_img);
#endif

#ifdef SHOW_COLOR_IMG
		// 显示颜色二值化图片
		show_img = putFrame(binary_color_img, frame);
		cv::imshow("binary color image", show_img);
#endif

		//-------------考虑亮度--------------
		cv::cvtColor(img, gray_img, CV_RGB2GRAY);
		cv::threshold(gray_img, binary_light_img, 150, 255, CV_THRESH_BINARY);

#ifdef SHOW_LIGHT_IMG
		// 显示颜色二值化图片
		show_img = putFrame(binary_light_img, frame);
		cv::imshow("binary light image", show_img);
#endif

		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::dilate(binary_color_img, binary_color_img, element, cv::Point(-1, -1), 1);
		bin_img_rgb = binary_color_img & binary_light_img;

#ifdef SHOW_BIN_IMG
		// 显示二值化图片
		show_img = putFrame(bin_img_rgb, frame);
		cv::imshow("binary image", show_img);
#endif

		
		std::vector<std::vector<cv::Point>> contours_light;
		cv::findContours(bin_img_rgb, contours_light, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		
		std::vector<std::vector<cv::Point>> contours_brightness;
		cv::findContours(binary_light_img, contours_brightness, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		lights.reserve(contours_brightness.size());

		std::vector<int> is_processes(contours_brightness.size());
		for (unsigned int i = 0; i < contours_light.size(); ++i) {
			for (unsigned int j = 0; j < contours_brightness.size(); ++j) {
				if (!is_processes[j]) {
					if (cv::pointPolygonTest(contours_brightness[j], contours_light[i][0], true) >= 0.0) {
						cv::RotatedRect single_light = cv::minAreaRect(contours_brightness[j]);
						lights.push_back(single_light);
						is_processes[j] = true;
						break;
					}
				}
			} 
		} 

		//cout << lights.size() << endl;

}

//--------------------------------------【detectLights( )函数】-----------------------------------------
//		描述：检测灯条
//-----------------------------------------------------------------------------------------------
//#define HSV
#define RBG
#define DRAW_CONTOURS_RGB
void detectLights(std::vector<cv::RotatedRect> &lights)
{
#ifdef HSV
	//HSVMethod(seq1);
	bin_img = bin_img_hsv;
#endif

#ifdef RBG	
	RGBMethod(lights);
	bin_img = bin_img_rgb;
#endif

	//// auto contours_light = FindContours(binary_light_img);
	//std::vector<std::vector<cv::Point>> contours_hsv;
	//vector<Vec4i> hierarchy_hsv;
	//cv::findContours(bin_img_hsv, contours_hsv, hierarchy_hsv, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//// auto contours_brightness = FindContours(binary_brightness_img);
	//std::vector<std::vector<cv::Point>> contours_rgb;
	//vector<Vec4i> hierarchy_rgb;
	//cv::findContours(bin_img_rgb, contours_rgb, hierarchy_rgb, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

//#ifdef DRAW_CONTOURS_HSV
//	show_img = bin_img_hsv;
//	for (int i = 0;i < contours_hsv.size();i++)
//	{
//		//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数  
//		for (int j = 0;j < contours_hsv[i].size();j++)
//		{
//			//绘制出contours向量内所有的像素点  
//			Point P = Point(contours_hsv[i][j].x, contours_hsv[i][j].y);			
//		}
//		//绘制轮廓  
//		drawContours(show_img, contours_hsv, i, Scalar(255), 1, 8, hierarchy_hsv);
//	}
//	cv::imshow("binary image with contours", show_img);
//	waitKey();
//#endif

//#ifdef DRAW_CONTOURS_RGB
//	show_img = img;
//	for (int i = 0;i < contours_rgb.size();i++)
//	{
//		//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数  
//		for (int j = 0;j < contours_rgb[i].size();j++)
//		{
//			//绘制出contours向量内所有的像素点  
//			Point P = Point(contours_rgb[i][j].x, contours_rgb[i][j].y);
//		}
//		//绘制轮廓  
//		drawContours(show_img, contours_rgb, i, Scalar(0,255,0), 1, 8, hierarchy_rgb);
//	}
//	cv::imshow("binary image with contours", show_img);
//	waitKey();
//#endif

	//lights.reserve(contours_rgb.size());
	//// TODO: To be optimized
	//std::vector<int> is_processes(contours_rgb.size());
	//for (unsigned int i = 0; i < contours_hsv.size(); ++i) {
	//	for (unsigned int j = 0; j < contours_rgb.size(); ++j) {
	//		if (!is_processes[j]) {
	//			if (cv::pointPolygonTest(contours_rgb[j], contours_hsv[i][0], true) >= 0.0) {
	//				cv::RotatedRect single_light = cv::minAreaRect(contours_rgb[j]);
	//				lights.push_back(single_light);
	//				is_processes[j] = true;
	//				break;
	//			}
	//		}
	//	} // for j loop
	//} // for i loop

//#ifdef DRAW_LIGHT_RECT
//	//show_img = putFrame(img, frame);
//	show_img = img;
//	for (int i = 0; i < lights.size(); i++)
//	{
//		Point2f vertices[4];
//		lights[i].points(vertices);
//		for (int i = 0; i < 4; i++)
//			line(show_img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255));//白色矩形		
//	}
//	cv::imshow("image with rect", show_img);
//#endif

}

//---------------------------------------【adjustRect( )函数】-------------------------------
//                                     描述：调整为标准矩形
//---------------------------------------------------------------------------------------	

void adjustRect(cv::RotatedRect &rect)
{
	if (rect.size.width > rect.size.height)
	{
		auto temp = rect.size.height;
		rect.size.height = rect.size.width;
		rect.size.width = temp;
		rect.angle += 90;
		if (rect.angle > 180)
			rect.angle -= 180;
	}

	if (rect.angle > 90)
		rect.angle -= 90;
	else if (rect.angle < -90)
		rect.angle += 90;   // 左灯条角度为负, 右灯条角度为正
}

//--------------------------------------【filterLights( )函数】-----------------------------------------
//		描述：
//-----------------------------------------------------------------------------------------------
void choseLights(std::vector<cv::RotatedRect> &lights)
{
	std::vector<cv::RotatedRect> light_rects;
	light_rects.clear();
	for (uchar i = 0; i < lights.size(); i++) {
		adjustRect(lights[i]);
	}
	for (const auto &armor_light : lights)
	{
		auto rect = std::minmax(armor_light.size.width, armor_light.size.height);
		auto width_height = rect.second / rect.first;
		auto angle = armor_light.angle;

		if (armor_light.size.area() >= 20
			&& armor_light.size.area() < 0.04 * img.size().height * img.size().width
			&& abs(angle) < 35) 
		{
			light_rects.push_back(armor_light);
		}

	}
	lights = light_rects;
}

cv::RotatedRect boundingRRect(const cv::RotatedRect & left, const cv::RotatedRect & right) {
	const Point & pl = left.center, &pr = right.center;
	Point2f center;
	center.x = (pl.x + pr.x) / 2.0;
	center.y = (pl.y + pr.y) / 2.0;
	cv::Size2f wh_l = left.size;
	cv::Size2f wh_r = right.size;
	float width = POINT_DIST(pl, pr);
	if (frame >= 521 && frame < 699)
	{
		width = width - 15;
	}
	float height = 2*std::max(wh_l.height, wh_r.height);
	float angle = std::atan2(right.center.y - left.center.y, right.center.x - left.center.x);
	return RotatedRect(center, Size2f(width, height), angle * 180 / CV_PI);
}

void chooseArmors(std::vector<cv::RotatedRect> &lights, std::vector<cv::RotatedRect> &possible_armor)
{
	int armor_area_max = 18000;
	std::vector<cv::RotatedRect> rect;
	for (int i = 0; i < lights.size(); ++i)
	{
		for (int j = i; j < lights.size(); ++j)
		{
			auto rect1 = std::minmax(lights[i].size.width, lights[i].size.height);
			auto width_height1 = rect1.second / rect1.first;
			auto rect2 = std::minmax(lights[j].size.width, lights[j].size.height);
			auto width_height2 = rect2.second / rect2.first;

			auto angle_diff = abs(lights[i].angle - lights[j].angle);
			auto height_diff = abs(lights[i].size.height - lights[j].size.height) / std::max(lights[i].size.height, lights[j].size.height);
			auto width_diff = abs(lights[i].size.width - lights[j].size.width) / std::max(lights[i].size.width, lights[j].size.width);

			if (lights[i].angle * lights[j].angle >= 0            
				&& abs(angle_diff) < 30                            			   
				)
			{
				cv::RotatedRect possible_rect;
				if (1 < width_height1 && width_height1 <= 1.5
					&& 1 < width_height2 && width_height2 <= 1.5
					&& abs(width_height1 - width_height2) < 0.5)
				{
					if (abs(lights[i].angle) > 60 && abs(lights[j].angle) > 60 || abs(lights[i].angle) < 30 && abs(lights[j].angle) < 30)
					{
						if (lights[i].center.x < lights[j].center.x)
							possible_rect = boundingRRect(lights[i], lights[j]);
						else
							possible_rect = boundingRRect(lights[j], lights[i]);

						auto armor_ratio = possible_rect.size.width / possible_rect.size.height;
						auto armor_angle = possible_rect.angle;
						auto armor_area = possible_rect.size.area();
						auto armor_light_angle_diff = abs(armor_angle - lights[i].angle) + abs(armor_angle - lights[j].angle); 

						if (armor_area > 50
							&& armor_area < armor_area_max
							&& armor_ratio > 1 
							&& armor_ratio < 5   
							&& abs(armor_angle) < 35
							&& armor_light_angle_diff < 35
							)
						{						
							rect.push_back(possible_rect);
						}
					}
				}
				
				else if (1.5 < width_height1 
					&& 1.5 < width_height2 
					&& (lights[i].center.y + lights[i].size.height / 2) >(lights[j].center.y - lights[j].size.height / 2)
					&& (lights[j].center.y + lights[j].size.height / 2) >(lights[i].center.y - lights[i].size.height / 2)
					&& abs(lights[i].angle) < 30 && abs(lights[j].angle) < 30)
				{
					 if (lights[i].center.x < lights[j].center.x)
						possible_rect = boundingRRect(lights[i], lights[j]);
					 else
						possible_rect = boundingRRect(lights[j], lights[i]);

					 auto armor_ratio = possible_rect.size.width / possible_rect.size.height;
					 auto armor_angle = possible_rect.angle;
					 auto armor_area = possible_rect.size.area();
					 auto armor_light_angle_diff = abs(armor_angle - lights[i].angle) + abs(armor_angle - lights[j].angle); 

					 if (armor_area > 50
						&& armor_area < armor_area_max
						&& armor_ratio > 1 
						&& armor_ratio < 4.5 
						&& abs(armor_angle) < 35
						&& armor_light_angle_diff < 35) 
					 {
						rect.push_back(possible_rect);
					 }
				}
				

			} 
			else if (abs(angle_diff) < 35)    
			{
				cv::RotatedRect possible_rect;
				if (1 < width_height1 && width_height1 < 1.5
					&& 1 < width_height2 && width_height2 < 1.5
					&& abs(width_height1 - width_height2) < 1
					&& abs(lights[i].angle) < 30 
					&& abs(lights[j].angle) < 30)
				{
					if (lights[i].center.x < lights[j].center.x)
						possible_rect = boundingRRect(lights[i], lights[j]);
					else
						possible_rect = boundingRRect(lights[j], lights[i]);

					auto armor_ratio = possible_rect.size.width / possible_rect.size.height;
					auto armor_angle = possible_rect.angle;
					auto armor_area = possible_rect.size.area();
					auto armor_light_angle_diff = abs(armor_angle - lights[i].angle) + abs(armor_angle - lights[j].angle); 

					if (armor_area > 50
						&& armor_area < armor_area_max
						&& armor_ratio > 1 
						&& armor_ratio <5 
						&& abs(armor_angle) < 35
						&& armor_light_angle_diff < 35 
						)
					{
						rect.push_back(possible_rect);
					}
				}

				else if (1.5 < width_height1 
					&& 1.5 < width_height2 
					&& (lights[i].center.y + lights[i].size.height / 2) >(lights[j].center.y - lights[j].size.height / 2)
					&& (lights[j].center.y + lights[j].size.height / 2) >(lights[i].center.y - lights[i].size.height / 2))
				{
					if (lights[i].center.x < lights[j].center.x)
						possible_rect = boundingRRect(lights[i], lights[j]);
					else
						possible_rect = boundingRRect(lights[j], lights[i]);

					auto armor_ratio = possible_rect.size.width / possible_rect.size.height;
					auto armor_angle = possible_rect.angle;
					auto armor_area = possible_rect.size.area();
					auto armor_light_angle_diff = abs(armor_angle - lights[i].angle) + abs(armor_angle - lights[j].angle); 

					 if (armor_area > 50
						&& armor_area < armor_area_max
						&& armor_ratio > 1 
						&& armor_ratio < 4.5 
						&& abs(armor_angle) < 35
						&& armor_light_angle_diff < 35
						) 
						 rect.push_back(possible_rect);
					 
				}
			} 
			

		} 
	} 

	possible_armor = rect;
} 

bool makeRectSafe(cv::Rect & rect, cv::Size size) {
	if (rect.x < 0)
		rect.x = 0;
	if (rect.x + rect.width > size.width)
		rect.width = size.width - rect.x;
	if (rect.y < 0)
		rect.y = 0;
	if (rect.y + rect.height > size.height)
		rect.height = size.height - rect.y;
	if (rect.width <= 0 || rect.height <= 0)
		return false;
	return true;
}

int img_idx = 0;
void drawRotatedRectAndSave(const cv::Mat &image, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness)
{
	cv::Point2f vertex[4];

	rect.points(vertex);
	auto center = rect.center;
	cv::Mat rot_mat = cv::getRotationMatrix2D(rect.center, rect.angle, 1);
	cv::Mat rot_image;
	cv::Mat roi_image;
	cv::Mat gray_image;
	warpAffine(image, rot_image, rot_mat, rot_image.size(), INTER_LINEAR, BORDER_CONSTANT);  // warpAffine use 2ms
	cv::Rect target = cv::Rect(center.x - (rect.size.width / 2),
							   center.y - (rect.size.height / 2),
							   rect.size.width, rect.size.height);
	if (makeRectSafe(target, image.size()) == true)
	{
		roi_image = rot_image(target);
		cv::resize(roi_image, roi_image, cv::Size(80, 60));
		cv::cvtColor(roi_image, gray_image, CV_RGB2GRAY);
		cv::Mat output;
		cv::equalizeHist(gray_img, output);
		char str[100];
		sprintf_s(str, "D:\\theThirdYear\\RM\\task1\\hist\\%04d.jpg", img_idx++);
		cv::imwrite(str, output);
		//for (int i = 0; i < 4; i++)
		//	cv::line(image, vertex[i], vertex[(i + 1) % 4], color, thickness);
		//cv::imshow("rot", rot_image);
		//cv::imshow("roi", roi_image);
	}

}

//--------------------------------------【main( )函数】-----------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	std::string frame_file = "D:\\theThirdYear\\RM\\task1\\image\\%04d.jpg";
	cv::VideoCapture Sequence(frame_file);
	if (!Sequence.isOpened())
		std::cerr << "Failed to open image dataset" << std::endl;
	//cv::namedWindow("armor", 1);
	while (1)
	{
		Sequence >> img;
		if (frame < 521 && frame >= 699)
		{
			frame++;
			continue;
		}
		if (img.empty())
		{
			std::cout << "End of Sequence" << std::endl;
			break;
		}

		std::vector<cv::RotatedRect> lights;
		detectLights(lights);
		choseLights(lights);

		std::vector<cv::RotatedRect> possible_armors;

		if (lights.size() > 1)
		{
			chooseArmors(lights, possible_armors);			
			//Armors(possible_armors, frame);

			for (auto armor : possible_armors)
			{
				drawRotatedRectAndSave(img, armor, Scalar(0, 0, 255), 2);
				//if (frame > 500)
				//	cv::waitKey();
			}
			
			//cv::imshow("armor", img);
		}

		frame++;
		//std::cout << "frame " << frame << " done." << std::endl;
		//cv::waitKey();

	}
	//HSVMethod(Sequence);
	//bin_img = bin_img_hsv;
}
