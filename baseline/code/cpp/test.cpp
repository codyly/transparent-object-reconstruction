#include <opencv2/opencv.hpp>
#include <aruco/aruco.h>
int main(int argc, char** argv)
{
    cv::Mat img = cv::imread("/home/ren/Pictures/th.jpeg");
    cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Test", img);
    cv::waitKey(0);
    
    return 0;
}