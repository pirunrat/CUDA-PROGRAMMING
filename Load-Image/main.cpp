#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "✅ OpenCV test started." << std::endl;

    cv::Mat img = cv::imread("me.jpg");
    if (img.empty()) {
        std::cerr << "❌ Failed to load 'me.jpg'. Check file path and format." << std::endl;
        return -1;
    }

    std::cout << "✅ Image loaded successfully." << std::endl;

    cv::imshow("My Image", img);
    std::cout << "ℹ️ Press any key in the image window to close..." << std::endl;
    cv::waitKey(0);

    return 0;
}
