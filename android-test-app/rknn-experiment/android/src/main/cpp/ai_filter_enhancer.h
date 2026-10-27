#pragma once
#include <mutex>
#include <string>
#include <vector>
#include <rknn_api.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct LandmarkPoint { float x; float y; };
class ExperimentalRknnEngine {
public:
    ExperimentalRknnEngine() = default;
    ~ExperimentalRknnEngine();
    bool initialize(const std::string& modelPath);
    cv::Mat enhanceFull(const cv::Mat& inputBgr, const std::vector<LandmarkPoint>& landmarks);
private:
    cv::Mat affineTransform(const std::vector<LandmarkPoint>& landmarks) const;
    cv::Mat runAlignedFace(const cv::Mat& alignedFace);
    rknn_context context_ = 0;
    std::mutex mutex_;
};
