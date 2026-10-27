#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct LandmarkPoint {
    float x;
    float y;
};

class AiFilterEngine {
public:
    AiFilterEngine();
    bool initialize(const std::string& modelPath);
    cv::Mat enhanceFull(
        const cv::Mat& inputBgr,
        const std::vector<LandmarkPoint>& landmarks
    );

private:
    cv::Mat affineTransform(const std::vector<LandmarkPoint>& landmarks) const;
    cv::Mat runAlignedFace(const cv::Mat& alignedFace);

    Ort::Env environment_;
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::string outputName_;
    std::mutex mutex_;
};
