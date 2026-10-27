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

class FaceEnhancerEngine {
public:
    FaceEnhancerEngine();
    bool init(const std::string& modelPath, bool useNNAPI);
    cv::Mat enhanceCroppedFace(
        const cv::Mat& inputBgr,
        const std::vector<LandmarkPoint>& landmarks,
        int targetW,
        int targetH
    );
    cv::Mat enhanceFullImage(
        const cv::Mat& inputBgr,
        const std::vector<LandmarkPoint>& landmarks
    );

private:
    cv::Mat getAffineTransform512(const std::vector<LandmarkPoint>& landmarks) const;
    cv::Mat runAlignedFace(const cv::Mat& alignedFace512);

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::string outputName_;
    std::mutex mutex_;
};
