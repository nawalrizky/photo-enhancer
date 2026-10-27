#ifndef NATIVE_ENHANCER_H
#define NATIVE_ENHANCER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

struct LandmarkPoint {
    float x;
    float y;
};

class FaceEnhancerEngine {
public:
    FaceEnhancerEngine();
    ~FaceEnhancerEngine();

    // Initialize ONNX Runtime Session with model file path (from Android assets)
    bool init(const std::string& modelPath, bool useGPU = true);

    // Crop, align face to 512x512, run ONNX face enhancer, and resize output to target dimensions for FR
    cv::Mat enhanceCroppedFace(const cv::Mat& inputBgr, const std::vector<LandmarkPoint>& landmarks, int targetW = 112, int targetH = 112);

    // Enhance face and paste back into original full resolution image
    cv::Mat enhanceFullImage(const cv::Mat& inputBgr, const std::vector<LandmarkPoint>& landmarks);

private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;

    std::string inputName_;
    std::string outputName_;

    // Helper functions
    cv::Mat getAffineTransform512(const std::vector<LandmarkPoint>& landmarks);
    cv::Mat preprocessInputTensor(const cv::Mat& alignedFace512);
    cv::Mat postprocessOutputTensor(const float* outputData, int width, int height);
};

#endif // NATIVE_ENHANCER_H
