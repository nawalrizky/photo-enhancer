#include "native_enhancer.h"

#include <algorithm>
#include <android/log.h>
#include <array>
#include <cstdio>
#include <stdexcept>

#include <nnapi_provider_factory.h>

namespace {
constexpr char LOG_TAG[] = "PhotoEnhancerNative";
constexpr int FACE_SIZE = 512;

const std::array<cv::Point2f, 5> REFERENCE_LANDMARKS = {{
    {192.9814f, 239.9470f},
    {318.9032f, 240.1436f},
    {256.6341f, 314.0193f},
    {201.2612f, 371.4104f},
    {313.0804f, 371.1511f},
}};

void logError(const std::string& message) {
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "%s", message.c_str());
}

void logMemory(const char* stage) {
    FILE* status = std::fopen("/proc/self/status", "r");
    if (!status) return;
    long rssKb = 0;
    long highWaterKb = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), status)) {
        std::sscanf(line, "VmRSS: %ld kB", &rssKb);
        std::sscanf(line, "VmHWM: %ld kB", &highWaterKb);
    }
    std::fclose(status);
    __android_log_print(
        ANDROID_LOG_INFO,
        "EnhancerMem",
        "%s RSS=%ld KB HWM=%ld KB",
        stage,
        rssKb,
        highWaterKb
    );
}
}  // namespace

FaceEnhancerEngine::FaceEnhancerEngine()
    : env_(ORT_LOGGING_LEVEL_WARNING, LOG_TAG) {}

bool FaceEnhancerEngine::init(const std::string& modelPath, bool useNNAPI) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        logMemory("session_init_before");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.DisableCpuMemArena();
        options.DisableMemPattern();
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (useNNAPI) {
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(options, 0));
            } catch (const std::exception& error) {
                __android_log_print(
                    ANDROID_LOG_WARN,
                    LOG_TAG,
                    "NNAPI unavailable, using CPU: %s",
                    error.what()
                );
            }
        }

        auto candidate = std::make_unique<Ort::Session>(env_, modelPath.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = candidate->GetInputNameAllocated(0, allocator);
        auto outputName = candidate->GetOutputNameAllocated(0, allocator);

        inputName_ = inputName.get();
        outputName_ = outputName.get();
        session_ = std::move(candidate);
        logMemory("session_init_after");
        return true;
    } catch (const std::exception& error) {
        session_.reset();
        logError(std::string("Initialization failed: ") + error.what());
        return false;
    }
}

cv::Mat FaceEnhancerEngine::getAffineTransform512(
    const std::vector<LandmarkPoint>& landmarks
) const {
    if (landmarks.size() != 5) return {};

    std::array<cv::Point2f, 5> source{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = {landmarks[i].x, landmarks[i].y};
    }
    return cv::estimateAffinePartial2D(source, REFERENCE_LANDMARKS);
}

cv::Mat FaceEnhancerEngine::runAlignedFace(const cv::Mat& alignedFace512) {
    logMemory("inference_prepare_before");
    constexpr size_t spatialSize = FACE_SIZE * FACE_SIZE;
    std::vector<float> inputValues(3 * spatialSize);
    for (int y = 0; y < FACE_SIZE; ++y) {
        const auto* row = alignedFace512.ptr<cv::Vec3b>(y);
        for (int x = 0; x < FACE_SIZE; ++x) {
            const size_t offset = static_cast<size_t>(y) * FACE_SIZE + x;
            inputValues[offset] = row[x][2] / 127.5f - 1.0f;
            inputValues[spatialSize + offset] = row[x][1] / 127.5f - 1.0f;
            inputValues[2 * spatialSize + offset] = row[x][0] / 127.5f - 1.0f;
        }
    }

    const std::array<int64_t, 4> inputShape = {1, 3, FACE_SIZE, FACE_SIZE};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputValues.data(),
        inputValues.size(),
        inputShape.data(),
        inputShape.size()
    );

    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};
    logMemory("inference_run_before");
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );
    logMemory("inference_run_after");
    if (outputs.empty() || !outputs[0].IsTensor()) {
        throw std::runtime_error("Model returned no tensor output");
    }

    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 4 || shape[1] != 3 || shape[2] != FACE_SIZE || shape[3] != FACE_SIZE) {
        throw std::runtime_error("Expected float32 output tensor [1,3,512,512]");
    }

    const float* output = outputs[0].GetTensorData<float>();
    cv::Mat enhanced(FACE_SIZE, FACE_SIZE, CV_8UC3);
    for (int y = 0; y < FACE_SIZE; ++y) {
        auto* row = enhanced.ptr<cv::Vec3b>(y);
        for (int x = 0; x < FACE_SIZE; ++x) {
            const size_t offset = static_cast<size_t>(y) * FACE_SIZE + x;
            const auto channel = [&](size_t index) {
                return static_cast<uint8_t>(std::clamp((output[index] + 1.0f) * 127.5f, 0.0f, 255.0f));
            };
            row[x] = {
                channel(2 * spatialSize + offset),
                channel(spatialSize + offset),
                channel(offset),
            };
        }
    }
    logMemory("inference_output_ready");
    return enhanced;
}

cv::Mat FaceEnhancerEngine::enhanceCroppedFace(
    const cv::Mat& inputBgr,
    const std::vector<LandmarkPoint>& landmarks,
    int targetW,
    int targetH
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (inputBgr.empty() || !session_ || landmarks.size() != 5 || targetW <= 0 || targetH <= 0) {
        return {};
    }

    const cv::Mat affine = getAffineTransform512(landmarks);
    if (affine.empty()) return {};

    cv::Mat aligned;
    cv::warpAffine(
        inputBgr,
        aligned,
        affine,
        {FACE_SIZE, FACE_SIZE},
        cv::INTER_LINEAR,
        cv::BORDER_REFLECT_101
    );
    cv::Mat enhanced = runAlignedFace(aligned);
    if (targetW == FACE_SIZE && targetH == FACE_SIZE) return enhanced;

    cv::Mat resized;
    cv::resize(enhanced, resized, {targetW, targetH}, 0, 0, cv::INTER_AREA);
    return resized;
}

cv::Mat FaceEnhancerEngine::enhanceFullImage(
    const cv::Mat& inputBgr,
    const std::vector<LandmarkPoint>& landmarks
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (inputBgr.empty() || !session_ || landmarks.size() != 5) return {};
    logMemory("full_start");

    const cv::Mat affine = getAffineTransform512(landmarks);
    if (affine.empty()) return {};

    cv::Mat aligned;
    cv::warpAffine(
        inputBgr,
        aligned,
        affine,
        {FACE_SIZE, FACE_SIZE},
        cv::INTER_LINEAR,
        cv::BORDER_REFLECT_101
    );
    cv::Mat enhanced = runAlignedFace(aligned);
    logMemory("full_inference_complete");

    cv::Mat inverse;
    cv::invertAffineTransform(affine, inverse);
    cv::Mat warpedFace;
    cv::warpAffine(enhanced, warpedFace, inverse, inputBgr.size(), cv::INTER_LINEAR);
    logMemory("full_face_warped");

    cv::Mat cropMask(FACE_SIZE, FACE_SIZE, CV_32FC1, cv::Scalar(0.0f));
    cv::ellipse(cropMask, {FACE_SIZE / 2, FACE_SIZE / 2}, {220, 245}, 0, 0, 360, 1.0f, -1);
    cv::GaussianBlur(cropMask, cropMask, {51, 51}, 0);
    cv::Mat warpedMask;
    cv::warpAffine(cropMask, warpedMask, inverse, inputBgr.size(), cv::INTER_LINEAR);
    logMemory("full_mask_warped");

    cv::Mat result(inputBgr.size(), CV_8UC3);
    for (int y = 0; y < inputBgr.rows; ++y) {
        const auto* sourceRow = inputBgr.ptr<cv::Vec3b>(y);
        const auto* faceRow = warpedFace.ptr<cv::Vec3b>(y);
        const auto* maskRow = warpedMask.ptr<float>(y);
        auto* resultRow = result.ptr<cv::Vec3b>(y);
        for (int x = 0; x < inputBgr.cols; ++x) {
            const float alpha = maskRow[x];
            for (int channel = 0; channel < 3; ++channel) {
                resultRow[x][channel] = cv::saturate_cast<uint8_t>(
                    sourceRow[x][channel] * (1.0f - alpha) + faceRow[x][channel] * alpha
                );
            }
        }
    }
    logMemory("full_direct_blended");
    logMemory("full_result_ready");
    return result;
}
