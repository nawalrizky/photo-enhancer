#include "ai_filter_enhancer.h"

#include <algorithm>
#include <android/log.h>
#include <array>
#include <chrono>
#include <stdexcept>

namespace {
constexpr char LOG_TAG[] = "AiFilterEnhancer";
constexpr int FACE_SIZE = 512;
using Clock = std::chrono::steady_clock;

long elapsedMs(Clock::time_point started) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count();
}

const std::array<cv::Point2f, 5> REFERENCE_LANDMARKS = {{
    {192.9814f, 239.9470f},
    {318.9032f, 240.1436f},
    {256.6341f, 314.0193f},
    {201.2612f, 371.4104f},
    {313.0804f, 371.1511f},
}};
}  // namespace

AiFilterEngine::AiFilterEngine()
    : environment_(ORT_LOGGING_LEVEL_WARNING, LOG_TAG) {}

bool AiFilterEngine::initialize(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        const auto started = Clock::now();
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(8);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.DisableCpuMemArena();
        options.DisableMemPattern();
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        auto candidate = std::make_unique<Ort::Session>(
            environment_,
            modelPath.c_str(),
            options
        );
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = candidate->GetInputNameAllocated(0, allocator);
        auto outputName = candidate->GetOutputNameAllocated(0, allocator);

        inputName_ = inputName.get();
        outputName_ = outputName.get();
        session_ = std::move(candidate);
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "stage=session_init ms=%ld", elapsedMs(started));
        return true;
    } catch (const std::exception& error) {
        session_.reset();
        __android_log_print(
            ANDROID_LOG_ERROR,
            LOG_TAG,
            "Initialization failed: %s",
            error.what()
        );
        return false;
    }
}

cv::Mat AiFilterEngine::affineTransform(
    const std::vector<LandmarkPoint>& landmarks
) const {
    if (landmarks.size() != 5) return {};

    std::array<cv::Point2f, 5> source{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = {landmarks[i].x, landmarks[i].y};
    }
    return cv::estimateAffinePartial2D(source, REFERENCE_LANDMARKS);
}

cv::Mat AiFilterEngine::runAlignedFace(const cv::Mat& alignedFace) {
    const auto started = Clock::now();
    constexpr size_t spatialSize = FACE_SIZE * FACE_SIZE;
    std::vector<float> inputValues(3 * spatialSize);
    for (int y = 0; y < FACE_SIZE; ++y) {
        const auto* row = alignedFace.ptr<cv::Vec3b>(y);
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
    const auto inputReady = Clock::now();

    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );
    const auto inferenceReady = Clock::now();
    if (outputs.empty() || !outputs[0].IsTensor()) {
        throw std::runtime_error("Model returned no tensor output");
    }

    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 4 || shape[1] != 3 ||
        shape[2] != FACE_SIZE || shape[3] != FACE_SIZE) {
        throw std::runtime_error("Expected float32 output tensor [1,3,512,512]");
    }

    const float* output = outputs[0].GetTensorData<float>();
    cv::Mat enhanced(FACE_SIZE, FACE_SIZE, CV_8UC3);
    for (int y = 0; y < FACE_SIZE; ++y) {
        auto* row = enhanced.ptr<cv::Vec3b>(y);
        for (int x = 0; x < FACE_SIZE; ++x) {
            const size_t offset = static_cast<size_t>(y) * FACE_SIZE + x;
            const auto channel = [&](size_t index) {
                return static_cast<uint8_t>(
                    std::clamp((output[index] + 1.0f) * 127.5f, 0.0f, 255.0f)
                );
            };
            row[x] = {
                channel(2 * spatialSize + offset),
                channel(spatialSize + offset),
                channel(offset),
            };
        }
    }
    __android_log_print(
        ANDROID_LOG_INFO, LOG_TAG,
        "stage=face preprocess_ms=%ld inference_ms=%ld postprocess_ms=%ld",
        std::chrono::duration_cast<std::chrono::milliseconds>(inputReady - started).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(inferenceReady - inputReady).count(),
        elapsedMs(inferenceReady)
    );
    return enhanced;
}

cv::Mat AiFilterEngine::enhanceFull(
    const cv::Mat& inputBgr,
    const std::vector<LandmarkPoint>& landmarks
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (inputBgr.empty() || !session_ || landmarks.size() != 5) return {};
    const auto started = Clock::now();

    const cv::Mat affine = affineTransform(landmarks);
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
    const auto faceReady = Clock::now();

    cv::Mat inverse;
    cv::invertAffineTransform(affine, inverse);
    cv::Mat warpedFace;
    cv::warpAffine(enhanced, warpedFace, inverse, inputBgr.size(), cv::INTER_LINEAR);

    cv::Mat cropMask(FACE_SIZE, FACE_SIZE, CV_32FC1, cv::Scalar(0.0f));
    cv::ellipse(
        cropMask,
        {FACE_SIZE / 2, FACE_SIZE / 2},
        {220, 245},
        0,
        0,
        360,
        1.0f,
        -1
    );
    cv::GaussianBlur(cropMask, cropMask, {51, 51}, 0);
    cv::Mat warpedMask;
    cv::warpAffine(cropMask, warpedMask, inverse, inputBgr.size(), cv::INTER_LINEAR);

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
                    sourceRow[x][channel] * (1.0f - alpha) +
                    faceRow[x][channel] * alpha
                );
            }
        }
    }
    __android_log_print(
        ANDROID_LOG_INFO, LOG_TAG,
        "stage=full face_ms=%ld pasteback_ms=%ld total_ms=%ld",
        std::chrono::duration_cast<std::chrono::milliseconds>(faceReady - started).count(),
        elapsedMs(faceReady),
        elapsedMs(started)
    );
    return result;
}
