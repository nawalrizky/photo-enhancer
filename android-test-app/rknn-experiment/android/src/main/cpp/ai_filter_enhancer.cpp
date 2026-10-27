#include "ai_filter_enhancer.h"

#include <algorithm>
#include <android/log.h>
#include <array>
#include <chrono>
#include <stdexcept>
#include <cmath>

namespace {
constexpr char LOG_TAG[] = "ExperimentalRknn";
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

ExperimentalRknnEngine::~ExperimentalRknnEngine() {
    if (context_) rknn_destroy(context_);
}

bool ExperimentalRknnEngine::initialize(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(mutex_);
    rknn_context candidate = 0;
    const auto started = Clock::now();
    try {
        if (rknn_init(&candidate, const_cast<char*>(modelPath.c_str()), 0, 0, nullptr) != 0)
            throw std::runtime_error("RKNN initialization failed");
        rknn_input_output_num counts{};
        rknn_tensor_attr input{}, output{};
        if (rknn_query(candidate, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts)) != 0 ||
            counts.n_input != 1 || counts.n_output != 1 ||
            rknn_query(candidate, RKNN_QUERY_NATIVE_INPUT_ATTR, &input, sizeof(input)) != 0 ||
            rknn_query(candidate, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)) != 0)
            throw std::runtime_error("RKNN tensor query failed");
        if (input.type != RKNN_TENSOR_FLOAT16 || input.fmt != RKNN_TENSOR_NHWC ||
            input.n_dims != 4 || input.dims[0] != 1 || input.dims[1] != 512 ||
            input.dims[2] != 512 || input.dims[3] != 3 || input.size_with_stride != 1572864 ||
            output.fmt != RKNN_TENSOR_NCHW || output.n_dims != 4 || output.dims[0] != 1 ||
            output.dims[1] != 3 || output.dims[2] != 512 || output.dims[3] != 512)
            throw std::runtime_error("Unsupported experimental RKNN tensor contract");
        if (context_) rknn_destroy(context_);
        context_ = candidate;
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "stage=session_init ms=%ld", elapsedMs(started));
        return true;
    } catch (const std::exception& error) {
        if (candidate) rknn_destroy(candidate);
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Initialization failed: %s", error.what());
        return false;
    }
}

cv::Mat ExperimentalRknnEngine::affineTransform(
    const std::vector<LandmarkPoint>& landmarks
) const {
    if (landmarks.size() != 5) return {};

    std::array<cv::Point2f, 5> source{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = {landmarks[i].x, landmarks[i].y};
    }
    return cv::estimateAffinePartial2D(source, REFERENCE_LANDMARKS);
}

cv::Mat ExperimentalRknnEngine::runAlignedFace(const cv::Mat& alignedFace) {
    const auto started = Clock::now();
    constexpr size_t spatialSize = FACE_SIZE * FACE_SIZE;
    std::vector<_Float16> inputValues(3 * spatialSize);
    for (int y = 0; y < FACE_SIZE; ++y) {
        const auto* row = alignedFace.ptr<cv::Vec3b>(y);
        for (int x = 0; x < FACE_SIZE; ++x) {
            const size_t offset = (static_cast<size_t>(y) * FACE_SIZE + x) * 3;
            for (int c = 0; c < 3; ++c)
                inputValues[offset+c] = static_cast<_Float16>(row[x][2-c] / 127.5f - 1.0f);
        }
    }
    const auto inputReady = Clock::now();
    rknn_input input{};
    input.buf = inputValues.data();
    input.size = inputValues.size() * sizeof(_Float16);
    input.pass_through = 1;
    input.type = RKNN_TENSOR_FLOAT16;
    input.fmt = RKNN_TENSOR_NHWC;
    if (rknn_inputs_set(context_, 1, &input) != 0 || rknn_run(context_, nullptr) != 0)
        throw std::runtime_error("RKNN inference failed");
    rknn_output output{};
    output.want_float = 1;
    if (rknn_outputs_get(context_, 1, &output, nullptr) != 0)
        throw std::runtime_error("RKNN output retrieval failed");
    struct Release {
        rknn_context context; rknn_output* output;
        ~Release() { rknn_outputs_release(context, 1, output); }
    } release{context_, &output};
    const auto inferenceReady = Clock::now();
    if (!output.buf || output.size != 3 * spatialSize * sizeof(float))
        throw std::runtime_error("Unexpected RKNN output size");
    const float* values = static_cast<const float*>(output.buf);
    cv::Mat enhanced(FACE_SIZE, FACE_SIZE, CV_8UC3);
    for (int y = 0; y < FACE_SIZE; ++y) {
        auto* row = enhanced.ptr<cv::Vec3b>(y);
        for (int x = 0; x < FACE_SIZE; ++x) {
            const size_t offset = static_cast<size_t>(y) * FACE_SIZE + x;
            const auto channel = [&](size_t index) {
                if (!std::isfinite(values[index])) throw std::runtime_error("Nonfinite RKNN output");
                return static_cast<uint8_t>(std::clamp((values[index]+1.0f)*127.5f, 0.0f, 255.0f));
            };
            row[x] = {channel(2*spatialSize+offset), channel(spatialSize+offset), channel(offset)};
        }
    }
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
        "stage=face preprocess_ms=%ld inference_ms=%ld postprocess_ms=%ld",
        std::chrono::duration_cast<std::chrono::milliseconds>(inputReady-started).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(inferenceReady-inputReady).count(),
        elapsedMs(inferenceReady));
    return enhanced;
}

cv::Mat ExperimentalRknnEngine::enhanceFull(
    const cv::Mat& inputBgr,
    const std::vector<LandmarkPoint>& landmarks
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (inputBgr.empty() || !context_ || landmarks.size() != 5) return {};
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
