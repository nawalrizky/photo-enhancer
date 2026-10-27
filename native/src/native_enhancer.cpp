#include "native_enhancer.h"
#include <iostream>
#include <vector>
#include <algorithm>

FaceEnhancerEngine::FaceEnhancerEngine()
    : env_(ORT_LOGGING_LEVEL_WARNING, "PhotoEnhancerNative") {
}

FaceEnhancerEngine::~FaceEnhancerEngine() {
}

bool FaceEnhancerEngine::init(const std::string& modelPath, bool useGPU) {
    try {
        sessionOptions_.SetIntraOpNumThreads(4);
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#if defined(__ANDROID__)
        if (useGPU) {
            // Enable NNAPI Execution Provider for hardware acceleration on Android devices
            uint32_t nnapi_flags = 0;
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions_, nnapi_flags));
        }
#endif

        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr inputNameAlloc = session_->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr outputNameAlloc = session_->GetOutputNameAllocated(0, allocator);

        inputName_ = inputNameAlloc.get();
        outputName_ = outputNameAlloc.get();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Native Enhancer Init Failed: " << e.what() << std::endl;
        return false;
    }
}

// Standard 5-point facial landmark reference points for 512x512 target face alignment
static const float REF_LANDMARKS[5][2] = {
    {192.9814f, 239.9470f}, // Left eye
    {318.9032f, 240.1436f}, // Right eye
    {256.6341f, 314.0193f}, // Nose tip
    {201.2612f, 371.4104f}, // Left mouth
    {313.0804f, 371.1511f}  // Right mouth
};

cv::Mat FaceEnhancerEngine::getAffineTransform512(const std::vector<LandmarkPoint>& landmarks) {
    if (landmarks.size() < 5) {
        return cv::Mat();
    }

    cv::Point2f srcPts[5];
    cv::Point2f dstPts[5];
    for (int i = 0; i < 5; ++i) {
        srcPts[i] = cv::Point2f(landmarks[i].x, landmarks[i].y);
        dstPts[i] = cv::Point2f(REF_LANDMARKS[i][0], REF_LANDMARKS[i][1]);
    }

    // Estimate similarity transform (rotation + scale + translation)
    cv::Mat trans = cv::estimateAffinePartial2D(srcPts, dstPts);
    return trans;
}

cv::Mat FaceEnhancerEngine::enhanceCroppedFace(const cv::Mat& inputBgr, const std::vector<LandmarkPoint>& landmarks, int targetW, int targetH) {
    if (inputBgr.empty() || !session_) {
        return inputBgr;
    }

    // 1. Calculate affine alignment matrix
    cv::Mat affineTrans = getAffineTransform512(landmarks);
    cv::Mat alignedFace;
    if (!affineTrans.empty()) {
        cv::warpAffine(inputBgr, alignedFace, affineTrans, cv::Size(512, 512), cv::INTER_LINEAR);
    } else {
        cv::resize(inputBgr, alignedFace, cv::Size(512, 512));
    }

    // 2. Preprocess to ONNX tensor [1, 3, 512, 512] normalized to [-1, 1]
    cv::Mat rgbMat;
    cv::cvtColor(alignedFace, rgbMat, cv::COLOR_BGR2RGB);

    std::vector<float> inputTensorValues(1 * 3 * 512 * 512);
    int spatialSize = 512 * 512;
    for (int h = 0; h < 512; ++h) {
        for (int w = 0; w < 512; ++w) {
            cv::Vec3b pixel = rgbMat.at<cv::Vec3b>(h, w);
            // Normalize RGB [0, 255] -> [-1, 1]
            inputTensorValues[0 * spatialSize + h * 512 + w] = (pixel[0] / 127.5f) - 1.0f;
            inputTensorValues[1 * spatialSize + h * 512 + w] = (pixel[1] / 127.5f) - 1.0f;
            inputTensorValues[2 * spatialSize + h * 512 + w] = (pixel[2] / 127.5f) - 1.0f;
        }
    }

    std::vector<int64_t> inputShape = {1, 3, 512, 512};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size()
    );

    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};

    // 3. Run ONNX Model Inference
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1
    );

    float* outputData = outputTensors[0].GetTensorMutableData<float>();

    // 4. Post-process tensor to BGR OpenCV Mat
    cv::Mat enhanced512(512, 512, CV_8UC3);
    for (int h = 0; h < 512; ++h) {
        for (int w = 0; w < 512; ++w) {
            float r = outputData[0 * spatialSize + h * 512 + w];
            float g = outputData[1 * spatialSize + h * 512 + w];
            float b = outputData[2 * spatialSize + h * 512 + w];

            // Denormalize [-1, 1] -> [0, 255]
            uint8_t uR = static_cast<uint8_t>(std::clamp((r + 1.0f) * 127.5f, 0.0f, 255.0f));
            uint8_t uG = static_cast<uint8_t>(std::clamp((g + 1.0f) * 127.5f, 0.0f, 255.0f));
            uint8_t uB = static_cast<uint8_t>(std::clamp((b + 1.0f) * 127.5f, 0.0f, 255.0f));

            enhanced512.at<cv::Vec3b>(h, w) = cv::Vec3b(uB, uG, uR); // BGR order
        }
    }

    // 5. Resize enhanced face to target W x H (e.g., 112x112 for FR engine)
    cv::Mat finalFace;
    if (targetW != 512 || targetH != 512) {
        cv::resize(enhanced512, finalFace, cv::Size(targetW, targetH), 0, 0, cv::INTER_AREA);
    } else {
        finalFace = enhanced512;
    }

    return finalFace;
}

cv::Mat FaceEnhancerEngine::enhanceFullImage(const cv::Mat& inputBgr, const std::vector<LandmarkPoint>& landmarks) {
    if (inputBgr.empty() || landmarks.size() < 5) {
        return inputBgr;
    }

    cv::Mat enhancedCropped512 = enhanceCroppedFace(inputBgr, landmarks, 512, 512);
    cv::Mat affineTrans = getAffineTransform512(landmarks);

    if (affineTrans.empty()) {
        return inputBgr;
    }

    // Calculate inverse affine transform to paste enhanced face back to original image
    cv::Mat invAffine;
    cv::invertAffineTransform(affineTrans, invAffine);

    cv::Mat restoredFull;
    inputBgr.copyTo(restoredFull);

    cv::Mat warpedFace;
    cv::warpAffine(enhancedCropped512, warpedFace, invAffine, inputBgr.size(), cv::INTER_LINEAR);

    // Create mask for smooth blending
    cv::Mat mask(512, 512, CV_8UC1, cv::Scalar(255));
    cv::Mat warpedMask;
    cv::warpAffine(mask, warpedMask, invAffine, inputBgr.size(), cv::INTER_LINEAR);

    warpedFace.copyTo(restoredFull, warpedMask);
    return restoredFull;
}
