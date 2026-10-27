#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "ai_filter_enhancer.h"

namespace {
void throwRuntimeException(JNIEnv* environment, const std::string& message) {
    const jclass exceptionClass = environment->FindClass("java/lang/RuntimeException");
    if (exceptionClass) environment->ThrowNew(exceptionClass, message.c_str());
}

cv::Mat bitmapToBgr(JNIEnv* environment, jobject bitmap) {
    AndroidBitmapInfo info{};
    void* pixels = nullptr;
    if (!bitmap ||
        AndroidBitmap_getInfo(environment, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        AndroidBitmap_lockPixels(environment, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return {};
    }

    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels, info.stride);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(environment, bitmap);
    return bgr;
}

jobject bgrToBitmap(JNIEnv* environment, const cv::Mat& bgr) {
    if (bgr.empty()) return nullptr;
    const jclass bitmapClass = environment->FindClass("android/graphics/Bitmap");
    const jclass configClass = environment->FindClass("android/graphics/Bitmap$Config");
    if (!bitmapClass || !configClass) return nullptr;

    const jfieldID argbField = environment->GetStaticFieldID(
        configClass,
        "ARGB_8888",
        "Landroid/graphics/Bitmap$Config;"
    );
    const jobject config = environment->GetStaticObjectField(configClass, argbField);
    const jmethodID createMethod = environment->GetStaticMethodID(
        bitmapClass,
        "createBitmap",
        "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;"
    );
    jobject bitmap = environment->CallStaticObjectMethod(
        bitmapClass,
        createMethod,
        bgr.cols,
        bgr.rows,
        config
    );
    if (!bitmap || environment->ExceptionCheck()) return nullptr;

    AndroidBitmapInfo info{};
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(environment, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        AndroidBitmap_lockPixels(environment, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return nullptr;
    }
    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels, info.stride);
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    AndroidBitmap_unlockPixels(environment, bitmap);
    return bitmap;
}

std::vector<LandmarkPoint> readLandmarks(
    JNIEnv* environment,
    jfloatArray coordinates
) {
    if (!coordinates || environment->GetArrayLength(coordinates) != 10) return {};
    std::vector<float> values(10);
    environment->GetFloatArrayRegion(coordinates, 0, 10, values.data());
    if (environment->ExceptionCheck()) return {};

    std::vector<LandmarkPoint> landmarks(5);
    for (size_t i = 0; i < landmarks.size(); ++i) {
        landmarks[i] = {values[i * 2], values[i * 2 + 1]};
    }
    return landmarks;
}

ExperimentalRknnEngine* fromHandle(jlong handle) {
    return reinterpret_cast<ExperimentalRknnEngine*>(handle);
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_enhancer_rknn_RknnEnhancer_initializeNative(
    JNIEnv* environment,
    jobject,
    jstring modelPath
) {
    if (!modelPath) return 0;
    const char* path = environment->GetStringUTFChars(modelPath, nullptr);
    if (!path) return 0;
    const std::string model(path);
    environment->ReleaseStringUTFChars(modelPath, path);

    auto engine = std::make_unique<ExperimentalRknnEngine>();
    if (!engine->initialize(model)) return 0;
    return reinterpret_cast<jlong>(engine.release());
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_enhancer_rknn_RknnEnhancer_enhanceFullNative(
    JNIEnv* environment,
    jobject,
    jlong handle,
    jobject bitmap,
    jfloatArray coordinates
) {
    try {
        const auto started = std::chrono::steady_clock::now();
        ExperimentalRknnEngine* engine = fromHandle(handle);
        if (!engine) return nullptr;
        const cv::Mat input = bitmapToBgr(environment, bitmap);
        const auto landmarks = readLandmarks(environment, coordinates);
        if (input.empty() || landmarks.size() != 5) return nullptr;
        const auto inputReady = std::chrono::steady_clock::now();
        const cv::Mat output = engine->enhanceFull(input, landmarks);
        const auto engineReady = std::chrono::steady_clock::now();
        jobject result = bgrToBitmap(environment, output);
        const auto outputReady = std::chrono::steady_clock::now();
        const auto ms = [](auto end, auto begin) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        };
        __android_log_print(
            ANDROID_LOG_INFO, "ExperimentalRknn",
            "stage=jni bitmap_in_ms=%ld engine_ms=%ld bitmap_out_ms=%ld total_ms=%ld",
            ms(inputReady, started), ms(engineReady, inputReady),
            ms(outputReady, engineReady), ms(outputReady, started)
        );
        return result;
    } catch (const std::exception& error) {
        throwRuntimeException(environment, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_enhancer_rknn_RknnEnhancer_releaseNative(
    JNIEnv*,
    jobject,
    jlong handle
) {
    delete fromHandle(handle);
}
