#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "native_enhancer.h"

namespace {
std::unique_ptr<FaceEnhancerEngine> engine;
std::mutex engineMutex;

void throwRuntimeException(JNIEnv* env, const std::string& message) {
    const jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
    if (exceptionClass) env->ThrowNew(exceptionClass, message.c_str());
}

cv::Mat bitmapToBgr(JNIEnv* env, jobject bitmap) {
    AndroidBitmapInfo info{};
    void* pixels = nullptr;
    if (!bitmap || AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return {};
    }

    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels, info.stride);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bitmap);
    return bgr;
}

jobject bgrToBitmap(JNIEnv* env, const cv::Mat& bgr) {
    if (bgr.empty()) return nullptr;
    const jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    const jclass configClass = env->FindClass("android/graphics/Bitmap$Config");
    if (!bitmapClass || !configClass) return nullptr;

    const jfieldID argbField = env->GetStaticFieldID(
        configClass,
        "ARGB_8888",
        "Landroid/graphics/Bitmap$Config;"
    );
    const jobject config = env->GetStaticObjectField(configClass, argbField);
    const jmethodID createMethod = env->GetStaticMethodID(
        bitmapClass,
        "createBitmap",
        "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;"
    );
    jobject bitmap = env->CallStaticObjectMethod(bitmapClass, createMethod, bgr.cols, bgr.rows, config);
    if (!bitmap || env->ExceptionCheck()) return nullptr;

    AndroidBitmapInfo info{};
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return nullptr;
    }
    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels, info.stride);
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    AndroidBitmap_unlockPixels(env, bitmap);
    return bitmap;
}

std::vector<LandmarkPoint> readLandmarks(JNIEnv* env, jfloatArray coordinates) {
    if (!coordinates || env->GetArrayLength(coordinates) != 10) return {};
    std::vector<float> values(10);
    env->GetFloatArrayRegion(coordinates, 0, 10, values.data());
    if (env->ExceptionCheck()) return {};

    std::vector<LandmarkPoint> result(5);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = {values[i * 2], values[i * 2 + 1]};
    }
    return result;
}
}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_enhancer_PhotoEnhancer_initNative(
    JNIEnv* env,
    jobject,
    jstring modelPath,
    jboolean useNNAPI
) {
    if (!modelPath) return JNI_FALSE;
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (!path) return JNI_FALSE;
    const std::string model(path);
    env->ReleaseStringUTFChars(modelPath, path);

    std::lock_guard<std::mutex> lock(engineMutex);
    auto candidate = std::make_unique<FaceEnhancerEngine>();
    if (!candidate->init(model, useNNAPI == JNI_TRUE)) return JNI_FALSE;
    engine = std::move(candidate);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_enhancer_PhotoEnhancer_enhanceCroppedFaceNative(
    JNIEnv* env,
    jobject,
    jobject bitmap,
    jfloatArray coordinates,
    jint targetWidth,
    jint targetHeight
) {
    try {
        const cv::Mat input = bitmapToBgr(env, bitmap);
        const auto landmarks = readLandmarks(env, coordinates);
        std::lock_guard<std::mutex> lock(engineMutex);
        if (!engine || input.empty() || landmarks.size() != 5) return nullptr;
        return bgrToBitmap(
            env,
            engine->enhanceCroppedFace(input, landmarks, targetWidth, targetHeight)
        );
    } catch (const std::exception& error) {
        throwRuntimeException(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_enhancer_PhotoEnhancer_enhanceFullImageNative(
    JNIEnv* env,
    jobject,
    jobject bitmap,
    jfloatArray coordinates
) {
    try {
        const cv::Mat input = bitmapToBgr(env, bitmap);
        const auto landmarks = readLandmarks(env, coordinates);
        std::lock_guard<std::mutex> lock(engineMutex);
        if (!engine || input.empty() || landmarks.size() != 5) return nullptr;
        return bgrToBitmap(env, engine->enhanceFullImage(input, landmarks));
    } catch (const std::exception& error) {
        throwRuntimeException(env, error.what());
        return nullptr;
    }
}
