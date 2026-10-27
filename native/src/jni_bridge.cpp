#include <jni.h>
#include <string>
#include <vector>
#include <android/bitmap.h>
#include "native_enhancer.h"

static FaceEnhancerEngine* g_engine = nullptr;

// Helper function to convert Android Bitmap to OpenCV BGR Mat
static cv::Mat bitmapToMat(JNIEnv* env, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return cv::Mat();
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return cv::Mat();

    cv::Mat mat(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgrMat;
    cv::cvtColor(mat, bgrMat, cv::COLOR_RGBA2BGR);

    AndroidBitmap_unlockPixels(env, bitmap);
    return bgrMat;
}

// Helper function to convert OpenCV BGR Mat to Android RGBA Bitmap
static jobject matToBitmap(JNIEnv* env, const cv::Mat& bgrMat) {
    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMethod = env->GetStaticMethodID(
        bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;"
    );

    jclass configClass = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb8888Field = env->GetStaticFieldID(configClass, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    jobject configObj = env->GetStaticObjectField(configClass, argb8888Field);

    jobject bitmap = env->CallStaticObjectMethod(
        bitmapClass, createBitmapMethod, bgrMat.cols, bgrMat.rows, configObj
    );

    AndroidBitmapInfo info;
    void* pixels = nullptr;
    AndroidBitmap_getInfo(env, bitmap, &info);
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    cv::Mat rgbaMat(info.height, info.width, CV_8UC4, pixels);
    cv::cvtColor(bgrMat, rgbaMat, cv::COLOR_BGR2RGBA);

    AndroidBitmap_unlockPixels(env, bitmap);
    return bitmap;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_enhancer_PhotoEnhancer_initNative(JNIEnv* env, jobject thiz, jstring modelPath, jboolean useGPU) {
    const char* pathStr = env->GetStringUTFChars(modelPath, nullptr);
    if (!g_engine) {
        g_engine = new FaceEnhancerEngine();
    }
    bool success = g_engine->init(std::string(pathStr), useGPU);
    env->ReleaseStringUTFChars(modelPath, pathStr);
    return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jobject JNICALL
Java_com_enhancer_PhotoEnhancer_enhanceCroppedFaceNative(
        JNIEnv* env, jobject thiz, jobject inputBitmap, jfloatArray landmarkCoords, jint targetW, jint targetH) {

    if (!g_engine) return nullptr;

    cv::Mat bgrMat = bitmapToMat(env, inputBitmap);
    if (bgrMat.empty()) return nullptr;

    // Convert landmark float array [x0, y0, x1, y1, ...]
    jfloat* landmarksArr = env->GetFloatArrayElements(landmarkCoords, nullptr);
    jsize len = env->GetArrayLength(landmarkCoords);
    std::vector<LandmarkPoint> landmarks;
    for (int i = 0; i < len; i += 2) {
        landmarks.push_back({landmarksArr[i], landmarksArr[i + 1]});
    }
    env->ReleaseFloatArrayElements(landmarkCoords, landmarksArr, JNI_ABORT);

    cv::Mat enhancedFace = g_engine->enhanceCroppedFace(bgrMat, landmarks, targetW, targetH);
    return matToBitmap(env, enhancedFace);
}

JNIEXPORT jobject JNICALL
Java_com_enhancer_PhotoEnhancer_enhanceFullImageNative(
        JNIEnv* env, jobject thiz, jobject inputBitmap, jfloatArray landmarkCoords) {

    if (!g_engine) return nullptr;

    cv::Mat bgrMat = bitmapToMat(env, inputBitmap);
    if (bgrMat.empty()) return nullptr;

    jfloat* landmarksArr = env->GetFloatArrayElements(landmarkCoords, nullptr);
    jsize len = env->GetArrayLength(landmarkCoords);
    std::vector<LandmarkPoint> landmarks;
    for (int i = 0; i < len; i += 2) {
        landmarks.push_back({landmarksArr[i], landmarksArr[i + 1]});
    }
    env->ReleaseFloatArrayElements(landmarkCoords, landmarksArr, JNI_ABORT);

    cv::Mat enhancedFull = g_engine->enhanceFullImage(bgrMat, landmarks);
    return matToBitmap(env, enhancedFull);
}

} // extern "C"
