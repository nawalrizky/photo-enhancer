plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.enhancer.rknn"
    compileSdk = 34
    ndkVersion = "27.2.12479018"
    defaultConfig {
        minSdk = 28
        consumerProguardFiles("consumer-rules.pro")
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake {
            cppFlags += listOf("-std=c++17", "-O3")
            arguments += "-DANDROID_STL=c++_shared"
        } }
    }
    buildTypes { release { isMinifyEnabled = false } }
    externalNativeBuild { cmake {
        path = file("src/main/cpp/CMakeLists.txt")
        version = "3.22.1"
    } }
    buildFeatures { prefab = true }
    packaging { jniLibs {
        useLegacyPackaging = true
        pickFirsts += "**/libc++_shared.so"
    } }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}
dependencies { compileOnly("org.opencv:opencv:4.10.0") }
