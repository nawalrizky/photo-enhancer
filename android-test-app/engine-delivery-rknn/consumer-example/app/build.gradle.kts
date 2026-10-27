plugins { id("com.android.application") }
android {
    namespace = "com.enhancer.integrationcheck"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.enhancer.integrationcheck"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "1.1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    buildTypes {
        release {
            isMinifyEnabled = true
            // Debug signing is for the demonstration APK only.
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
dependencies {
    implementation(files("../../ai-filter-enhancer-rknn-1.1.0.aar"))
    implementation("org.jetbrains.kotlin:kotlin-stdlib:1.9.24")
}
