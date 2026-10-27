pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "PhotoEnhancerTest"
include(":app")
include(":enhancer-sdk")
include(":rknn-experiment")
project(":rknn-experiment").projectDir = file("rknn-experiment/android")
