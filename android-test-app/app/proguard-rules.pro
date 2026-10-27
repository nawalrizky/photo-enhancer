# JNI entry points are resolved by their Java class and method names.
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile
