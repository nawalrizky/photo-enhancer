#include "rknn_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string quote(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') result << '\\' << c;
        else if (c < 32) result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else result << c;
    }
    result << '"';
    return result.str();
}

template <typename T> T symbol(void* library, const char* name) {
    void* address = dlsym(library, name);
    if (!address) throw std::runtime_error(std::string("Missing runtime symbol: ") + name);
    return reinterpret_cast<T>(address);
}

void check(int status, const char* stage) {
    if (status != RKNN_SUCC) throw std::runtime_error(std::string(stage) + " failed: " + std::to_string(status));
}

std::string tensorJson(const rknn_tensor_attr& attr) {
    std::ostringstream out;
    out << "{\"name\":" << quote(attr.name) << ",\"dims\":[";
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        if (i) out << ',';
        out << attr.dims[i];
    }
    out << "],\"elements\":" << attr.n_elems << ",\"format\":" << attr.fmt
        << ",\"type\":" << attr.type << ",\"quantization\":" << attr.qnt_type
        << ",\"bytes\":" << attr.size << ",\"bytes_with_stride\":" << attr.size_with_stride << '}';
    return out.str();
}

std::string memoryJson() {
    std::ifstream status("/proc/self/status");
    std::string line;
    long rss = 0, hwm = 0;
    while (std::getline(status, line)) {
        std::sscanf(line.c_str(), "VmRSS: %ld kB", &rss);
        std::sscanf(line.c_str(), "VmHWM: %ld kB", &hwm);
    }
    return "{\"rss_kib\":" + std::to_string(rss) + ",\"hwm_kib\":" + std::to_string(hwm) + '}';
}

void writeBytes(const std::string& path, const void* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char*>(data), size);
    if (!file) throw std::runtime_error("Could not write: " + path);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: rknn_probe RUNTIME_SO MODEL_RKNN OUTPUT_PREFIX [RUNS=4] [INPUT_FLOAT_BIN=-] [PROFILE=0] [NATIVE_INPUT=0]\n";
        return 2;
    }
    rknn_context context = 0;
    void* library = nullptr;
    decltype(&rknn_destroy) destroy = nullptr;
    try {
        const int runs = argc > 4 ? std::atoi(argv[4]) : 4;
        if (runs < 1 || runs > 10) throw std::runtime_error("Runs must be 1 to 10");
        const bool profile = argc > 6 && std::atoi(argv[6]) != 0;
        const bool nativeInput = argc > 7 && std::atoi(argv[7]) != 0;
        const std::string prefix = argv[3];
        library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(std::string("Could not load runtime: ") + dlerror());
        const auto init = symbol<decltype(&rknn_init)>(library, "rknn_init");
        const auto query = symbol<decltype(&rknn_query)>(library, "rknn_query");
        const auto setInputs = symbol<decltype(&rknn_inputs_set)>(library, "rknn_inputs_set");
        const auto run = symbol<decltype(&rknn_run)>(library, "rknn_run");
        const auto getOutputs = symbol<decltype(&rknn_outputs_get)>(library, "rknn_outputs_get");
        const auto releaseOutputs = symbol<decltype(&rknn_outputs_release)>(library, "rknn_outputs_release");
        destroy = symbol<decltype(&rknn_destroy)>(library, "rknn_destroy");
        const auto initStarted = Clock::now();
        check(init(&context, argv[2], 0, profile ? RKNN_FLAG_COLLECT_PERF_MASK : 0, nullptr), "rknn_init");
        const double initMs = milliseconds(initStarted);
        rknn_sdk_version version{};
        check(query(context, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)), "SDK version query");
        rknn_input_output_num counts{};
        check(query(context, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts)), "I/O count query");
        if (counts.n_input != 1 || counts.n_output != 1) throw std::runtime_error("Expected one input and one output");
        rknn_tensor_attr inputAttr{}, outputAttr{};
        check(query(context, RKNN_QUERY_INPUT_ATTR, &inputAttr, sizeof(inputAttr)), "Input query");
        check(query(context, RKNN_QUERY_OUTPUT_ATTR, &outputAttr, sizeof(outputAttr)), "Output query");
        rknn_tensor_attr nativeInputAttr{};
        check(query(context, RKNN_QUERY_NATIVE_INPUT_ATTR, &nativeInputAttr, sizeof(nativeInputAttr)), "Native input query");
        std::cout << "RKNN_PROBE {\"event\":\"native_input_contract\",\"input\":" << tensorJson(nativeInputAttr) << "}" << std::endl;
        std::cout << "RKNN_PROBE {\"event\":\"initialized\",\"model\":" << quote(argv[2])
                  << ",\"init_ms\":" << initMs << ",\"api_version\":" << quote(version.api_version)
                  << ",\"driver_version\":" << quote(version.drv_version)
                  << ",\"profiling\":" << (profile ? "true" : "false")
                  << ",\"native_input\":" << (nativeInput ? "true" : "false")
                  << ",\"input\":" << tensorJson(inputAttr) << ",\"output\":" << tensorJson(outputAttr)
                  << ",\"memory\":" << memoryJson() << "}" << std::endl;
        if (!inputAttr.n_elems || !outputAttr.n_elems) throw std::runtime_error("Empty input/output tensor");
        std::vector<float> input(inputAttr.n_elems);
        if (argc > 5 && std::string(argv[5]) != "-") {
            std::ifstream file(argv[5], std::ios::binary | std::ios::ate);
            if (!file || file.tellg() != static_cast<std::streamoff>(input.size() * sizeof(float)))
                throw std::runtime_error("Input file size does not match float32 tensor");
            file.seekg(0);
            file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
            if (!file) throw std::runtime_error("Could not read input tensor");
        } else {
            for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 257) / 128.0f - 1.0f;
        }
        writeBytes(prefix + ".input.bin", input.data(), input.size() * sizeof(float));
        std::vector<float> nativeFloats;
        std::vector<_Float16> nativeHalves;
        if (nativeInput) {
            if (inputAttr.type != RKNN_TENSOR_FLOAT16 && inputAttr.type != RKNN_TENSOR_FLOAT32)
                throw std::runtime_error("Native input mode requires float16 or float32 model input");
            nativeFloats = input;
            if (inputAttr.fmt == RKNN_TENSOR_NHWC) {
                if (inputAttr.n_dims != 4 || inputAttr.dims[0] != 1)
                    throw std::runtime_error("Native NHWC mode requires a single image");
                const uint32_t height = inputAttr.dims[1], width = inputAttr.dims[2], channels = inputAttr.dims[3];
                for (uint32_t y = 0; y < height; ++y)
                    for (uint32_t x = 0; x < width; ++x)
                        for (uint32_t c = 0; c < channels; ++c)
                            nativeFloats[(y * width + x) * channels + c] = input[(c * height + y) * width + x];
            } else if (inputAttr.fmt != RKNN_TENSOR_NCHW) {
                throw std::runtime_error("Unsupported native input layout");
            }
            if (inputAttr.type == RKNN_TENSOR_FLOAT16) {
                nativeHalves.reserve(nativeFloats.size());
                for (float value : nativeFloats) nativeHalves.push_back(static_cast<_Float16>(value));
            }
        }
        std::vector<std::string> records;
        for (int i = 0; i < runs; ++i) {
            rknn_input tensor{};
            tensor.buf = input.data();
            tensor.size = input.size() * sizeof(float);
            tensor.type = RKNN_TENSOR_FLOAT32;
            tensor.fmt = RKNN_TENSOR_NCHW;
            if (nativeInput) {
                tensor.pass_through = 1;
                tensor.type = inputAttr.type;
                tensor.fmt = inputAttr.fmt;
                if (inputAttr.type == RKNN_TENSOR_FLOAT16) {
                    tensor.buf = nativeHalves.data();
                    tensor.size = nativeHalves.size() * sizeof(_Float16);
                } else {
                    tensor.buf = nativeFloats.data();
                    tensor.size = nativeFloats.size() * sizeof(float);
                }
            }
            const auto callStarted = Clock::now();
            check(setInputs(context, 1, &tensor), "Input set");
            const double inputMs = milliseconds(callStarted);
            const auto runStarted = Clock::now();
            check(run(context, nullptr), "Inference");
            const double runMs = milliseconds(runStarted);
            rknn_output output{};
            output.want_float = 1;
            const auto outputStarted = Clock::now();
            check(getOutputs(context, 1, &output, nullptr), "Output get");
            const double outputMs = milliseconds(outputStarted);
            const double callMs = milliseconds(callStarted);
            if (!output.buf || output.size != outputAttr.n_elems * sizeof(float)) {
                releaseOutputs(context, 1, &output);
                throw std::runtime_error("Float32 output size does not match tensor contract");
            }
            const float* values = static_cast<const float*>(output.buf);
            float minimum = std::numeric_limits<float>::infinity();
            float maximum = -std::numeric_limits<float>::infinity();
            uint32_t nonfinite = 0;
            for (uint32_t j = 0; j < outputAttr.n_elems; ++j) {
                if (!std::isfinite(values[j])) ++nonfinite;
                else { minimum = std::min(minimum, values[j]); maximum = std::max(maximum, values[j]); }
            }
            std::string minJson = std::isfinite(minimum) ? std::to_string(minimum) : "null";
            std::string maxJson = std::isfinite(maximum) ? std::to_string(maximum) : "null";
            writeBytes(prefix + ".run" + std::to_string(i + 1) + ".output.bin", output.buf, output.size);
            check(releaseOutputs(context, 1, &output), "Output release");
            std::ostringstream record;
            record << "{\"event\":\"inference\",\"run\":" << i + 1 << ",\"first_run\":" << (i == 0 ? "true" : "false")
                   << ",\"input_ms\":" << inputMs << ",\"run_ms\":" << runMs << ",\"output_ms\":" << outputMs
                   << ",\"call_ms\":" << callMs << ",\"nonfinite\":" << nonfinite
                   << ",\"minimum\":" << minJson << ",\"maximum\":" << maxJson << ",\"memory\":" << memoryJson() << '}';
            records.push_back(record.str());
            std::cout << "RKNN_PROBE " << record.str() << std::endl;
            if (nonfinite) throw std::runtime_error("Model output contains NaN or infinity");
        }
        if (profile) {
            rknn_perf_detail detail{};
            check(query(context, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail)), "Performance detail query");
            if (detail.perf_data && detail.data_len) writeBytes(prefix + ".perf.txt", detail.perf_data, detail.data_len);
        }
        rknn_mem_size runtimeMemory{};
        const int memoryStatus = query(context, RKNN_QUERY_MEM_SIZE, &runtimeMemory, sizeof(runtimeMemory));
        std::cout << "RKNN_PROBE {\"event\":\"runtime_memory\",\"query_status\":" << memoryStatus
                  << ",\"weight_bytes\":" << runtimeMemory.total_weight_size
                  << ",\"internal_bytes\":" << runtimeMemory.total_internal_size
                  << ",\"dma_allocated_bytes\":" << runtimeMemory.total_dma_allocated_size << "}" << std::endl;
        std::ostringstream summary;
        summary << "{\"init_ms\":" << initMs << ",\"profiling\":" << (profile ? "true" : "false")
                << ",\"native_input\":" << (nativeInput ? "true" : "false")
                << ",\"input\":" << tensorJson(inputAttr) << ",\"output\":" << tensorJson(outputAttr) << ",\"runs\":[";
        for (size_t i = 0; i < records.size(); ++i) { if (i) summary << ','; summary << records[i]; }
        summary << "]}\n";
        const auto data = summary.str();
        writeBytes(prefix + ".summary.json", data.data(), data.size());
        check(destroy(context), "Destroy");
        context = 0;
        dlclose(library);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RKNN_PROBE {\"event\":\"error\",\"message\":" << quote(error.what()) << "}" << std::endl;
        if (context && destroy) destroy(context);
        if (library) dlclose(library);
        return 1;
    }
}
