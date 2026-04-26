#ifndef __BOLT_UTILS_HH__
#define __BOLT_UTILS_HH__

#include <unistd.h>
#include <string>
#include <cmath>

#include "onnxruntime_cxx_api.h"

namespace bolt {

struct timepoint_raw {
    std::string symbol;
    float open;
    float close;
    float high;
    float low;
    int volume;
    std::string time;
    int n_trades;
    float volume_wa;
};

struct timepoint {
    float symbol;
    float open;
    float close;
    float high;
    float low;
    float volume;
    float time;
    float n_trades;
    float volume_wa;
};

union timepoint_t {
    timepoint tp;
    char bytes[sizeof(timepoint)];
};

static const int POLICY_HEAD_SIZE = 3;
static const int SUBSAMPLING_FACTOR = 8;

static const int CORE_COUNT = sysconf(_SC_NPROCESSORS_ONLN);
static const int OP_PARALLELISM = 4;
static const int INTRA_OP_NUM_THREADS = CORE_COUNT / OP_PARALLELISM;
static const int INTER_OP_NUM_THREADS = OP_PARALLELISM;

static Ort::SessionOptions session_options;
OrtCUDAProviderOptions cuda_options;

session_options.SetIntraOpNumThreads(INTRA_OP_NUM_THREADS);
session_options.SetInterOpNumThreads(INTER_OP_NUM_THREADS);
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

bool configure_cuda_provider(int device_id = 0) {
    try {
        cuda_options.device_id = device_id;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to configure CUDA provider: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Calculates the output length after subsampling.
 *
 * @param input_length The length of the input sequence.
 * @param subsampling_factor The factor by which the input sequence is subsampled.
 *
 * @return The length of the output sequence after subsampling.
 */
template <typename T>
T calculate_output_length(int input_length, int subsampling_factor) {
    return (T)std::ceil((float)input_length / (float)subsampling_factor);
}

/**
 * Circulates the 1st element of the buffer to the end, and
 * shifts all other elements to the left by one position.
 *
 * @param buffer The buffer to circulate.
 * @param size The size of the buffer.
 */
template <typename T>
void circulate_buffer(T* buffer, int size) {
    T* temp = buffer;
    for (int i = 0; i < size - 1; ++i) {
        std::memcpy(temp, temp + 1, sizeof(T));
        temp++;
    }
}

/**
 * Creates an Ort::MemoryInfo object for CPU memory.
 *
 * @return An Ort::MemoryInfo object configured for CPU memory.
 */
Ort::MemoryInfo create_memory_info_cpu() {
    return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

/**
 * Creates an Ort::MemoryInfo object for CUDA memory.
 *
 * @param device_id The ID of the CUDA device to use (default is 0).
 *
 * @return An Ort::MemoryInfo object configured for CUDA memory.
 */
Ort::MemoryInfo create_memory_info_cuda(int device_id = 0) {
    return Ort::MemoryInfo::CreateCuda(device_id);
}

}; // namespace bolt

#endif // __BOLT_UTILS_HH__
