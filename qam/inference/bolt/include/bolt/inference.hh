#ifndef __BOLT_INFERENCE_HH__
#define __BOLT_INFERENCE_HH__


#include <iostream>

#include "./model.hh"
#include "./utils.hh"


namespace bolt {

void run_inference() {
    int batch_size = 1, state_point_count = 3;

    bolt::QAMModel model("", bolt::session_options);
    Ort::MemoryInfo memory_info = bolt::create_memory_info_cpu();

    // Example input data
    std::vector<float> input(batch_size * bolt::MODEL_INPUT_SIZE * bolt::INPUT_DIM, 1.0f);
    std::vector<int> input_length(batch_size, bolt::MODEL_INPUT_SIZE);
    std::vector<float> state_point(batch_size * state_point_count * bolt::INPUT_DIM, 0.0f);

    std::vector<float> output(batch_size * bolt::OUTPUT_DIM, 0.0f);

    int64_t input_shape[] = {batch_size, bolt::MODEL_INPUT_SIZE, bolt::INPUT_DIM};
    int16_t input_length_shape[] = {batch_size};
    int64_t state_point_shape[] = {batch_size, state_point_count, bolt::INPUT_DIM};

    int64_t output_shape[] = {batch_size, bolt::OUTPUT_DIM};

    model.infer(
        &memory_info,
        input,
        input_length,
        state_point,
        output,
        input_shape,
        input_length_shape,
        state_point_shape,
        output_shape
    );

    std::cout << "Buy: " << output[0] << ", Sell: " << output[1] << ", Hold: " << output[2] << std::endl;

}

}


#endif // __BOLT_INFERENCE_HH__
