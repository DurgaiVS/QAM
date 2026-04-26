#ifndef __BOLT_MODEL_HH__
#define __BOLT_MODEL_HH__


#include "./utils.hh"

namespace bolt {

class QAMModel {
private:
    const Ort::Env env;
    const Ort::SessionOptions session_options;
    const Ort::Session session;

public:
    QAMModel(
        const char* model_path,
        Ort::SessionOptions session_options,
        int model_max_length,
        int model_input_size,
        int model_output_size,
    ) : env(ORT_LOGGING_LEVEL_WARNING, "QAM")
    , session_options(session_options)
    , session(this->env, model_path, session_options)
    { };

    ~QAMModel();

    Ort::Value infer(
        Ort::MemoryInfo* memory_info,
        std::vector<bolt::timepoint>& input,
        std::vector<int>& input_shape,
        std::vector<bolt::timepoint>& state_points,
        std::vector<std::array<float, bolt::POLICY_HEAD_SIZE>>& output
    );

} // class QAMModel

QAMModel::infer(
    Ort::MemoryInfo* memory_info,
    std::vector<bolt::timepoint>& input,
    std::vector<int>& input_shape,
    std::vector<bolt::timepoint>& state_points,
    std::vector<std::array<float, bolt::POLICY_HEAD_SIZE>>& output
) {

    Ort::Value input_tensor = Ort::Value::CreateTensor<bolt::timepoint>(
        *memory_info,
        input.data(),
        input.size(),
    );

    Ort::Value input_shape_tensor = Ort::Value::CreateTensor<int>(
        *memory_info,
        input_shape.data(),
        input_shape.size()
    );

    Ort::Value state_point_tensor = Ort::Value::CreateTensor<bolt::timepoint>(
        *memory_info,
        state_points.data(),
        state_points.size()
    );

    // Run inference
    const char* input_names[] = {"input", "ip_lengths", "state_point"};
    const char* output_names[] = {"logits"};

    auto output_tensors = this->session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        (const Ort::Value*[]){&input_tensor, &input_shape_tensor, &state_point_tensor},
        3,
        output_names,
        1
    );

    // Copy output to output vector
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    int output_size = 1;
    for (int i = 0; i < output_tensors[0].GetTensorTypeAndShapeInfo().GetDimensionsCount(); i++) {
        output_size *= output_tensors[0].GetTensorTypeAndShapeInfo().GetDimensions()[i];
    }

}; // namespace bolt

#endif // __BOLT_MODEL_HH__
