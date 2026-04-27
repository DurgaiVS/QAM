#ifndef __BOLT_MODEL_HH__
#define __BOLT_MODEL_HH__


#include "./utils.hh"

namespace bolt {

class QAMModel {
protected:
    const Ort::Env env;
    const Ort::SessionOptions session_options;
    const Ort::Session session;

public:
    QAMModel(
        const char* model_path,
        Ort::SessionOptions session_options,
    ) : env(ORT_LOGGING_LEVEL_WARNING, "QAM")
    , session_options(session_options)
    , session(this->env, model_path, session_options)
    { };

    ~QAMModel();

    inline void infer(
        Ort::MemoryInfo* memory_info,
        std::vector<float>& input,
        std::vector<int>& input_length,
        std::vector<float>& state_point,
        std::vector<float>& output,
        int64_t* input_shape,
        int16_t* input_length_shape,
        int64_t* state_point_shape,
        int64_t* output_shape,
    );

} // class QAMModel

inline void QAMModel::infer(
    Ort::MemoryInfo* memory_info,
    std::vector<float>& input,
    std::vector<int>& input_length,
    std::vector<float>& state_point,
    std::vector<float>& output,
    int64_t* input_shape,
    int16_t* input_length_shape,
    int64_t* state_point_shape,
    int64_t* output_shape,
) {

    const char* input_names[] = {"input", "input_length", "state_point"};
    Ort::Value input_tensors[] = {
        // "input" tensor
        Ort::Value::CreateTensor<float>(
            *memory_info,
            input.data(),
            input.size(),
            input_shape,
            3
        ),

        // "input_length" tensor
        Ort::Value::CreateTensor<int>(
            *memory_info,
            input_length.data(),
            input_length.size(),
            input_length_shape,
            1
        ),

        // "state_point" tensor
        Ort::Value::CreateTensor<float>(
            *memory_info,
            state_point.data(),
            state_point.size(),
            state_point_shape,
            3
        )
    };

    const char* output_names[] = {"logits"};
    Ort::Value output_tensors[] = {
        // "logits" tensor
        Ort::Value::CreateTensor<float>(
            *memory_info,
            output.data(),
            output.size(),
            output_shape,
            2
        )
    };

    this->session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        input_tensors,
        3,
        output_names,
        output_tensors,
        1,
    );
}

}; // namespace bolt

#endif // __BOLT_MODEL_HH__
