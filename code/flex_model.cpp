#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>


std::vector<torch::Tensor> get_module_parameters(torch::jit::script::Module& model, const std::string& module_name) {
    std::vector<torch::Tensor> params;
    for (const auto& module : model.named_modules()) {
        if (module.name == module_name) {
            for (const auto& param : module.value.parameters()) {
                params.push_back(param);
            }
            break;  // Found the module, no need to continue the loop
        }
    }
    return params;
}


class FlexibleModel {
public:
    torch::jit::script::Module model;
     std::shared_ptr<torch::optim::Adam> optimizer;

    FlexibleModel(const std::string& model_path) {
        model = torch::jit::load(model_path);
        optimizer = std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>(), torch::optim::AdamOptions(1e-3));
    }


    void setup_optimizer() {
        auto& param_groups = optimizer->param_groups();

        std::vector<torch::Tensor> low_params = get_module_parameters(model, "lr_low");
        param_groups.push_back(torch::optim::OptimizerParamGroup({low_params}));

        std::vector<torch::Tensor> mid_params = get_module_parameters(model, "lr_mid");
        param_groups.push_back( torch::optim::OptimizerParamGroup({mid_params}));

        std::vector<torch::Tensor> high_params = get_module_parameters(model, "lr_high");
        param_groups.push_back(torch::optim::OptimizerParamGroup({high_params}));

        optimizer = std::make_shared<torch::optim::Adam>(param_groups);


    }


    void update_learning_rate(double lr_low, double lr_mid, double lr_high) {

        static_cast<torch::optim::AdamOptions &>(optimizer->param_groups()[0].options()).lr(lr_low);
        static_cast<torch::optim::AdamOptions &>(optimizer->param_groups()[1].options()).lr(lr_mid);
        static_cast<torch::optim::AdamOptions &>(optimizer->param_groups()[2].options()).lr(lr_high);

    }

    void set_freeze(const std::string& freeze_module_name, int freeze_layers) {
        auto params = get_module_parameters(model, freeze_module_name);

        int i = 0;
        for (auto& param : params) {
            param.set_requires_grad(i >= freeze_layers);
            i++;
        }
    }

    std::tuple<float, float> train_it(torch::Tensor inputs, torch::Tensor targets) {
        model.train(); // Set the model to training mode
        optimizer->zero_grad();

        // Forward pass
        auto outputs = model.run_method("compute_loss_acc",inputs, targets);
        auto loss = outputs.toTuple()->elements()[0].toTensor();
        auto acc = outputs.toTuple()->elements()[1].toTensor();

        loss.backward();
        optimizer->step();

        return std::make_tuple(loss.item<float>(),acc.item<float>());
    }

    std::tuple<float, float> eval_it(torch::Tensor inputs, torch::Tensor targets) {
        model.eval(); // Set the model to evaluation mode
        torch::NoGradGuard no_grad;

        // Forward pass for evaluation
        //std::vector<torch::jit::IValue> input_values = {inputs, targets};
        auto outputs = model.run_method("compute_loss_acc", inputs, targets);
        auto loss = outputs.toTuple()->elements()[0].toTensor();
        auto acc = outputs.toTuple()->elements()[1].toTensor();

        return std::make_tuple(loss.item<float>(),acc.item<float>());    }



};



int main() {

    FlexibleModel flexModel("custom_resnet_scripted.pt");

    // Set learning rates for different modules
    flexModel.setup_optimizer();
    flexModel.update_learning_rate(.0001,.0001,.0001);

    // Unfreeze certain layers
    flexModel.set_freeze("freeze_mod", 3);

    // Train one iteration
    auto train_inputs = torch::zeros({5,3,64,64}); // Your training input tensor
    auto train_targets = torch::ones({5,10}); // Your training target tensor
    auto [train_loss, train_acc] = flexModel.train_it(train_inputs, train_targets);
    std::cout << "train_loss: " << train_loss << " train_acc: " << train_acc << std::endl;
    // Evaluate one iteration
    auto eval_inputs = torch::zeros({5,3,64,64}); // Your training input tensor
    auto eval_targets = torch::ones({5,10}); // Your training target tensor
    auto [eval_loss, eval_acc] = flexModel.eval_it(eval_inputs, eval_targets);
    std::cout << "eval_loss: " << eval_loss << " eval_acc: " << eval_acc << std::endl;
    // Use train_loss, train_acc, eval_loss, eval_acc as needed
    // ...
    return 0;
}
