// Copyright 2024 111
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NNETWORK__NNETWORK_NODE_HPP_
#define NNETWORK__NNETWORK_NODE_HPP_

#include <memory>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>
#include <random>
#include <rclcpp/rclcpp.hpp>

#include "nnetwork/nnetwork.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/bool.hpp" 
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "autoware_auto_control_msgs/msg/ackermann_control_command.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"



namespace nnetwork
{
float sigmoid(float x)
{
    return 2*(1.0 / (1.0 + exp(x))-0.5);
    //return std::tanh(x);
}

class SimpleNeuralNetwork {
private:
    int input_nodes;
    int hidden_nodes;
    int output_nodes;

public:
    float fitness;
    std::vector<std::vector<float>> weights_ih; // Weights from input to hidden layer
    std::vector<std::vector<float>> weights_ho; // Weights from hidden to output layer
    std::vector<float> bias_h; // Bias for hidden layer
    std::vector<float> bias_o; // Bias for output layer
    SimpleNeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) {
        // srand(static_cast<unsigned>(time(nullptr))); // Seed for random number generation
        this->input_nodes = input_nodes;
        this->hidden_nodes = hidden_nodes;
        this->output_nodes = output_nodes;
        //Initialize weights and biases for input to hidden layer
        // weights_ih = initializeMatrix(this->input_nodes, this->hidden_nodes);
        // bias_h = initializeVector(this->hidden_nodes);

        // // Initialize weights for hidden to output layer
        // weights_ho = initializeMatrix(this->hidden_nodes, this->output_nodes);
        // bias_o = initializeVector(this->output_nodes);

        weights_ih = {
        {0.149142, -0.770397},
        {-0.604177, 0.251801},
        {-0.706744, 1.18058  },
    };
        bias_h = {-0.397729, -0.197582  };

        // Initialize weights for hidden to output layer
        weights_ho = {
        {-0.0965195, -0.165319  },
        {0.896422, -1.00664  }
    };
        bias_o = {0.259774, -0.727357};
        //display_weights();
    }

    void reinicialize_weights()
    {
        fitness = 0;
        weights_ih = {
        {0.149142, -0.770397},
        {-0.604177, 0.251801},
        {-0.706744, 1.18058 },
    };
        bias_h = {-0.397729, -0.197582 };

        // Initialize weights for hidden to output layer
        weights_ho = {
        {-0.0965195, -0.165319 },
        {0.896422, -1.00664 }
    };
        bias_o = {0.259774, -0.727357};
        // weights_ih = initializeMatrix(input_nodes, hidden_nodes);
        // bias_h = initializeVector(hidden_nodes);
        // weights_ho = initializeMatrix(hidden_nodes, output_nodes);
        // bias_o = initializeVector(output_nodes);
        // // display_weights();

    }

    void display_weights()
    {
        for(auto& row:weights_ih)
        {
            for(auto& col:row)
            {
                std::cout<<col<<std::endl;
            }
        }
    }

    std::vector<std::vector<float>> initializeMatrix(int rows, int cols) 
    {
        std::random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0, 1.0);
        std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                matrix[i][j] = dis(gen);
        return matrix;
    }

    std::vector<float> initializeVector(int size) {
        std::vector<float> vec(size);
        std::random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0, 1.0);
        for (int i = 0; i < size; ++i)
            vec[i] = dis(gen);
        return vec;
    }

    // Forward propagation
    std::vector<float> forward(const std::vector<float>& inputs) {
        std::vector<float> hidden_layer_output = matrixVectorMultiplication(weights_ih, inputs, bias_h);
        applyActivationFunction(hidden_layer_output);

        std::vector<float> output_layer_output = matrixVectorMultiplication(weights_ho, hidden_layer_output, bias_o);
        applyActivationFunction(output_layer_output);
        // output_layer_output[1]*=5;
        // output_layer_output[1] = abs(output_layer_output[1]);
        return output_layer_output;
    }

    std::vector<float> matrixVectorMultiplication(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec, const std::vector<float>& bias) {
        std::vector<float> result(bias.size(), 0.0);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[0].size(); ++j) {
                result[j] += matrix[i][j] * vec[i];
            }
        }
        for (size_t i = 0; i < bias.size(); ++i) {
            result[i] += bias[i];
        }
        return result;
    }

    float clip(float n, float lower, float upper) 
    {
        return std::max(lower, std::min(n, upper));
    }

    void applyActivationFunction(std::vector<float>& vec) {
        for (float& val : vec) {
            val = clip(val, -10, 10);
            val = sigmoid(val);
            // if(std::isnan(val))
            // {
            //     val = -0.1;
            // }
        }
    }
    void saveWeightsAndBiases(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
            return;
        }
        file << "Fitness Score: " << fitness << "\n";
        // Save weights from input to hidden layer
        file << "weights_ih\n"; 
        for (const auto& row : weights_ih) {
            for (const auto& val : row) {
                file << val << " ";
            }
            file << "\n";
        }

        // Save biases for hidden layer
        file << "bias_h\n";
        for (const auto& b : bias_h) {
            file << b << " ";
        }
        file << "\n";

        // Save weights from hidden to output layer
        file << "weights_ho\n";
        for (const auto& row : weights_ho) {
            for (const auto& val : row) {
                file << val << " ";
            }
            file << "\n";
        }

        // Save biases for output layer
        file << "bias_o\n";
        for (const auto& b : bias_o) {
            file << b << " ";
        }
        file << "\n";

        file.close();
    }
};

using NnetworkPtr = std::unique_ptr<nnetwork::Nnetwork>;

class NNETWORK_PUBLIC NnetworkNode : public rclcpp::Node
{
public:
  explicit NnetworkNode(const rclcpp::NodeOptions & options);
  std::vector<SimpleNeuralNetwork> genetic_networks;
  std::vector<SimpleNeuralNetwork> childs_networks;
  bool is_first_generation = true;

private:
  NnetworkPtr nnetwork_{nullptr};
  SimpleNeuralNetwork neural_network_;
  geometry_msgs::msg::Pose last_known_pose_;
  geometry_msgs::msg::Pose first_pose;
  geometry_msgs::msg::Pose second_pose;
  SimpleNeuralNetwork crossover(const SimpleNeuralNetwork& parent1, const SimpleNeuralNetwork& parent2);
  SimpleNeuralNetwork selectParent(const std::vector<SimpleNeuralNetwork>& networks, double total_fitness);
  void mutate(SimpleNeuralNetwork& nn, double mutation_rate);
  double calculateMeanFitness(const std::vector<SimpleNeuralNetwork>& networks);
  float distance=0.0;
  int new_pose=0;
  int mutation_count=0;
  int generation_count=0;
  int reset_counter = 0;
  std::string lap_timer="0";
  std::string last_total_time="0";
  std::string total_time="0";
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr lidar_means_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr localization_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr collision_detected_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr elapsed_time_sub_;
  rclcpp::Publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>::SharedPtr control_cmd_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr reset_position_pub_;
  void lidarMeansCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void localizationMeansCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void collisionDetectedCallback(std_msgs::msg::Bool::SharedPtr msg);
  void elapsedTimeCallback(diagnostic_msgs::msg::DiagnosticArray::SharedPtr msg);
  std::vector<float> filtered_lidar_data_;
  int64_t param_name_{123};
};
}  // namespace nnetwork

#endif  // NNETWORK__NNETWORK_NODE_HPP_
