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

#include "nnetwork/nnetwork_node.hpp"

namespace nnetwork
{
auto custom_qos = rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data);
NnetworkNode::NnetworkNode(const rclcpp::NodeOptions & options)
:  Node("nnetwork", options), neural_network_(3, 2, 2)
{
  srand(static_cast<unsigned>(time(nullptr))); 
  auto custom_qos_ackerman = rclcpp::QoS(rclcpp::KeepLast(10)).durability(rmw_qos_durability_policy_t::RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  nnetwork_ = std::make_unique<nnetwork::Nnetwork>();
  param_name_ = this->declare_parameter("param_name", 456);
  lidar_means_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "/filtered_lidar", 
    custom_qos, // QoS profile: Adjust the queue size as needed
    std::bind(&NnetworkNode::lidarMeansCallback, this, std::placeholders::_1));

  localization_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/localization/cartographer/pose", 
    custom_qos, // QoS profile: Adjust the queue size as needed
    std::bind(&NnetworkNode::localizationMeansCallback, this, std::placeholders::_1));

  collision_detected_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/collision_detected", custom_qos, std::bind(&NnetworkNode::collisionDetectedCallback, this, std::placeholders::_1));

  elapsed_time_sub_ = this->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", custom_qos, std::bind(&NnetworkNode::elapsedTimeCallback, this, std::placeholders::_1));
  control_cmd_pub_ = this->create_publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>("/control/command/control_cmd", custom_qos_ackerman);
  initialpose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 10);
  reset_position_pub_ = this->create_publisher<std_msgs::msg::Bool>("reset_position", 10);
  nnetwork_->foo(param_name_);
}
void NnetworkNode::lidarMeansCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    // Process the incoming lidar means data
    // For example, log the size of the received vector
    filtered_lidar_data_ = msg->data;
    for (auto &value : filtered_lidar_data_) {
        value = round(value*100)/100;
    }
    std::vector<float> subvector(filtered_lidar_data_.begin() + 3, filtered_lidar_data_.end() - 3);
    // for (const auto &value : subvector) {
    //     std::cout << value << " ";
    // }
    // std::cout << std::endl;
    std::vector<float> nn_output = neural_network_.forward(subvector);
    if (distance>60)
    {
      neural_network_.saveWeightsAndBiases("best_weights.txt");
    }
    auto command_msg = std::make_shared<autoware_auto_control_msgs::msg::AckermannControlCommand>();
    if (nn_output.size() >= 2) {
        command_msg->lateral.steering_tire_angle = nn_output[0];
        command_msg->longitudinal.acceleration = nn_output[1];


    }
    control_cmd_pub_->publish(*command_msg);
    // std::cout<<"START"<<std::endl;
    // for( auto& el: nn_output)
    // {
    //   std::cout<<el<<std::endl;
    // }
    // std::cout<<"END"<<std::endl;
    
}

void NnetworkNode::localizationMeansCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    // Process the incoming lidar means data
    // For example, log the size of the received vector
    last_known_pose_ = msg->pose;
    reset_counter+=1;
    //geometry_msgs::msg::Pose tmp;
   
    if(new_pose==0)
    {

    first_pose = msg->pose;
    new_pose=1;
    }
    else if(new_pose==1)
    {
      second_pose=msg->pose;
      new_pose=2;
    }
    else
    {
      
      first_pose=second_pose;
      second_pose=msg->pose;
      

      //zmienić to pod kątem możliwego znaku ujemnego pozycji.
      if(abs(second_pose.position.x-first_pose.position.x)>0.005 || abs(second_pose.position.y-first_pose.position.y)>0.005)
      {
        if(abs(second_pose.position.x-first_pose.position.x)<0.1 && abs(second_pose.position.y-first_pose.position.y)<0.1)
        {
        reset_counter=0;
        distance=distance+sqrt(pow(abs(second_pose.position.x)-abs(first_pose.position.x),2)+pow(abs(second_pose.position.y)-abs(first_pose.position.y),2));
        //std::cout<<distance<<std::endl;
        }
      }
      else
      {
        if(reset_counter == 300)
        {
          reset_counter=0;
          std_msgs::msg::Bool msg;
          msg.data = true;
          reset_position_pub_->publish(msg);
        }

      }
    }
}
    

void NnetworkNode::collisionDetectedCallback(std_msgs::msg::Bool::SharedPtr msg)
{
    auto message = geometry_msgs::msg::PoseWithCovarianceStamped();
    message.header.stamp = this->now();
    message.header.frame_id = "map"; // or whatever your fixed frame is
    //message.pose.pose = last_known_pose_;
    message.pose.pose.position.x=0.0;
    message.pose.pose.position.y=0.0;
    message.pose.pose.position.z=0.0;
    if (msg->data==1)
    {
      neural_network_.fitness = distance;
      genetic_networks.emplace_back(neural_network_);
      if(genetic_networks.size() == 25)
      {
        mutation_count=0;
        generation_count+=1;
        // std::cout<<"=============NEW GENERATION: "<<generation_count<<"============="<<std::endl;
        is_first_generation = false;
        std::sort(genetic_networks.begin(), genetic_networks.end(), [](const SimpleNeuralNetwork& a, const SimpleNeuralNetwork& b) {return a.fitness > b.fitness;});
        genetic_networks[0].saveWeightsAndBiases("weights.txt");
         [[maybe_unused]] double mean_fitness = calculateMeanFitness(genetic_networks);
        double total_fitness = std::accumulate(genetic_networks.begin(), genetic_networks.end(), 0.0, 
                                        [](double sum, const SimpleNeuralNetwork& nn) { return sum + nn.fitness; });
        // std::cout<<"MEAN FINTESS OF GENERATION: "<<mean_fitness<<std::endl;
        // for(int i=0; i<int(genetic_networks.size()-20); i++)
        // {
        //   for(int j=0; j<int(genetic_networks.size()-20); j++)
        //   {
        //     auto child = crossover(genetic_networks[i], genetic_networks[j]);
        //     mutate(child, 0.05);
        //     childs_networks.emplace_back(child);
        //   }
        // }
        for (int i = 0; i < 25; i++)
        {
          const SimpleNeuralNetwork& parent1 = selectParent(genetic_networks, total_fitness);
          const SimpleNeuralNetwork& parent2 = selectParent(genetic_networks, total_fitness);
          
          auto child = crossover(parent1, parent2);
          mutate(child, 0.05);
          childs_networks.emplace_back(child);
        }
        // std::cout<<"Mutation counter: "<<mutation_count<<std::endl;
        // std::cout<<"child_size: "<<childs_networks.size()<<std::endl;
        neural_network_ = childs_networks[0];
        childs_networks.erase(childs_networks.begin());
        genetic_networks.clear();

      }
      else
      {
        if(is_first_generation == 1)
        {
          // std::cout<<"tt: "<<total_time<<" lt: "<<lap_timer<<" ltt: "<<last_total_time<<std::endl;
           [[maybe_unused]] float time_result = std::stof(total_time)+std::stof(lap_timer)-std::stof(last_total_time);
          // std::cout<<"Fitness score: "<<neural_network_.fitness<<" Lap timer: "<<time_result<<std::endl;
          if (total_time=="0.0")
          {
            total_time = lap_timer;
          }
          last_total_time = total_time;
          neural_network_.reinicialize_weights();
        }
        else
        {
          // std::cout<<"tt: "<<total_time<<" lt: "<<lap_timer<<" ltt: "<<last_total_time<<std::endl;
           [[maybe_unused]] float time_result = std::stof(total_time)+std::stof(lap_timer)-std::stof(last_total_time);
          // std::cout<<"Fitness score: "<<neural_network_.fitness<<" Lap timer: "<<time_result<<std::endl;
          if (total_time=="0.0")
          {
            total_time = lap_timer;
          }
          last_total_time = total_time;
          neural_network_ = childs_networks[0];
          childs_networks.erase(childs_networks.begin());
        }
      }
      initialpose_pub_->publish(message);
      // std::cout<<"Collision !!!"<<std::endl;
      distance=0.0;
      new_pose=0;
    }
}

void NnetworkNode::elapsedTimeCallback(diagnostic_msgs::msg::DiagnosticArray::SharedPtr msg)
{
  std::string currentLap;
  for (const auto& status : msg->status) {
        if (status.name == "LapTimer") {
            for (const auto& value : status.values) {
              if (value.key == "TOTAL_TIME") {
                    total_time = value.value;
                }
                if (value.key == "CURRENT_LAP") {
                    currentLap =  value.value;
                }

                }
            }
    }
  if (!currentLap.empty()) {
        lap_timer = currentLap;
    }
};

SimpleNeuralNetwork NnetworkNode::crossover(const SimpleNeuralNetwork& parent1, const SimpleNeuralNetwork& parent2)
{
    int input_nodes = parent1.weights_ih.size();
    int hidden_nodes = parent1.weights_ih[0].size();
    int output_nodes = parent1.weights_ho[0].size();
    
    SimpleNeuralNetwork child(input_nodes, hidden_nodes, output_nodes);

    // Crossover for weights from input to hidden
    for (size_t i = 0; i < child.weights_ih.size(); ++i) {
        for (size_t j = 0; j < child.weights_ih[i].size(); ++j) {
            child.weights_ih[i][j] = (rand() % 2) ? parent1.weights_ih[i][j] : parent2.weights_ih[i][j];
        }
    }

    // Crossover for weights from hidden to output
    for (size_t i = 0; i < child.weights_ho.size(); ++i) {
        for (size_t j = 0; j < child.weights_ho[i].size(); ++j) {
            child.weights_ho[i][j] = (rand() % 2) ? parent1.weights_ho[i][j] : parent2.weights_ho[i][j];
        }
    }

    // Crossover for biases
    for (size_t i = 0; i < child.bias_h.size(); ++i) {
        child.bias_h[i] = (rand() % 2) ? parent1.bias_h[i] : parent2.bias_h[i];
    }
    for (size_t i = 0; i < child.bias_o.size(); ++i) {
        child.bias_o[i] = (rand() % 2) ? parent1.bias_o[i] : parent2.bias_o[i];
    }

    return child;

}

void NnetworkNode::mutate(SimpleNeuralNetwork& nn, double mutation_rate)
{
  auto mutate_element = [mutation_rate, this](double elem) {
        std::random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        if (dis(gen) < mutation_rate) {
            std::uniform_real_distribution<float> dis(-0.2, 0.2);
            this->mutation_count++;
            elem += dis(gen);  // Small mutation
        }
        return elem;
    };

    // Mutate weights and biases
    for (auto& row : nn.weights_ih)
    {
        for (auto& weight : row)
        {
            weight = mutate_element(weight);
        }
    }

    for (auto& row : nn.weights_ho)
    {
        for (auto& weight : row)
        {
            weight = mutate_element(weight);
        }
    }

    for (auto& bias : nn.bias_h)
    {
        bias = mutate_element(bias);
    }

    for (auto& bias : nn.bias_o)
    {
      bias = mutate_element(bias);
    }

}
double NnetworkNode::calculateMeanFitness(const std::vector<SimpleNeuralNetwork>& networks)
{
  if (networks.empty()) {
        return 0.0;  // To handle the case where the vector is empty
    }

    double sum_fitness = 0.0;
    for (const auto& nn : networks) {
        sum_fitness += nn.fitness;
    }

    double mean_fitness = sum_fitness / networks.size();
    return mean_fitness;
}
SimpleNeuralNetwork NnetworkNode::selectParent(const std::vector<SimpleNeuralNetwork>& networks, double total_fitness)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, total_fitness);

    double random_point = dis(gen);
    double cumulative_fitness = 0.0;

    for (const auto& network : networks) {
        cumulative_fitness += network.fitness;
        if (cumulative_fitness >= random_point) {
            return network;
        }
    }
    return networks.back(); // In case of floating point imprecision
}


}  // namespace nnetwork

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(nnetwork::NnetworkNode)
