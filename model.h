#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <sstream>

#include "Vmath.h"

#define deg_to_rad (M_PI/180)

std::vector<float>sum_vectors(std::vector<float>v0,std::vector<float>v1)
{
 if (v0.size()>v1.size())
 {return{};} 
 std::vector<float>out;
 for (size_t i=0;i<v0.size();++i)
 {out.push_back(v0[i]+v1[i]);}
 return out;
 }

std::vector<float>sub_vectors(std::vector<float>v0,std::vector<float>v1)
{
 if (v0.size()>v1.size())
 {return{};}
 std::vector<float>out;
 for (size_t i=0;i<v0.size();++i)
 {out.push_back(v0[i]-v1[i]);}
 return out;
}

std::vector<float>mul_vectors(std::vector<float>v0,std::vector<float>v1)
{
 if (v0.size()>v1.size())
 {return{};}
 std::vector<float>out;
 for (size_t i=0;i<v0.size();++i)
 {out.push_back(v0[i]*v1[i]);}
 return out;
 }

std::vector<float>div_vectors(std::vector<float>v0,std::vector<float>v1)
{
 if (v0.size()>v1.size())
 {return{};}
 std::vector<float>out;
 for (size_t i=0;i<v0.size();++i)
 {out.push_back(v0[i]/v1[i]);}
 return out;
}

float vector_sum(std::vector<float>v)
{
 if (v.size()<1)
 {return 0;}
 float out=0;
 for (size_t i=0;i<v.size();++i)
 {out+=v[i];}
 return out;  
}

 float randfloat(float min,float max)
 {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min,max);
  return dis(gen);
 }

float vector_average(std::vector<float>&v)
{
float average=0;
for (float&x:v)
{average+=x;}
return average/v.size();
}

struct neuron
{
 std::vector<float>weights;
 float bias=0;
};

struct data_set
{
 std::vector<std::vector<float>>inputs;
 std::vector<std::vector<float>>desired;
};


typedef std::vector<neuron> layer;
typedef std::vector<layer>  model;


float activation_f(float x)
{return tanhf(x);}

float activation_f_d(float x)
{
    float tanh_x = tanhf(x);
    return 1.0f - tanh_x * tanh_x;
}

float run_neuron(neuron n,std::vector<float>input)
{
float result=0;
if (n.weights.size()>input.size())
{return 0;}
for (size_t w=0;w<n.weights.size();w++)
{result+=input[w]*n.weights[w]+n.bias;}
return activation_f(result);

}

std::vector<float>run_layer(layer l,std::vector<float>input)
{
std::vector<float>result;
for (size_t n=0;n<l.size();++n)
{result.push_back(run_neuron(l[n],input));}
return result;
}

std::vector<float>run_model(model&mdl,std::vector<float>input)
{
std::vector<float>result=input;
for (size_t l=0;l<mdl.size();++l)
{result=run_layer(mdl[l],result);}
return result;
}

neuron gen_neuron(uint64_t wheight_count)
{
neuron generated;
for (size_t w=0;w<wheight_count;++w)
{generated.weights.push_back(randfloat(-1,1));}
generated.bias=randfloat(-1,1);
return generated;
}

layer gen_layer(uint64_t layer_sizes,uint64_t input_size)
{
layer gen;
for (size_t w=0;w<layer_sizes;++w)
{gen.push_back(gen_neuron(input_size));}
return gen;
}

model gen_random_model(std::vector<uint64_t>layer_sizes)
{
model gen;
for (size_t n=0;n<layer_sizes.size();++n)
{
 uint64_t input_size;
 if (n==0)
 {input_size=layer_sizes[0];}
 else
 {input_size=layer_sizes[n-1];}
 gen.push_back(gen_layer(layer_sizes[n],input_size));
}
return gen;
}

std::vector<std::vector<float>>get_errors(model&mdl,data_set&data)
{
std::vector<std::vector<float>>errors;

for (size_t w=0;w<data.inputs.size();++w)
{

 std::vector<float>output=run_model(mdl,data.inputs[w]);
 errors.push_back(sub_vectors(output,data.desired[w]));

}

return errors;
}

void save_model(model mdl, const std::string& path) {
std::ofstream file(path.c_str(), std::ios::binary);
if (file.is_open()) {
size_t numLayers = mdl.size();
file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));
for (const auto& layer : mdl) {
size_t numNeurons = layer.size();
file.write(reinterpret_cast<const char*>(&numNeurons), sizeof(numNeurons));
for (const auto& n : layer) {
file.write(reinterpret_cast<const char*>(&n.bias), sizeof(n.bias));
size_t numWeights = n.weights.size();
file.write(reinterpret_cast<const char*>(&numWeights), sizeof(numWeights));
file.write(reinterpret_cast<const char*>(n.weights.data()), numWeights * sizeof(float));
}
}
file.close();
}
}

model load_model(const std::string& path) {
model loadedmodel;
std::ifstream file(path.c_str(), std::ios::binary);
if (file.is_open()) {
size_t numLayers;
file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
for (size_t i = 0; i < numLayers; ++i) {
size_t numNeurons;
file.read(reinterpret_cast<char*>(&numNeurons), sizeof(numNeurons));
layer l(numNeurons);
for (size_t j = 0; j < numNeurons; ++j) {
file.read(reinterpret_cast<char*>(&l[j].bias), sizeof(l[j].bias));
size_t numWeights;
file.read(reinterpret_cast<char*>(&numWeights), sizeof(numWeights));
l[j].weights.resize(numWeights);
file.read(reinterpret_cast<char*>(l[j].weights.data()), numWeights * sizeof(float));
}
loadedmodel.push_back(l);
}
file.close();
}
return loadedmodel;
}

std::vector<uint64_t>load_model_config(std::string path)
{
std::vector<uint64_t>out;
std::ifstream file(path.c_str());
std::string line;
while (std::getline(file,line))
{
 out.push_back(std::stoi(line));
}

return out;
}