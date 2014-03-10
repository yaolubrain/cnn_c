#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <string>

#include "parameter.h"
#include "common.h"
#include "nn.h"

class Layer {
 public:
  Layer(NNParameter* nn_para, LayerParameter* layer_para) {
    nn_para_ = nn_para;
  };

  ~Layer() {};  

  void Im2Col(double* im, int im_size, int filter_size, double* im_col);
  void Col2Im(double* im_col, int im_size, int filter_size, double* im);
  void Conv2(double *im, double *Wc, int im_size, int filter_size, int filter_num, double alpha, double *output);
  void MaxPool(double *im, int im_size, int pool_dim, double *pooled, int *pool_idx);
  void UpSample(double *pooled, int *pool_idx, int output_dim, int pool_dim, double *up);
  void Constants(double *x, int size, double value);
  void Zeros(double *x, int size);
  void RandUniform(double *x, int size, double value);
  void RandGauss(double *x, int size, double value); 

  virtual void Init(Layer* top, Layer* bottom) {};  
  virtual void Forward() {};
  virtual void Backward() {};
  virtual void GetGradient() {} ;
  virtual void Update() {};
  
  std::string type_;

  int input_im_size_;
  int input_size_;
  int input_num_;  
  int output_im_size_;
  int output_size_;
  int output_num_;  

  int label_;

  NNParameter* nn_para_;

  double* input_;
  double* output_;
  double* delta_;
  Layer* top_;
  Layer* bottom_;
};


Layer* GetLayer(NNParameter* nn_para, LayerParameter *layer_para);


class ConvLayer : public Layer {  
 public: 
  ConvLayer(NNParameter* nn_para, LayerParameter *layer_para) : Layer(nn_para, layer_para) {
    type_ = "conv";    
    filter_num_ = layer_para->filter_num_;
    filter_size_ = layer_para->filter_size_;      
    output_num_ = filter_num_;    
  };

  ~ConvLayer() {    
    delete[] output_;
    delete[] delta_;
    delete[] weight_;
    delete[] bias_;
    delete[] grad_weight_;
    delete[] grad_bias_;
  };  

  void Init(Layer* bottom, Layer* top);
  void Forward();
  void Backward();
  void GetGradient();
  void Update();
  
  int filter_num_;
  int filter_size_;      
  double* weight_;
  double* bias_;  
  double* grad_weight_;
  double* grad_bias_;
  double* im_col_;
  double* im_col_getgradient_;
  double* ones_;
};

class PoolLayer : public Layer {  
 public: 
  PoolLayer(NNParameter* nn_para, LayerParameter *layer_para) : Layer(nn_para, layer_para) {
    type_ = "pool";
    pool_dim_ = layer_para->pool_dim_;
  };

  ~PoolLayer() {
    delete[] output_;
    delete[] delta_;
    delete[] pool_idx_;
  };    

  void Init(Layer* bottom, Layer* top);
  void Forward();
  void Backward();
  void GetGradient();
  void Update();

  int pool_dim_;      
  int* pool_idx_;
};

class FullLayer : public Layer {  
 public:   
  FullLayer(NNParameter* nn_para, LayerParameter *layer_para) : Layer(nn_para, layer_para) {    
    type_ = "full";
    output_im_size_ = 0;
    output_size_ = layer_para->output_size_;
    output_num_ = 1;
  };

  ~FullLayer() {
    delete[] output_;
    delete[] delta_;    
    delete[] weight_;
    delete[] bias_;  
    delete[] grad_weight_;
    delete[] grad_bias_;
  };      

  void Init(Layer* bottom, Layer* top);
  void Forward();
  void Backward();
  void GetGradient();
  void Update();
  
  double* weight_;
  double* bias_;  
  double* grad_weight_;
  double* grad_bias_;
};

#endif