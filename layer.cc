#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cblas.h>
#include <cstdlib>
#include <cfloat> 

#include "layer.h"
#include "parameter.h"

Layer* GetLayer(NNParameter* nn_para, LayerParameter* layer_para)  {
  if (layer_para->type_ == "conv") {   
    return new ConvLayer(nn_para, layer_para);
  } else if (layer_para->type_ == "pool") {   
    return new PoolLayer(nn_para, layer_para);
  } else if (layer_para->type_ == "full") {   
    return new FullLayer(nn_para, layer_para);  
  } else {
    std::cout << "no each layer" << std::endl;
  }
}      

void Layer::Im2Col(double* im, int im_size, int filter_size, double* im_col) {  
  int height_col = im_size - filter_size + 1;
  int width_col = im_size - filter_size + 1;
  int channels_col = filter_size * filter_size;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % filter_size;
    int h_offset = (c / filter_size) % filter_size;
    int c_im = c / filter_size / filter_size;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        im_col[(c * height_col + h) * width_col + w] =
            im[(c_im * im_size + h + h_offset) * im_size
                + w + w_offset];
      }
    }
  }
}

void Layer::Col2Im(double* im_col, int im_size, int filter_size, double* im) {  
  int height_col = im_size - filter_size + 1;
  int width_col = im_size - filter_size + 1;
  int channels_col = filter_size*filter_size;
  memset(im, 0, im_size*im_size*sizeof(double));
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % filter_size;
    int h_offset = (c / filter_size) % filter_size;
    int c_im = c / filter_size / filter_size;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        im[(c_im * im_size + h  + h_offset) * im_size + w 
           + w_offset] += im_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

void Layer::MaxPool(double *im, int im_size, int pool_dim, double *pooled, int *pool_idx) {    
  int output_dim = im_size / pool_dim;
  int max_idx = 0;
  int this_idx = 0;
  double max_value = 0.0;  
  double this_value = 0.0;
  for (int i = 0; i < output_dim; ++i) {
    for (int j = 0; j < output_dim; ++j) {
      max_value = -DBL_MAX;
      max_idx = 0;
      for (int m = 0; m < pool_dim; ++m) {
        for (int n = 0; n < pool_dim; ++n) {
          this_idx = (i*pool_dim + m)*im_size + j*pool_dim + n;
          this_value = im[this_idx];
          if (this_value > max_value) {
            max_value = this_value;
            max_idx = this_idx;
          }
        }              
      }
      pooled[i*output_dim + j] = max_value;
      pool_idx[i*output_dim + j] = max_idx;      
    }
  }
}

void Layer::UpSample(double *pooled, int *pool_idx, int output_dim, int pool_dim, double *up) {  
  for (int i = 0; i < output_dim*output_dim; ++i) {
    up[pool_idx[i]] = pooled[i];
  }
}

void Layer::Constants(double *x, int size, double value) {
  for (int i = 0; i < size; ++i) {
    x[i] = value;
  }
}

void Layer::Zeros(double *x, int size) {  
  memset(x, 0, size*sizeof(double));
}

void Layer::RandUniform(double *x, int size, double value) {
  for (int i = 0; i < size; ++i) {
    x[i] = value * (double) (rand() + 1.0) / (RAND_MAX+1.0);
  } 
}

void Layer::RandGauss(double *x, int size, double value) {    
  for (int i = 0; i < size; ++i) {
    double rand_1 = (double) (rand() + 1.0) / (RAND_MAX+1.0);
    double rand_2 = (double) (rand() + 1.0) / (RAND_MAX+1.0);
    x[i] = value * sqrt(-2*log(rand_1)) * cos(2*3.14159265*rand_2);
  } 
}
        
void ConvLayer::Init(Layer* bottom, Layer* top) {    
  if (bottom == NULL) {
    input_ = NULL;      
    input_im_size_ = nn_para_->im_size_;
    input_size_ = output_im_size_*output_im_size_;
    input_num_ = 1;
  } else if (bottom != NULL) {
    input_ = bottom->output_;
    input_im_size_ = bottom->output_im_size_;    
    input_size_ = bottom->output_size_;  
    input_num_ = bottom->output_num_;    
  }

  bottom_ = bottom;
  top_ = top;

  output_im_size_ = input_im_size_ - filter_size_ + 1;
  output_size_ = output_im_size_*output_im_size_;  

  output_ = new double[filter_num_*output_size_];
  delta_ = new double[filter_num_*output_size_];
  weight_ = new double[input_num_*filter_num_*filter_size_*filter_size_];
  bias_ = new double[input_num_*filter_num_];      
  grad_weight_ = new double[input_num_*filter_num_*filter_size_*filter_size_];
  grad_bias_ = new double[input_num_*filter_num_];
  im_col_ = new double[output_im_size_*output_im_size_*filter_size_*filter_size_];
  im_col_getgradient_ = new double[output_im_size_*output_im_size_*(filter_size_ + 1)*(filter_size_ + 1)];
  ones_ = new double[output_im_size_*output_im_size_];

  RandGauss(weight_, input_num_*filter_num_*filter_size_*filter_size_, 0.001);
  Zeros(bias_, input_num_*filter_num_);        
  Zeros(delta_, filter_num_*output_size_);
  Constants(ones_, output_im_size_*output_im_size_, 1.0);
}

void ConvLayer::Forward() {    
  Zeros(output_, filter_num_*output_size_);
  for (int i = 0; i < input_num_; ++i) {
    Im2Col(input_ + i*input_im_size_*input_im_size_, input_im_size_, filter_size_, im_col_);  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                filter_num_, output_im_size_*output_im_size_, filter_size_*filter_size_,
                1.0, weight_ + i*filter_num_*filter_size_*filter_size_, filter_size_*filter_size_, im_col_, output_im_size_*output_im_size_, 
                1, output_, output_im_size_*output_im_size_);  
  }
  
  for (int i = 0; i < filter_num_; ++i) {
    cblas_daxpy(output_im_size_*output_im_size_, bias_[i], ones_, 1, output_ + i*output_im_size_*output_im_size_, 1);
  }
}

void ConvLayer::Backward() {  
  if (bottom_ == NULL) {
    return;
  }

  for (int i = 0; i < input_num_; ++i) {
    memset(im_col_, 0, output_im_size_*output_im_size_*filter_size_*filter_size_*sizeof(double));         
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                filter_size_*filter_size_, output_im_size_*output_im_size_,  filter_num_,
                1, weight_ + i*filter_num_*filter_size_*filter_size_, 
                filter_size_*filter_size_, delta_, output_im_size_*output_im_size_, 1, im_col_, output_im_size_*output_im_size_);  
    Col2Im(im_col_, input_im_size_, filter_size_, bottom_->delta_ + i*input_im_size_*input_im_size_);       
  }
}  

void ConvLayer::GetGradient() {
  Zeros(grad_weight_, input_num_*filter_num_*filter_size_*filter_size_);
  Zeros(grad_bias_, input_num_*filter_num_);

  // Convolute input_ with delta_  
  int CONV_DIM = filter_size_ + 1;
  int FILTER_SIZE = output_im_size_;
  for (int i = 0; i < input_num_; ++i) {

    Im2Col(input_ + i*input_im_size_*input_im_size_, input_im_size_, FILTER_SIZE, im_col_getgradient_);  

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              filter_num_, CONV_DIM*CONV_DIM, FILTER_SIZE*FILTER_SIZE,
              (double)1/(nn_para_->mb_size_), delta_, FILTER_SIZE*FILTER_SIZE, im_col_getgradient_, 
              CONV_DIM*CONV_DIM, 1, grad_weight_ + i*filter_num_*filter_size_*filter_size_, CONV_DIM*CONV_DIM);
  }

  // grad_bc2
  for (int i = 0; i < filter_num_; ++i) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 1, output_im_size_*output_im_size_, (double)1/(nn_para_->mb_size_), 
                delta_ + i*output_im_size_*output_im_size_, output_im_size_*output_im_size_, ones_, 1, 1, &grad_bias_[i], 1);          
  }

}

void ConvLayer::Update() {
  cblas_daxpy(input_num_*filter_num_*filter_size_*filter_size_, - nn_para_->learn_rate_, grad_weight_, 1, weight_, 1);    
  cblas_daxpy(filter_num_, - nn_para_->learn_rate_, grad_bias_, 1, bias_, 1);    
}

void PoolLayer::Init(Layer* bottom, Layer* top) {
  if (bottom == NULL) {
    input_ = NULL;      
    input_im_size_ = nn_para_->im_size_;
    input_size_ = output_im_size_*output_im_size_;
    input_num_ = 1;
  } else if (bottom != NULL) {
    input_ = bottom->output_;
    input_im_size_ = bottom->output_im_size_;    
    input_size_ = bottom->output_size_;    
    input_num_ = bottom->output_num_;    
  }

  bottom_ = bottom;
  top_ = top;

  output_num_ = input_num_;    
  output_im_size_ = input_im_size_ / pool_dim_;
  output_size_ = output_im_size_ * output_im_size_;
  output_ = new double[input_num_*output_size_];
  delta_ = new double[input_num_*output_size_];
  pool_idx_ = new int[input_num_*output_size_]; 

  Zeros(delta_, input_num_*output_size_);
}    

void PoolLayer::Forward() {
  Zeros(output_, input_num_*output_size_);
  for (int i = 0; i < input_num_; ++i) {
    MaxPool(input_ + i*input_im_size_*input_im_size_, input_im_size_, pool_dim_, 
            output_ + i*output_im_size_*output_im_size_, pool_idx_ + i*output_im_size_*output_im_size_);      
  }  
}

void PoolLayer::Backward() {  
  for (int i = 0; i < input_num_; ++i) {
    UpSample(delta_ + i*output_im_size_*output_im_size_, pool_idx_ + i*output_im_size_*output_im_size_,
             output_im_size_, pool_dim_, bottom_->delta_ + i*input_im_size_*input_im_size_);
  }
}

void PoolLayer::GetGradient() {}     

void PoolLayer::Update() {}      


void FullLayer::Init(Layer* bottom, Layer* top) {
  if (bottom == NULL) {
    input_ = NULL;      
    input_im_size_ = nn_para_->im_size_;
    input_size_ = output_im_size_*output_im_size_;
    input_num_ = 1;
  } else {
    input_ = bottom->output_;
    input_im_size_ = bottom->output_im_size_;      
    input_size_ = bottom->output_size_;  
    input_num_ = bottom->output_num_;    
  }

  bottom_ = bottom;
  top_ = top;

  output_im_size_ = 0;
  output_num_ = 1;    

  output_ = new double[output_size_];
  delta_ = new double[output_size_];
  weight_ = new double[input_num_*input_size_*output_size_];
  bias_ = new double[output_size_];    
  grad_weight_ = new double[input_size_*output_size_];
  grad_bias_ = new double[output_size_];
  
  Zeros(delta_, output_size_);
  RandGauss(weight_, input_num_*input_size_*output_size_, 0.001);
  Zeros(bias_, output_size_);  

}

void FullLayer::Forward() {
  Zeros(output_, output_size_);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, output_size_, input_num_*input_size_, 1.0, 
              weight_, input_num_*input_size_, input_, 1, 0.0, output_, 1);   

  cblas_daxpy(output_size_, 1, bias_, 1, output_, 1);                           
}

void FullLayer::Backward() {
  if (top_ == NULL) {
    // delta_d = o3 - yLayer* bottom
    memcpy(delta_, output_, output_size_*sizeof(double));
    delta_[label_] -= 1;
  }

  cblas_dgemv(CblasRowMajor, CblasTrans, output_size_, input_size_*input_num_, 1.0,
              weight_, input_size_*input_num_, delta_, 1, 0.0, bottom_->delta_, 1);
}

void FullLayer::GetGradient() {
  Zeros(grad_weight_, input_size_*output_size_);
  Zeros(grad_bias_, output_size_);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              input_num_*input_size_, output_size_, 1,
              (double)1/(nn_para_->mb_size_), input_ , 1, delta_, output_size_, 1, grad_weight_, output_size_);  

  cblas_daxpy(output_size_, (double)1/(nn_para_->mb_size_), delta_, 1, grad_bias_, 1);    
}

void FullLayer::Update() {
  cblas_daxpy(input_num_*input_size_*output_size_, - nn_para_->learn_rate_, grad_weight_, 1, weight_, 1);    
  cblas_daxpy(output_size_, - nn_para_->learn_rate_, grad_bias_, 1, bias_, 1);    
}
