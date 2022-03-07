#ifndef __LAYERS_H__
#define __LAYERS_H__

#include "gradient.h"
#include "initializer.h"

class Layer
{
private:
    VALUETYPE gradient_ =0;
public:
    Layer(){}
    ~Layer(){}
    virtual Tensor forward(Tensor& x){}
    virtual Tensor backward(Tensor& dout){}
};

class Relu : public Layer
{
private:

public:
    Relu()
    {
        //self.mask;
    }
    ~Relu(){}
    Tensor forward(Tensor& x)
    {
        return relu(x);
    }

    Tensor backward(Tensor& dout)
    {
        return relu_grad(dout);
    }
};

class Sigmoid: public Layer
{
    private:
    Tensor out_;
    public:
    Sigmoid();
    ~Sigmoid();

    Tensor forward(Tensor& x)
    {
        out_ = sigmoid(x);
        return out_;
    }

    Tensor backward(Tensor& dout)
    {
        Tensor dx;
        dx = dout * (1.0 - out_) * out_;

        return dx;
    }
};

class Affine: public Layer
{
private:
    Tensor x_; //(input_size)
    int* original_x_shape_ =nullptr;

public:
    Tensor W_; //(input_size, ouput_size)
    Tensor b_; //(input_size, ouput_size)
    Tensor dW_; //(input_size, ouput_size)
    Tensor db_; //(input_size, ouput_size)

    void (*initializer)(Tensor&) = gaussianRandomInit;
    
    Affine(){}
    ~Affine(){}

    Affine(int input_size, int output_size)
    {
        W_.createTensor(input_size, output_size);
        b_.createTensor(output_size);
        x_.createTensor(input_size);
        dW_.createTensor(input_size, output_size);
        db_.createTensor(output_size);
    }

    void init(int input_size, int output_size)
    {
        W_.createTensor(input_size, output_size);
        b_.createTensor(input_size, output_size);
        x_.createTensor(input_size);
        dW_.createTensor(input_size, output_size);
        db_.createTensor(input_size, output_size);
        if (initializer != nullptr)
        {
            initializer(W_);
        }
        else
        {
            printf("affine init error. initializer is not defined\n");
        }
    }


    Tensor forward(Tensor& x)
    {
        original_x_shape_ = x.getRawShape();
        //x_ = x.flatten();  flatten 아닐걸?
        if (x.rank() > 2)
            x_ = x.reshape(x.getShape(0), x.getSize() / x.getShape(0));

        Tensor out;
        out = x_.dotMul(W_) + b_;

        return out;
    }
    Tensor backward(Tensor& dout)
    {
        Tensor dx;
        dx = dout.dotMul(W_.transpose());
        dW_ = x_.transpose().dotMul(dout);
        db_ = dout.sum(0);
        
        Tensor result;
        if (dx.rank() > 2)
            result = dx.reshape(original_x_shape_);
        else
            result = dx;
        return result;
    }
};

class SoftmaxWithLoss: public Layer
{
private:
        Tensor loss_;
        Tensor y_;
        Tensor t_;

public:
    SoftmaxWithLoss(){}
    ~SoftmaxWithLoss(){}
    void init()
    {}

    void setLabel(Tensor &t)
    {
        t_ = t;
    }

    Tensor forward(Tensor& x)
    {
        y_ = softmax(x);
        loss_ = cross_entropy_error(y_, t_);
        
        return loss_;
    }

    Tensor backward(Tensor& dout)
    {
        //dout = 1;
        int batch_size = t_.getShape(0);
        int t_size = t_.getSize();
        int y_size = y_.getSize();

        Tensor dx;
        if (t_size == y_size) // 教師データがone-hot-vectorの場合
        {
            dx = (y_ - t_) / batch_size;
        }
        else
        {
            dx = y_;
            //dx[np.arange(batch_size), self.t] -= 1; //TODO: Fix it
            dx = dx / batch_size;
            printf("softmaxWithLoss Error\n");
        }
        
        return dx;
    }
};

class Dropout : public Layer
{
    //http://arxiv.org/abs/1207.0580
private :
    int dropout_ratio_ =50; //TODO : float -> int
    Tensor mask_;
    bool train_flg = true;

public:
    Dropout(float dropout_ratio)
    {
        dropout_ratio_ = dropout_ratio * 100;
    }

    void setTrainFlag(bool flg){
        train_flg = flg;
    }

    Tensor dropout_atomic(double dropout_ratio)
    {
        return (rand()%100) > dropout_ratio? 1: 0;
    }

    Tensor forward(Tensor& x)
    {
        if (train_flg)
        {
            mask_.createTensor(x.getRawShape());
            int shape[5] = {0};
            
            for (int i =0 ; i< 5; i ++)
                shape[i] = x.getRawShape(i);

            for(int d1_idx = 0 ; d1_idx < shape[0]; d1_idx++)
                for( int d2_idx = 0; d2_idx < shape[1] ;d2_idx++)
                    for( int d3_idx = 0; d3_idx < shape[2] ;d3_idx++)
                        for( int d4_idx = 0; d4_idx < shape[3] ; d4_idx++)
                            for( int d5_idx = 0; d5_idx < shape[4] ;d5_idx++)
                                mask_(d1_idx,d2_idx,d3_idx,d4_idx,d5_idx) = (rand()%100) > dropout_ratio_ ? 1: 0;
        }
        else
        {
            return x * (1.0 - ((float)dropout_ratio_)/100);
        }
    }

    Tensor backward(Tensor& dout)
    {
        return dout * mask_;
    }
};

/*

class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
*/

#endif