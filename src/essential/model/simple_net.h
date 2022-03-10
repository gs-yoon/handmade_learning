#include"layers.h"


class SimpleNet
{
    public:
    Tensor W1;
    Tensor b1;
    Tensor W2;
    Tensor b2;
    Tensor W3;
    Tensor b3;

    void init_network()
    {
        W1.createTensor(2,3);
        b1.createTensor(3);
        W2.createTensor(3,2);
        b2.createTensor(2);
        W3.createTensor(2,2);
        b3.createTensor(2);

        W1(0,0)=0.1;W1(0,1)=0.3;W1(0,2)=0.5;
        W1(1,0)=0.2;W1(1,1)=0.4;W1(1,2)=0.6;
        b1(0,0)=0.1;b1(0,1)=0.2;b1(0,2)=0.3;
        W2(0,0)=0.1;W2(0,1)=0.4;
        W2(1,0)=0.2;W2(1,1)=0.5;
        W2(2,0)=0.3;W2(2,1)=0.6;
        b2(0,0)=0.1;b2(0,1)=0.2;
        W3(0,0)=0.1;W3(0,1)=0.3;
        W3(1,0)=0.2;W3(1,1)=0.4;
        b3(0,0)=0.1;b3(0,1)=0.2;
    }

    Tensor forward(Tensor x)
    {
        Tensor a1,z1,a2,z2,a3;
        std::cout<<"x =";
        std::cout<<x;
        std::cout<<"-x =";
        std::cout<<x*(-1);
        std::cout<<"W1 =";
        std::cout<<W1;
        std::cout<<"b1 =";
        std::cout<<b1;
        std::cout<<"dot(x,W1) =";
        std::cout<<x.dotMul(W1);
        a1 = x.dotMul(W1) + b1;
        std::cout<<"a1 =";
        std::cout<<a1;
        z1 = sigmoid(a1);
        std::cout<<"z1 =";
        std::cout<<z1;
        a2 = z1.dotMul(W2) +b2;
        z2 = sigmoid(a2);
        a3 = z2.dotMul(W3) + b3;
        Tensor y;
        y = identity_function(a3);

        return y;
    }
};