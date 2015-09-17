clc,clear;
load mnist_uint8;
%含有70000个手写数字样本其中60000作为训练样本，10000作为测试样本。?把数据转成相应的格式，并归一化。
train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');