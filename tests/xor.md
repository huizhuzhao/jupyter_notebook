## xor data 
([code][https://github.com/huizhuzhao/Snow/blob/master/examples/xor.py])

首先 xor 问题的数据如下，也就是输入为 X, 输出为 Y

![](http://oe5p7f8mz.bkt.clouddn.com/xor_x_y.png)
我们希望训练一个简单的多层感知器 (MLP) 模型 f 能够满足上面的要求，即将输入正确的映射到输出
```
Y = f(X)
```

模型的示意图如下:

![](http://ojwkl64pe.bkt.clouddn.com/xor_mlp.png?imageView2/2/h/200)

在大概明确了数据和模型以后，接下来我们使用 `python` 生成训练模型的数据， 代码如下：
```
X_test = np.asarray([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
Y_test = np.asarray([[1.], [0.], [0.], [1.]], dtype=np.float32)

X_train = np.tile(X_test, (100, 1))
Y_train = np.tile(Y_test, (100, 1))

>>> X_test.shape, Y_test.shape
(4, 2), (4, 1)
>>> X_train.shape, Y_train.shape
(400, 2), (400, 1)
```

我们使用 `keras` 来搭建 MLP 模型，实现的代码如下：

```
input_dim, output_dim = 2, 1
hidden_dim = 10
model = Sequential()
model.add(Dense(hidden_dim, input_dim=input_dim, activation='tanh', name='hidden'))
model.add(Dense(output_dim, activation='softmax'), name='output')
xx
>>> model.input_shape, model.output_shape
(None, 2), (None, 1)
>>> model.summary()
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
hidden (Dense)               (None, 10)                30        
_________________________________________________________________
output (Dense)               (None, 1)                 11        
=================================================================
Total params: 41.0
Trainable params: 41
Non-trainable params: 0.0
```

模型中我们将中间隐藏层的节点数目设置为 `hidden_dim=10`，两层网络中的参数数目如上面所示。

接着我们对模型 `model` 进行编译：

```
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

其中 `loss funtion` 选用 `binary_crossentropy`，该函数要求 `y_true.shape == y_pred.shape`，计算公式为：

```
loss = z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
 
(z = y_true, x = y_pred)
```

优化函数选用 `adam`, 评价指标选用准确率。

模型训练结果如下：

![](http://ojwkl64pe.bkt.clouddn.com/xor_res.png?imageView2/2/h/400) 准确率(acc 紫色) 和 损失函数 (loss 蓝色) 与 epoch 之间的关系图，可以看出在 `epoch~50` 时，训练集上的准确率达到 `100%`，但损失函数的下降趋势扔在继续。
