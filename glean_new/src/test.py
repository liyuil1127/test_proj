# 定义一个 my_layer.py
import torch


class MyLayer(torch.nn.Module):
    '''
    因为这个层实现的功能是：y=weights*sqrt(x2+bias),所以有两个参数：
    权值矩阵weights
    偏置矩阵bias
    输入 x 的维度是（in_features,)
    输出 y 的维度是（out_features,) 故而
    bias 的维度是（in_fearures,)，注意这里为什么是in_features,而不是out_features，注意体会这里和Linear层的区别所在
    weights 的维度是（in_features, out_features）注意这里为什么是（in_features, out_features）,而不是（out_features, in_features），注意体会这里和Linear层的区别所在
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(MyLayer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))  # 由于weights是可以训练的，所以使用Parameter来定义
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(in_features))  # 由于bias是可以训练的，所以使用Parameter来定义
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        input_ = torch.pow(input, 2) + self.bias
        y = torch.matmul(input_, self.weight)
        return y

N, D_in, D_out = 10, 5, 3  # 一共10组样本，输入特征为5，输出特征为3

# 先定义一个模型
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.mylayer1 = MyLayer(D_in, D_out)

    def forward(self, x):
        x = self.mylayer1(x)

        return x


model = MyNet()
print(model)
'''运行结果为：
MyNet(
  (mylayer1): MyLayer()   # 这就是自己定义的一个层
)
'''

# 创建输入、输出数据
x = torch.randn(N, D_in)  # （10，5）
y = torch.randn(N, D_out)  # （10，3）

# 定义损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
# 构造一个optimizer对象
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(10):  #

    # 第一步：数据的前向传播，计算预测值p_pred
    y_pred = model(x)

    # 第二步：计算计算预测值p_pred与真实值的误差
    loss = loss_fn(y_pred, y)
    print(f"第 {t} 个epoch, 损失是 {loss.item()}")

    # 在反向传播之前，将模型的梯度归零，这
    optimizer.zero_grad()

    # 第三步：反向传播误差
    loss.backward()

    # 直接通过梯度一步到位，更新完整个网络的训练参数
    optimizer.step()