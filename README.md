# LogisticRegression
LogisticRegression

### Logistic回归的思想

就是在线性回归的外面套了一个sigmoid函数，具体的例子如下：</br>

<a href="http://www.codecogs.com/eqnedit.php?latex=\sigma(z)=\frac{1}{1&plus;e^{-w^{T}x}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\sigma(z)=\frac{1}{1&plus;e^{-w^{T}x}}" title="\sigma(z)=\frac{1}{1+e^{-w^{T}x}}" /></a>

&emsp;&emsp; 在每个特征上都乘以一个回归系数，然后把所有的结果相加，将这个总和带入到Sigmoid函数中，进而得到一个范围在0-1之间的数值，任何一个大于0.5的数被分为1类，小于0.5的被分到0类。所以，Logistic可以被看做一种概率估计。

&emsp;&emsp; Logistic回归的主要问题就变成了------>求最佳回归系数。

### 最佳回归系数的确定------>梯度上升法、随机梯度上升法

在这里明确梯度的概念：

&emsp;emsp; 1. 梯度算子总之指向函数值增长最快的方向。

&emsp;&emsp; 2. 梯度上升算法用来求目标函数的最大值

&emsp;&emsp; 3. 梯度下降算法是用来求目标函数的最小值


#### 梯度上升算法

1. 每个回归系数都初始化为1

2. 重复 R 次  

 &emsp;&emsp;计算整个数据集的梯度  
 
 &emsp;&emsp;使用alpha\*gradient更新回归系数的向量
 
3. 返回回归系数

#### 随机梯度上升算法

1. 所有回归系数初始化为 1 

2. 对数据集中的每个样本

&emsp;&emsp;计算该样本的梯度

&emsp;&emsp;使用alpha\*gradient更新回归系数的向量

3. 返回回归系数 

#### 改进随机梯度上升算法

1. 每个回归系数都初始化为1

2. 重复 R 次  

 &emsp; &emsp;2.1 对于整个数据集
 
&emsp; &emsp; &emsp;从数据集中随机选择1个样本 
 
 &emsp;&emsp;&emsp;动态调整学习速率alpha
 
 &emsp;&emsp;&emsp;使用alpha\*gradient更新回归系数的向量
 
&emsp; &emsp;&emsp;从总体样本中排除已经计算过的样本
 
3. 返回回归系数




