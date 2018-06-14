from numpy import *
import numpy as np
# (1)先准备数据集
def loadDataSet():
    dataSet=[]
    label=[]
    with open("testSet.txt") as f:
        for line in f.readlines():
            line=line.strip().split()  # 这里读出来是string
            dataSet.append([1.0,float(line[0]),float(line[1])])
            label.append(int(line[-1]))
    return dataSet,label

# 对输入数据运用Sigmoid算法
def Sigmoid(inX):
    return 0.5 * (1 + np.tanh(0.5 * inX))
# 梯度上升算法
# 两种方法进行矩阵运算
#  1. 调用numpy.matmul()
#  2. 转化为矩阵（mat）用mat进行计算
def gradAscent(dataSetIn,labels):
    dataSet=np.array(dataSetIn)
    nSamples, Attr = dataSet.shape
    labels=np.array(labels).reshape(nSamples,1)

    # 权重初始化为
    weight=np.zeros((Attr,1),dtype=float)
    # 学习率初始化为
    learning_rate=0.001
    # 最大迭代次数
    MaxIter=500
    for k in range(MaxIter):
        h=Sigmoid(np.matmul(dataSet,weight))  # python中默认的* 是按元素乘 mat中的 * 是矩阵相乘
        error=labels-h
        weight=weight+learning_rate*(np.matmul(dataSet.transpose(),error))

    return weight

# 画出最佳拟合直线
def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = loadDataSet()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 改进的随机梯度下降算法
def stoGradAscent(dataSetIn,labels):
    dataSet=np.array(dataSetIn)
    nSamples, Attr = dataSet.shape

    # 权重初始化为
    weight=np.ones(Attr,dtype=float)
    # 学习率初始化为
    learning_rate=0.001
    # 最大迭代次数
    MaxIter=500
    for k in range(MaxIter):
        dataIndex=list(range(nSamples))
        for i in range(nSamples):
            learning_rate=1/(i+k+1.0)+0.01
            index=int(random.uniform(0,len(dataIndex)))
            h=Sigmoid(sum(dataSet[index]*weight))  # python中默认的* 是按元素乘 mat中的 * 是矩阵相乘
            error=labels[index]-h
            weight=weight+learning_rate*dataSet[index]*error
            del(dataIndex[index])
    return weight

DataSet,labels=loadDataSet()
weight=stoGradAscent(DataSet,labels)
print (weight)
plot_best_fit(weight)


# Logistic回归分类函数
def classify_vector(inx, weights):
    """ 
    它以回归系数和特征向量作为输入来计算对应的Sigmoid值。 
    如果Sigmoid值大于0.5函数返回1，否则返回0。 
    """
    prob = Sigmoid(sum(inx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    train_weights = stoGradAscent(np.array(training_set), training_labels, 1000)
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = (float(error_count) / num_test_vec)
    print("the error rate of this test is: %f" % error_rate)
    return error_rate

def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print("after %d iterations the average error rate is: %f" % (num_tests, error_sum / float(num_tests)))


if __name__ == '__main__':
    # dataMat, labelMat = load_data_set()
    # weights = grad_ascent(dataMat, labelMat)
    # weights = stoc_grad_ascent1(np.array(dataMat), labelMat)
    # plot_best_fit(weights)
    multi_test()