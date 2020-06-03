import tensorflow as tf
import numpy as np


#定义且仅定义模型
def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    """
    建立神经网络模型
    :param model: 模型类别   包括rnn，lstm，gru
    :param input_data: 输入数据  [64, ?]
    :param output_data: 输出数据  [64, ?]
    :param vocab_size: 词汇表大小
    :param rnn_size:  LSTM单元中的神经元数量，即输出神经元数量
    :param num_layers:  LSTM网络层数
    :param batch_size:  批处理大小
    :param learning_rate:  学习率大小
    :return:  end_points
    """
    end_points = {}   #返回数据字典

    #定义基本单元
    if model == 'rnn':   #若使用递归神经网络
        cell_fun = tf.contrib.rnn.BasicRNNCell   #rnn基本模型
        # BasicRNNCell是最基本的RNN cell单元。
        # 输入参数：
        # num_units：RNN层神经元的个数
        # input_size（该参数已被弃用）
        # activation: 内部状态之间的激活函数
        # reuse: Python布尔值, 描述是否重用现有作用域中的变量

    elif model == 'gru':   #若使用LSTM的变体GRU   （计算效率更高，即训练速度更快）
        cell_fun = tf.contrib.rnn.GRUCell

    elif model == 'lstm':  #若使用LSTM即长短时记忆网络
        cell_fun = tf.contrib.rnn.BasicLSTMCell
        # BasicLSTMCell类是最基本的LSTM循环神经网络单元。
        # 输入参数： num_units: LSTM cell层中的单元数
        # forget_bias: forget gates中的偏置
        # state_is_tuple: 还是设置为True吧, 返回(c_state, m_state)的二元组
        # activation: 状态之间转移的激活函数
        # reuse: Python布尔值, 描述是否重用现有作用域中的变量

    #一个单元细胞定义
    cell = cell_fun(rnn_size)   #神经元数量为rnn_size，  返回元组

    # tf.contrib.rnn.MultiRNNCell()：将多个单元细胞堆叠后得到一个2层的lstm网络
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if output_data is not None:  #若目标值非空则进行训练
        #初始化状态
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:  #否则进行预测
        initial_state = cell.zero_state(1, tf.float32)

    with tf.device("/cpu:0"):  #指定代码运行在0号cpu上

        #定义嵌入矩阵   形状是 [6111, 128]  各元素为-1到1之间的均匀分布随机值
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))

        #输入矩阵   形状是[64, ?, 128]
        inputs = tf.nn.embedding_lookup(embedding, input_data)  #input_data为64行？列  所以索引出的张量形状为[64,?,128]
        #tf.nn.embedding_lookup(a,b)的用法是在张量a中用索引b选取元素

    #LSTM输出  outputs形状为[64, ?, 128]  last_state为包含两个元素的元组  每个元素是一个LSTMStateTuple
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)

    #改变outputs的形状  [?, 128]
    output = tf.reshape(outputs, [-1, rnn_size])

    #初始化权重   产生截断正态分布随机数  形状为 [128, 6111]
    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))

    #初始化偏置   形状为[6111]   初始化为0
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))

    #将偏置项加到 output * weight 上  结果形状是  [?, 6111]  logits即为LSTM最后一层的输出，即预测值
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)    #tf.matmul()为矩阵乘法
################################################################################################################################

    if output_data is not None:  #如果目标值数据非空说明调用模型进行训练

        #labels为目标值
        # output_data 必须使用 one-hot 编码  -->[?, 6111]   depth表示转化为one-hot编码后输出尺寸
        #先将output_data转化为一维张量，再转为one-hot编码
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)

        #计算交叉熵损失  （传入实际目标值和LSTM输出值） 形状为[?, 6111]
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        #将损失降维求和   total_loss即为损失函数
        total_loss = tf.reduce_mean(loss)

        #梯度下降最小化损失  并且采用效率更高的Adam
        #本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
        #Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        #保存当前批次的训练数据供返回
        end_points['initial_state'] = initial_state   #当前批次训练的初始状态
        end_points['output'] = output                 #当前批次的output
        end_points['train_op'] = train_op             #当前批次的train_op  即最终在sess中运行的op
        end_points['total_loss'] = total_loss         #当前批次的损失均值
        end_points['loss'] = loss                     #当前批次的损失  （高维度[?,6111]）
        end_points['last_state'] = last_state         #当前批次的最终状态
        
    else:  #如果目标值为空则说明要求调用模型进行预测

        #返回预测结果
        #通过Softmax回归，将logistic的预测二分类的概率的问题推广到了6111分类的概率的问题。
        prediction = tf.nn.softmax(logits)  #返回各个词出现的概率(归一化[0,1]即概率值) 形状为[?,6111]

        end_points['initial_state'] = initial_state  #初始状态状态
        end_points['last_state'] = last_state        #最终状态
        end_points['prediction'] = prediction        #最终预测结果

    return end_points, model   #以字典形式返回
