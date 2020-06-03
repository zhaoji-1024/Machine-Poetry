import collections
import os
import sys
import numpy as np
import tensorflow as tf
from models.model import rnn_model
from dataset.poems import process_poems, generate_batch
import pymongo
import heapq
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   #消除tensorflow警告信息


#批处理数据大小(一次训练64首唐诗)
batch_size = 64
#学习率大小   默认0.01
learning_rate = 0.01
#训练数据源文件路径
file_path = './dataset/data/poems1.txt'
#训练代数
epochs = 50


start_token = 'G'  #诗句起始字符
end_token = 'E'    #诗句结束字符


def run_training():
    """
    模型训练
    :return: None
    """

    #调用process_poems方法预处理数据   并获取唐诗向量数据，汉字映射表，以及词汇表
    poems_vector, word_to_int, vocabularies = process_poems(file_path)

    #调用generate_batch方法获取批处理特征值和目标值
    batches_inputs, batches_outputs = generate_batch(batch_size, poems_vector, word_to_int)
    #batches_inputs和batches_outputs均为505块64行length列    length值不定

    #定义批处理的输入数据和输出目标数据的占位tensor  形状为 [64, ?]  length长度为不定值
    input_data = tf.placeholder(tf.int32, [batch_size, None])        #[64, ?]
    output_targets = tf.placeholder(tf.int32, [batch_size, None])    #[64, ?]

    #调用模型返回训练数据
    end_points = rnn_model(model='rnn', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=batch_size, learning_rate=learning_rate)

    #实例化一个模型保存对象供后续保存模型使用
    saver = tf.train.Saver(tf.global_variables())

    #创建初始化组合操作op   用于初始化全局变量和局部变量
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    #创建tf会话运行op
    with tf.Session() as sess:

        #初始化变量
        sess.run(init_op)

        #定义训练代数初始为第0代
        start_epoch = 0

        #打印提示训练开始
        print('[INFO] 开始训练...')

        try:
            #保存训练损失信息   后续写入mongodb数据库
            train_losses = []

            #循环epoch代进行训练
            for epoch in range(start_epoch, epochs):

                n = 0   #用于保存每个epoch下第几次训练的序号

                #计算训练一共有多少个数据块  每64首唐诗一个数据块  n_chunk实质为540，即全部数据包含540个批次
                n_chunk = len(poems_vector) // batch_size

                #当前epoch下进行540次循环训练，每次训练64首唐诗
                for batch in range(n_chunk):
                    #运行调用模型返回的op并且传入当前批次的特征值和目标值，即之前定义好的占位张量
                    #这里只接收返回的损失函数，训练状态以及最小化损失op只运行，无需保存返回值
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})

                    n += 1  #训练批次序号+1

                    #打印提示当前epoch序号，batch序号， 当前交叉熵损失
                    print('[INFO] epoch序号: %d , batch序号: %d , 当前交叉熵损失: %.6f' % (epoch, batch, loss))

                    #保存当前批次训练信息供后续写入mongoDB
                    train_loss = {"epoch序号": epoch, "batch序号": batch, "当前交叉熵损失": float(loss)}
                    train_losses.append(train_loss)

                #每训练6个epoch保存一次模型    global_step=epoch表示将epoch序号加入到保存模型文件后缀
                if epoch % 6 == 0:
                    saver.save(sess, './model/rnn_model/', global_step=epoch)

            #4代训练全部完成时打印提示信息
            print("[INFO] 训练已全部完成")

            #将训练信息写入数据库
            # 创建一个mongodb连接对象
            myclient = pymongo.MongoClient("mongodb://localhost:27017/")
            # 创建一个数据库  名为train_loss
            mydb = myclient["train_loss"]
            # 创建一个集合
            mycol = mydb["rnn_train_loss"]
            # 向集合插入文档
            mycol.insert_many(train_losses)
            #关闭数据库连接
            myclient.close()

        except KeyboardInterrupt:  #处理用户中断执行异常

            print('[INFO] 训练出现异常中断')



def to_word(predict, vocabs):
    """
    将预测返回的predict转为词汇表中对应的汉字
    :param predict: 概率表
    :param vocabs: 词汇表
    :return:  本次预测的汉字
    """
    #将predict转化为一维数组并且将第i各元素赋值为前i个元素的和  此时t必定是升序数组
    t = np.cumsum(predict)

    #对概率表求和
    s = np.sum(predict)

    #np.searchsorted(a,b)用于寻找b在a中的位置   a为升序数组
    #np.random.rand(1)返回一个包含一个[0，1)均匀分布的随机值的列表
    #sample接收t中np.random.rand(1) * s在t中的下标，即本次预测的汉字在词汇表中的下标
    sample = int(np.searchsorted(t, np.random.rand(1) * s))

    if sample > len(vocabs):
        sample = len(vocabs) - 1

    #返回本次预测的汉字
    return vocabs[sample]


def gen_poem(begin_word):
    """
    产生诗句方法
    :param begin_word: 作诗首个汉字
    :return:
    """
    batch_size = 1

    #打印提示信息
    print('[INFO] 正在从 %s 中加载语料库' % file_path)

    #获取唐诗向量， 映射表， 词汇表
    poems_vector, word_int_map, vocabularies = process_poems(file_path)

    #定义输入数据占位符  形状为[1, ?]
    input_data = tf.placeholder(tf.int32, [batch_size, None])

    #调用模型进行诗句预测   （output_data=None则为进行诗句预测）
    end_points, model_type = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=learning_rate)

    #实例化一个保存模型对象
    saver = tf.train.Saver(tf.global_variables())

    #定义op组合（初始化全局变量和局部变量）
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    #定义会话   运行op
    with tf.Session() as sess:

        #初始化变量
        sess.run(init_op)

        #恢复先前训练保存好的模型
        saver.restore(sess, './model/lstm_model/-48')

        #输入数据初始化为"G"即语料库的诗句起始字符
        x = np.array([list(map(word_int_map.get, start_token))])# x为np.ndarray  [[2]]  形状是[1, 1]

        #运行预测op返回预测开始字符G后的最终状态供下次预测使用
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],feed_dict={input_data: x})

        #如果正确输入首个汉字且该字在词汇表当中则继续进行诗句预测
        if begin_word in vocabularies:
            word = begin_word
        #否则打印提示错误信息，结束预测
        else:
            print("键入值错误或词汇表中不包含该汉字，请输入正确的汉字!")
            return None

        #初始化诗句为空字符串
        poem = ''

        #预测诗句直至预测到结束符E结束
        while word != end_token:
            #print('预测中...')
            poem += word
            #重置输入数据
            x = np.zeros((1, 1))
            #将上个预测出的汉字作为下次预测的输入
            x[0, 0] = word_int_map[word]
            #进行预测，返回预测结果以及预测状态 predict形状为[1,6111],表示个汉字出现的概率
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})

            #将预测结果转为为对应词汇表中的汉字
            word = to_word(predict, vocabularies)

        #返回预测的诗句结果
        return poem, model_type


def batch_gen_poem(rand_words):
    """
    批量产生诗句方法
    :param begin_word: 作诗首个汉字
    :return:
    """
    batch_size = 1

    #打印提示信息
    print('[INFO] 正在从 %s 中加载语料库' % file_path)

    #获取唐诗向量， 映射表， 词汇表
    poems_vector, word_int_map, vocabularies = process_poems(file_path)

    #定义输入数据占位符  形状为[1, ?]
    input_data = tf.placeholder(tf.int32, [batch_size, None])

    #调用模型进行诗句预测   （output_data=None则为进行诗句预测）
    end_points, model_type = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=learning_rate)

    #实例化一个保存模型对象
    saver = tf.train.Saver(tf.global_variables())

    #定义op组合（初始化全局变量和局部变量）
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    #定义会话   运行op
    with tf.Session() as sess:

        #初始化变量
        sess.run(init_op)

        #恢复先前训练保存好的模型
        saver.restore(sess, './model/lstm_model/-48')

        #输入数据初始化为"G"即语料库的诗句起始字符
        x = np.array([list(map(word_int_map.get, start_token))])# x为np.ndarray  [[2]]  形状是[1, 1]

        #运行预测op返回预测开始字符G后的最终状态供下次预测使用
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],feed_dict={input_data: x})

        #存储本批次产生的全部诗文
        poems = []

        for rand_word in rand_words:
            #初始化诗句为空字符串
            poem = ''

            #预测诗句直至预测到结束符E结束
            while rand_word != end_token:
                #print('预测中...')
                poem += rand_word
                #重置输入数据
                x = np.zeros((1, 1))
                #将上个预测出的汉字作为下次预测的输入
                x[0, 0] = word_int_map[rand_word]
                #进行预测，返回预测结果以及预测状态 predict形状为[1,6111],表示个汉字出现的概率
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x, end_points['initial_state']: last_state})

                #将预测结果转化为对应词汇表中的汉字
                rand_word = to_word(predict, vocabularies)
            #将当前诗文追加到poems
            poems.append(poem)

        #返回预测的全部诗句结果和模型类型
        return poems, model_type

def save_poems(poem, mosel_type):
    """
    将所作诗文写入数据库
    :param poem:
    :param mosel_type:
    :return: None
    """
    # 创建一个mongodb连接对象
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    # 创建一个数据库  名为train_loss
    mydb = myclient["generate_poems"]
    # 创建一个集合
    mycol = mydb[mosel_type + "_poems"]
    # 向集合插入文档
    mycol.insert({"诗文": poem})
    # 关闭数据库连接
    myclient.close()

    return "本次写入数据库成功..."

def pretty_print_poem(poem):
    """
    打印预测出的诗句
    :param poem: 预测诗句字符串
    :return: None
    """
    #如果正确预测出诗句则打印输出
    if poem:
        #把字符串按。分割返回列表  目的是将预测出超过两句的诗文每句一行打印
        poem_sentences = poem.split('。')
        for s in poem_sentences:
            #若预测诗句长度大于10则正确打印
             if  len(s) > 10:
                print(s + '。')

    return None


def main(is_train):

    #若命令行要求训练模型  则执行训练
    if is_train:
        print('[INFO] 模型训练中...')
        run_training()

    #若命令行要求调用模型  则执行调用
    else:
        print('[INFO] 模型写诗中...')

        begin_word = input('请输入起始字或批量作诗数量: ')

        #如果输入字符为整数，则进行随机批量写诗并写入数据库
        if begin_word.isdigit():
            #获取到词汇表
            _0, _1, words = process_poems("./dataset/data/poems.txt")
            #保存批量首字
            rand_words = []
            #循环从词汇表中生成批量首字
            for i in range(int(begin_word)):
                #首先从词汇表前1000个字中获取一个随机字作为本次作诗的首字
                rand_word = words[random.randint(3,1000)]
                rand_words.append(rand_word)
            #调用批量作诗方法返回诗文
            poems, model_type = batch_gen_poem(rand_words)

            #将诗句写入数据库
            for poem in poems:
                info = save_poems(poem, model_type + "_batch")
                print(poem + "\t" + info)

        else:
            poem2, model_type = gen_poem(begin_word)

            #将本次作诗存入数据库
            save_poems(poem2, model_type)

            #控制台打印诗句
            pretty_print_poem(poem2 + "\t" + "写入数据库成功...")


if __name__ == '__main__':
    tf.app.run()
