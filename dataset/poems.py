import collections
import os
import sys
import numpy as np

start_token = 'G'  #诗句开始标识符
end_token = 'E'    #诗句结束标识符

def process_poems(file_name):
    """
    唐诗数据预处理
    :param file_name:  数据文件路径
    :return:
    """
    poems = []  #存储处理后诗句的列表

    with open(file_name, "r", encoding='utf-8') as f:   #以只读方式打开唐诗数据文件到对象f

        for line in f.readlines():  #循环读取每行唐诗数据到  line  每个line是一首诗用一个字符串保存

            try:
                #删除唐诗首尾的空格后分割唐诗的标题和内容分别用变量保存
                title, content = line.strip().split(':')
                # 把诗句中的全部空格去掉
                content = content.replace(' ', '')

                #如果当前诗句中出现特殊符号则舍弃  (提前结束本次读取)
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or start_token in content or end_token in content or "—" in content:
                    continue

                #如果当前诗句过长或过短则舍弃  (提前结束本次读取)
                if len(content) < 5 or len(content) > 79:
                    continue

                #给诗句加上开始和结束标识符   并将诗句加入到poem[]列表当中
                content = start_token + content + end_token
                poems.append(content)

            except ValueError as e:
                pass

    #至此poems列表长度为32376   即poems中保存32376首诗文

    # 按诗的字数排序  reverse=True表示递减排序   最长诗文长度78  最短8
    poems = sorted(poems, key=lambda i: len(i), reverse=True)

    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]   #把所有诗句的单个汉字作为单个元素保存在列表all_words中
    #all_words长度为1704787  即包括起始和结束符一共出现近170万五千字

    #调用collections模块的Counter统计各个汉字出现的次数
    # counter为一个字典  键为唐诗中出现的汉字(不重复) 对应值为该汉字在唐诗中出现的次数
    counter = collections.Counter(all_words)

    #counter.items()返回一个列表，列表元素为元组，元组内两个元素分别代表汉字和该汉字出现次数
    #调用sorted()对上列表按元组第二个值递减排序，即出现次数多的汉字排在前面
    count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    #调用zip方法解压coumt_pairs
    #返回words和_都是元组  words中保存出现的所有单个汉字  _中保存对应汉字的出现次数
    words, _ = zip(*count_pairs)
    #word长度为6029   即词汇表大小为6029

    # 取前多少个常用汉字  过滤掉一些出现概率极低的汉字()
    words = words[:len(words)] + (' ',)

    # 每个汉字映射为一个数字
    #返回word_int_map为一个字典 键为汉字  对应值为数字
    word_int_map = dict(zip(words, range(len(words))))  #第n个汉字映射为数字n-1


    #返回唐诗数据映射的向量
    #map()用作映射  .get()获取字典指定键的对应值
    poems_vector = [list(map(lambda word: word_int_map.get(word), poem)) for poem in poems]
    #映射表为二维列表，包含32376个子列表，每个子列表代表一首诗的向量

    #返回处理后的唐诗向量数据，映射字典数据(映射表)，以及所有汉字(词汇表)
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    """
    批处理数据制作
    :param batch_size:  批处理大小  默认为64首诗一批
    :param poems_vec:   唐诗向量数据
    :param word_to_int: 汉字映射表
    :return:  批处理特征值和目标值
    """

    # 每次取batch_size(64)首诗进行训练   计算一代需要训练多少次用n_chunk保存
    n_chunk = len(poems_vec) // batch_size
    #n_chunk为32376 // 64 = 505   一代训练需要505次

    x_batches = []  #存放全部批次的特征值
    y_batches = []  #存放全部批次的目标值

    for i in range(n_chunk):  #循环n_chunk次  制作n_chunk个batch数据

        start_index = i * batch_size  # 0  64  128  192 ...本批次开始下标
        end_index = start_index + batch_size   #本批次结束下标

        #取出当前批次的64个向量数据
        batches = poems_vec[start_index:end_index]

        # 找出这个batch的所有poem中最长的poem的长度
        max_poem_length = max(map(lambda s: len(s), batches))

        # 填充一个形状为[64, length]的空batch，空的地方放空格对应的数字  即6029
        x_data = np.full((batch_size, max_poem_length), word_to_int.get(" "), np.int32)

        for row in range(batch_size):
            # 每一行就是一首诗，在原本的长度上把诗还原上去,至此制作好本批次的特征值
            x_data[row, :len(batches[row])] = batches[row]

        #制作本批次的目标值  先复制成特征值
        y_data = np.copy(x_data)    #x_data和y_data都是np.ndarray,形状是64行length列

        # 目标值即想要预测的诗句    y = x向左边也就是前面移动一个
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """

        #将本批次的batch数据追加到batches中
        x_batches.append(x_data)
        y_batches.append(y_data)
        #x_batches和y_batches形状为505块 每块64行  length列   每块是一个np.ndarray
        #length 为当前批次的最长诗文的长度

    return x_batches, y_batches


