import argparse
from inference import tang_poems

def parse_args():

    #创建一个命令行参数解析对象
    parser = argparse.ArgumentParser(description='Intelligence Poem')

    #参数的说明信息，简单描述参数的作用
    help_ = 'choose to train or generate.'

    #向命令行参数解析对象添加命令行参数信息
    parser.add_argument('--train', dest='train', action='store_true', help=help_)  #命令行输入--train则为训练模型
    parser.add_argument('--no-train', dest='train', action='store_false', help=help_)#命令行输入--no-train则为调用模型

    #设定命令行参数默认值为False  意味执行模型调用写诗
    parser.set_defaults(train=False)

    #调用parse_args返回对象供使用设定好的命令行参数对象
    args_ = parser.parse_args()

    return args_


if __name__ == '__main__':

    #获取命令行参数对象
    args = parse_args()

    #若命令行参数要求训练  则执行模型训练
    if args.train:
        tang_poems.main(True)

    #若命令行参数要求模型调用  则执行模型调用
    else:
        tang_poems.main(False)
