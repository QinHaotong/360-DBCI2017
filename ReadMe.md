#360文本分类的cnn实现

2.0

1. 改动了模型参数
2. loader.py将词典生成改为POSITIVE标签数据的精确分词
3. group.py将训练集和测试集进行了划分方便载入

---------

原始模型改动自：https://gaussic.github.io/2017/08/30/text-classification-cnn-tensorflow/

添加了详细注释

配置信息：

pycharm

anaconda

python3.5

tensorflow-gpu

cuda8.0

cudnn5.0