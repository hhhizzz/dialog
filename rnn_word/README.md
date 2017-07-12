## 介绍
`main.py`为rnn模型，用于训练语义解析模型
`best_model_weights.h5`是训练30次后的模型权重数据，可以直接读取,效果为`Precision = 94.11, Recall = 95.19, F1 = 94.65`

`talk.py`为测试效果的脚本，运行后输入句子会直接返回使用`best_model_weights.h5`的标注
```shell
python talk.py
i need to know information for flights leaving dallas on tuesday evening and returning to atlanta
```
返回值为：
```shell
i need to know information for flights leaving dallas on tuesday evening and returning to atlanta
```