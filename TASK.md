项目简介：

训练一个小型的 gpt 语言模型，
transformer decoder only 架构

先调整模型架构，要确保 10k 上下文的情况下显存占用 50% 以上


先 nvidia-smi 一下，看看有多大显存，
需要支持 10k 上下文，
在此基础上尽可能利用显卡。
同时模型大小不能超过 5GB

dataset 是我的小说合集，我希望训练一个能够模仿我文笔的小模型。
同时不要用太多通用知识污染这个模型。

环境:
```
conda activate py39
```

注意用 device

# stage 1

0、如果没有 result.json，先创建它，用来储存结果，如果已经有了，那就看看里面的结果，如果已经到达 0.8 以上，那就不用做后面的任务了。如果 ./param/model.pt 已经存在，先跑一次评估，如果达到要求就不用做了。
1、在dataset里递归搜索所有的txt文件作为数据集，统计最长样本的上下文长度，然后统计内存占用。
2、训练 bpe tokenizer, vocab_size=16000
3、必要时进行 linear attention 压缩，确保当前显存可以训练，5G 内存可以推理。如果内存足够就不用linear attention压缩。
模型写在model.py，评估写在eval.py，每个epoch保存参数，记得读取之前的model.pt。
训练的时候用nohup 这样结果我能实时看到

4、训练之后，参数保存到./params/model.pt，评估：把 Accuracy 写入 result.json 即可。格式类似于:

```
{
    "token_acc": 0.72,
    "ppl": 8.1,
    "params": 42000000
}
```

5、在 test.py 中测试模型续写能力。

目标：
不需要划分训练验证集
在我的dataset上预测下一token，token_acc 达到 0.8 以上即可。

stage 1 已经完成，现在开始 stage 2

# stage 2 通用知识、常识

模型保存为 ./params/model2.pt tokenizer保存为 ./params/tokenizer2.json

注意，要确保
1、模型是 decoder only 架构
2、确保加载 10k token 后显存利用在 50% 以上

如果没有 result2.json，创建一个
里面储存了在 dataset 上 和 dataset2 上的 token_acc
要求在 dataset 上达到 0.9 以上
同时在 dataset2 上达到 0.6 以上
如果已经达到了要求，不用执行后面任务


1、cd 到 dataset2 里，完成 subtask.md（已完成：补全 get_data.py/read.py/vertify.py）
2、完成 read_union
把 ./dataset2 和 ./dataset 的数据合成一个数据集
3、训练并保存到 result2.json
训练代码写到 train2.py, 测试为 test2.py


完成工作之后写报告到 report
按照 1.txt, 2.txt, ...
往后写一个，作为你的报告
