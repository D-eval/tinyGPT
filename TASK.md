项目简介：

训练一个小型的 gpt 语言模型，

先 nvidia-smi 一下，看看有多大显存，然后确定模型体量。
确保能在5G的内存上推理。

dataset 是我的小说合集，我希望训练一个能够模仿我文笔的小模型。
同时不要用太多通用知识污染这个模型。

环境:
```
conda activate py310
```

0、如果没有 result.json，先创建它，用来储存结果，如果已经有了，那就看看里面的结果，如果已经到达 0.7 以上，那就不用做后面的任务了。如果 ./param/model.pt 已经存在，先跑一次评估，如果达到要求就不用做了。
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
在我的dataset上预测下一token，token_acc 达到 0.7 以上即可。