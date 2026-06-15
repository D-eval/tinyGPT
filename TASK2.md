项目简介：

训练一个小型的 gpt 语言模型，
transformer decoder only 架构

环境:
```
conda activate py39
```

# stage 1
先调整模型架构，要确保 batch_size=8 的情况下 10k 上下文的情况下显存占用 50% 以上，而且能够稳定训练

完善 model.py 的测试
确保是 transformer decoder dense 架构
python3 model.py
用 dummy dataset 模拟 10k 上下文训练
然后看显存占用，
调整模型架构直到显存占用为 50% 以上而且不爆炸。

# stage 2

完善 read_union.py
是对
./dataset 和 ./dataset2
的整合

需要验证 read_union.py 包含了 ./dataset 和 ./dataset2 的所有数据
无遗漏

## streaming load

修改 ./dataset2/read.py
确保:
文本不能全部加载到内存里，最好先遍历一遍数据，保存一个加载顺序列表，保存为一个data.csv，路径+类型+行数，getitem的时候再加载数据，每条数据最多 10k 长度。超出这个长度的文本就进行切分。

禁止一次性将全部文本加载到内存的实现。

首次扫描数据集时遍历所有文件并生成 data.csv
记录：
path,file_type,item_index,start_offset,end_offset,text_length
至少包含：

路径
类型
数据项索引
文本长度

对于 jsonl 文件建议记录字节偏移量。

Dataset
实现真正的 Lazy Dataset：
dataset[i]
时：
根据 data.csv 定位数据
仅加载对应样本
不允许加载整个文件
文本切分
单条文本最大长度：
10000 字符
超过长度时自动切分：
sample_0
sample_1
sample_2
...
切分后的样本全部写入索引。

## index 的 index

考虑到
data.csv
本身也足够大，所以对保存一个
data.idx

内容：
[
    0,
    52,
    109,
    161,
    ...
]
表示：
第N行
在csv中的字节位置

读取：

csv_file.seek(
    idx[n]
)

Dataset init 时只需要把 data.idx 加载进内存即可。
getitem 时先获得 csv 的对应行内容，
然后再加载文本数据。

## preprocess

写 preprocess.py

用 tokenizer 预先处理 read_union 的所有数据，得到:

tokens_dataset1_train .bin 和 .idx
tokens_dataset2_train .bin 和 .idx
tokens_dataset1_valid .bin 和 .idx
tokens_dataset2_valid .bin 和 .idx

像这样：

tokens.bin
--------------------------------
123 54 67 89 234 ...
--------------------------------

tokens.idx
--------------------------------
sample0 -> 0
sample1 -> 10000
sample2 -> 20000
--------------------------------

preprocess 之后进行加载测试，用tokenizer解码，确保加载的数据正常

bin, idx 文件保存到了 ./preprocess 目录下

## 逐条保存

不行，对于 dataset2，还是每个数据分开保存成 bin, idx 比较好
这样新添数据之后才可以维护
也就是保存到
./dataset2/preprocess{上下文长度}/{data_name}.bin, {data_name}.idx
注意，最后一条 sample 有可能不满足{上下文长度}，可以剔除掉。

这个功能应该加到 ./dataset2 的 preprocess.py 里面
定期打印一下当前的文本预览，以及tokenizer处理后的结果

如果处理完某一个就保存一个 ./dataset2/preprocess/{data_name}.complete
运行的时候跳过这些 data_name

## 修改 read.py

然后修改 ./dataset2/read.py
Dataset init 的时候读取所有的 
./dataset2/preprocess{上下文长度}/{data_name}.idx
合并为一个列表

然后getitem只需要读取对应的 bin 文件的对应位置即可。

# stage 3

环境:

conda activate py39

完善
train2.py
test2.py

要求
调用 model.py 的 TinyGPT 为模型
以及 model.py 里的 Stage1Config 参数
batch_size 设置为 8
上下文窗口为 10k
保持 transformer decoder only 架构
train:valid = 99:1

## 数据集

## 权重分配
数据集为 read_union.py
其中，loss在不同数据集上有不同权重
dataset: 0.9 （风格化小说）
dataset2: 0.1 （通用知识）

如果训练无误，显存占用应该在 50 % 以上
如果没有达到这个值，就打断训练并检查哪一步出错了

训练若干个 epoch
保存 loss.png
用 nohup ... -u ... > train2.log 实时打印结果

每个epoch
模型保存为 ./params/model2.pt
tokenizer保存为 ./params/tokenizer2.json
可以读取检查点继续训练

对于 token_acc
要求在 dataset 上达到 0.98 以上
同时在 dataset2 上达到 0.8 以上

把当前的 token_acc 保存到 result2.json 里，
每个 epoch eval 一次

训练代码写到 train2.py,
评估代码写到 eval2.py
测试为 test2.py

完成工作之后写报告到 report
按照 1.txt, 2.txt, ...
往后写一个，作为你的报告

# stage 4

写文件

export_onnx.py
write.html
用户写一段话，按下续写按钮自动续写

把
/home/vipuser/wby/proj_params/tinyGPT/test2.py
模型导出为 onnx

然后网页：

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

加载：

const session =
    await ort.InferenceSession.create(
        "tinygpt.onnx"
    );

推理：

const outputs =
    await session.run(inputs);

上传到了 hugging face

然后接下来在 index.html 里直接加载 
https://huggingface.co/BoiWanKenobi/myNovelGPT/resolve/main/tinygpt.onnx
https://huggingface.co/BoiWanKenobi/myNovelGPT/resolve/main/tinygpt.onnx.json


nohup python3 -u train2.py --max-epochs 5000 > train2.log 2>&1 &

nohup python3 -u preprocess.py --max-epochs 5000 > preprocess.log 2>&1 &

这个 tokenizer 太大了，导致模型变得很大，不用兼容之前的 tokenizer 了，重新训练 tokenizer，写在 train_tokenizer.py 里，统计 read_union 的数据，得到 tokenizer2.json，保持 vocab_size = 16000

nohup python3 -u train_tokenizer.py --vocab-size 16000 --out params/tokenizer2.json --stats-out params/tokenizer2_stats.json --progress-every 10000  > train_tokenizer.log 2>&1 &



nohup python3 -u dataset2/preprocess.py > preprocess.log 2>&1 &



nohup python3 -u train_tokenizer.py --vocab-size 16000 --out params/tokenizer2.json --stats-out params/tokenizer2_stats.json --progress-every 10000 > train_tokenizer.log 2>&1 &
nohup python3 -u dataset2/preprocess.py > preprocess.log 2>&1 &
nohup python3 -u train2.py --max-epochs 5000 > train2.log 2>&1 &


nohup bash -c '
python3 -u train_tokenizer.py \
    --vocab-size 16000 \
    --out params/tokenizer2.json \
    --stats-out params/tokenizer2_stats.json \
    --progress-every 10000 &&
python3 -u dataset2/preprocess.py &&
python3 -u train2.py --max-epochs 5000
' > pipeline.log 2>&1 &

nohup bash -c '
python3 -u dataset2/preprocess.py &&
python3 -u train2.py --max-epochs 5000
' > pipeline.log 2>&1 &