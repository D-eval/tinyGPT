
目标：
获得逻辑训练 sft 数据
让模型获得基础逻辑能力

数据来源：

https://github.com/Yale-LILY/FOLIO.git

https://arxiv.org/pdf/2310.18659

https://arxiv.org/abs/2410.09207

https://arxiv.org/abs/2203.15099

从这些地方获得数据的下载地址

把数据整理到 ./data 下

然后写 read.py 构造 Dataset 类加载数据。

read.py 的加载格式统一成
<usr>: ...
<agent>: ...

dataset.getitem 的内容只要最终的文本格式，对每个数据集都改成
<usr>: ...
<agent>: ...
而且不要遗漏信息

agent的回复里的逻辑推导要保留

