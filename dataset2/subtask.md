
先读取 report 看看前任做了什么

stage 1

先看看 ./data 或者 meta.csv 是否匹配，补充 meta.csv

最少包含以下内容:
小说、文学作品、文献
生活常识
概念说明
词典
逻辑
数学
情感

stage 2

写 read.py，构建 Dataset 类

通过 head -c 100 之类的手段看看数据是什么格式的，来兼容不同的 json
