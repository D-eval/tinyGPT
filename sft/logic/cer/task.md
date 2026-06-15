# CER（Concept Extracting Retokenizer）任务规划

## 项目目标

探索一种区别于传统 Tokenizer 的推理架构。

传统大语言模型：

```text
文本
↓
Tokenizer
↓
Transformer
↓
答案
```

CER 架构：

```text
文本
↓
Concept Extractor
↓
临时变量（Var Tokens）
↓
Retokenizer
↓
Logic Transformer
↓
自然语言生成
```

目标是将：

* 语言理解
* 逻辑推理
* 自然语言表达

三个过程解耦。

---

# 核心思想

对于逻辑推理任务：

```text
所有鸟都会飞。
小明是鸟。
小明会飞吗？
```

模型真正需要推理的是：

```text
Bird
Fly
Tom
```

之间的关系。

而不是：

```text
鸟
飞
小明
```

这些字符串本身。

因此设计：

```text
Bird -> var1
Fly  -> var2
Tom  -> var3
```

然后在小词表上进行推理。

---

# 整体架构

## Step1 Concept Extractor

输入：

```text
原始文本
```

输出：

```text
var0 = Tom
var1 = Bird
var2 = Fly
```


预计预留：

```text
100个变量槽位
```

即：

```text
var0 ~ var99
```

实现方案：

### subtask 文本分割模型

找一个现有的文本分割模型。
模型参数保存到 ./params

要求：
```
从自然语言中抽取一阶逻辑符号（FOL Symbols）。

包括：
1. Constant（个体）
2. Predicate（谓词）
3. Relation（关系）

忽略：
1. 量词（all, some, no, every）
2. 连词（and, or, not）
3. 代词（someone, they）
4. 时态与语法词
```

测试：
从 read.py 里加载 dataset，能正确从 dataset[0] <usr> 的话里提取出所有名词、谓词

```
'<usr>: Context:\nNo criminal is kind.\nAll person who breaks the law is a criminals.\nPeople are either kind or evil.\nIf someone is evil, then they are ugly.\nIf someone is evil, then they are cold-blood.\nIf Garry is either evil and ugly or neither evil nor ugly, then Garry is not evil.\n\nQuestion:\nDoes the conclusion follow from the premises? Conclusion: If Garry is evil or breaks the law, then Garry is not both a criminal and breaking the law.\n\nOptions:\nTrue\nFalse\nUncertain\n\nAnswer:\n<agent>: Formal logic:\npremises-FOL:\n∀x (Criminal(x) → ¬Kind(x))\n∀x (BreakLaw(x) → Criminal(x))\n∀x (Kind(x) ⊕ Evil(x))\n∀x (Evil(x) → Ugly(x))\n∀x (Evil(x) → ColdBlood(x))\n((Evil(garry) ∧ Ugly(garry)) ⊕ (¬Evil(garry) ∧ ¬Ugly(garry))) → ¬Evil(garry)\n\nAnswer:\nTrue'
```

输出格式：

```
entities:
- garry

predicates:
- Criminal
- Kind
- BreakLaw
- Evil
- Ugly
- ColdBlood
```

目标：

```
抽取出的 symbol 集合
应尽可能覆盖 Formal Logic 中出现的

Criminal
Kind
BreakLaw
Evil
Ugly
ColdBlood
garry
```

评测标准：

给定数据中的：

<usr>

以及对应：

<agent>: Formal Logic

统计：

Recall =
抽取出的逻辑符号
/
Formal Logic 中出现的逻辑符号

目标：

Recall > 95%

### 用开源的逻辑抽取模型

不要用作弊的方法统计，构建词表，要求有一定的泛化能力，可以用一些开源的模型，模型参数保存到 ./params

可以尝试这些开源模型：

https://github.com/CogComp/SRL-English.git

https://arxiv.org/abs/2010.03147

---

啊不，test.py的输入只包含<usr>:的部分，然后用提取结果和真实的<agent>:里的进行对比

## Step2 Retokenizer

将原始文本转换为变量表示。

例如：

```text
All birds fly.
Tom is a bird.
```

转换为：

```text
forall x:
    var1(x) -> var2(x)

var1(var0)
```

目标：

* 降低词汇冗余
* 提高组合泛化能力
* 统一不同领域概念

---

## Step3 Logic Transformer

只允许生成逻辑词表。

禁止自然语言。

逻辑词表：

```text
var0 ~ var99

forall
exists
not
and
or
implies
equal

true
false
uncertain
```

示例：

```text
<logic>

forall x:
    var1(x) -> var2(x)

var1(var0)

therefore

var2(var0)

answer true

</logic>
```

---

## Step4 Talk Decoder

根据 Logic Transformer 输出结果生成自然语言。

例如：

```text
answer true
```

转换为：

```text
会
```

或者：

```text
True
```

---

# 数据集

## 第一阶段

ProofWriter

特点：

* 自动生成
* 逻辑链明确
* 支持 True/False/Unknown

---

## 第二阶段

PrOntoQA

特点：

* 虚构词汇
* 不依赖常识
* 测试纯逻辑能力

---

## 第三阶段

FOLIO

特点：

* 自然语言
* 对应一阶逻辑
* 包含复杂量词

---

# 实验设计

## Baseline

普通 Transformer

```text
Context
↓
Transformer
↓
Answer
```

---

## CER

```text
Context
↓
Concept Extractor
↓
Retokenizer
↓
Logic Transformer
↓
Answer
```

---

# 评测指标

## In-Domain Accuracy

原始测试集准确率。

---

## OOD Accuracy

随机替换概念名称。

例如：

训练：

```text
Bird
Fly
Tom
```

测试：

```text
Wumpus
Blicket
Zog
```

观察准确率下降程度。

---

## Concept Compression Ratio

统计：

```text
原始 Token 数
```

与

```text
Retokenize 后 Token 数
```

比值。

---

## Logic Vocabulary Usage

统计：

```text
逻辑词表覆盖率
```

以及：

```text
平均推理长度
```

---

# 长期目标

最终实现：

```text
自然语言
↓
概念抽取
↓
逻辑表示
↓
逻辑推理
↓
自然语言
```

并进一步扩展至：

* Lean
* 一阶逻辑
* 数学定理证明
* 康德范畴推理
* 长上下文知识组织

形成一种不同于传统 BPE Tokenizer 的对象化推理架构。
