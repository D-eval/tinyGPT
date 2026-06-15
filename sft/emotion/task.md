
从这些地方获得数据集，保存到 ./data

https://arxiv.org/abs/1710.03957

https://www.kaggle.com/datasets/atharvjairath/personachat

https://arxiv.org/abs/2306.08401


并写 read.py，加载数据集 Dataset
getitem:

<system>:...
<usr>:...
<agent>:...

格式加载文本


https://arxiv.org/abs/2103.02548
https://arxiv.org/abs/2205.14727

你怎么把好多连续的对话拆开了，保留原来的多轮对话
<usr>:
<agent>:
<usr>:
<agent>:
...


ds = Dataset(source="cped", split="train")


dailydialog 10429
personachat 64670
livechat 45
naturalconv 19354
cped 14786

dailydialog 10429
"<system>: You are a helpful dialogue agent. Respond naturally across the full conversation.\n<usr>: Say, Jim, how about going for a few beers after dinner?\n<agent>: You know that is tempting but is really not good for our fitness.\n<usr>: What do you mean? It will help us to relax.\n<agent>: Do you really think so? I don't. It will just make us fat and act silly. Remember last time?\n<usr>: I guess you are right. But what shall we do? I don't feel like sitting at home.\n<agent>: I suggest a walk over to the gym where we can play singsong and meet some of our friends.\n<usr>: That's a good idea. I hear Mary and Sally often go there to play pingpong. Perhaps we can make a foursome with them.\n<agent>: Sounds great to me! If they are willing, we could ask them to go dancing with us. That is excellent exercise and fun, too.\n<usr>: Good. Let' s go now.\n<agent>: All right."

personachat 64670
'<system>: You are a persona-consistent dialogue agent.\nPersona:\n- i like to remodel homes.\n- i like to go hunting.\n- i like to shoot a bow.\n- my favorite holiday is halloween.\n<usr>: hi , how are you doing ? i am getting ready to do some cheetah chasing to stay in shape .\n<agent>: you must be very fast . hunting is one of my favorite hobbies .\n<usr>: i am ! for my hobby i like to do canning or some whittling .\n<agent>: i also remodel homes when i am not out bow hunting .\n<usr>: that is neat . when i was in high school i placed 6th in 100m dash !\n<agent>: that is awesome . do you have a favorite season or time of year ?\n<usr>: i do not . but i do have a favorite meat since that is all i eat exclusively .\n<agent>: what is your favorite meat to eat ?\n<usr>: i would have to say its prime rib . do you have any favorite foods ?\n<agent>: i like chicken or macaroni and cheese .\n<usr>: do you have anything planned for today ? i think i am going to do some canning .\n<agent>: i am going to watch football . what are you canning ?\n<usr>: i think i will can some jam . do you also play footfall for fun ?\n<agent>: if i have time outside of hunting and remodeling homes . which is not much !'

livechat 45
'<system>: You are a Chinese live-streaming dialogue agent. Reply as the streamer.\nBasic profile: {"age": 2, "audiences": 1, "character": 9, "fans_num": 1, "gender": "1", "live_time": 2, "location": "安徽", "reply_barrage": 1, "skill": null}\nText profile:\n我不想再再那个了认清现实了\n我睡觉从来不打呼但是有有好像梦梦游过\n我今天开光了因为晚上\n嗯我还有几年到30岁我觉得我已经提前跨到这个坎了\n无人区在哪里啊我也想去我也要找一个没有人认识我的地方\n是的我跟你讲我最怕石头我最不喜欢石头了\n我余额不足了发不了红包了要命要命要命\n莫干山我是前年去的\n<usr>: 音乐大，说话声音带混响\n<agent>: 那我怎么调啊？'

naturalconv 19354
'<system>: You are a Chinese multi-turn topic-driven dialogue agent. Reply naturally and keep the topic transition smooth.\n<usr>: 你好！\n<agent>: 你好啊！\n<usr>: 你也是来看明星走红毯的嘛？\n<agent>: 对啊，我本来是来三亚旅游的，听说这里举办电影节，就想过来见见世面。\n<usr>: 那你可是来对了，这次可以说是众星云集，明星可以看到你想吐的。\n<agent>: 那我可要好好等等看完红毯仪式了，不知道有没有我的沈腾叔叔。\n<usr>: 这种级别的电影节肯定有他啊，再怎么说别人现在票房总量累计也是超过100亿的。\n<agent>: 我最开始喜欢他是从2012年春晚他的小品，“郝建”贱贱的小样子太好笑了。\n<usr>: 我也是他的粉丝，他每年春晚的小品的台词我都可以背出来，主要是太经典了。\n<agent>: 而且他现在在电影大荧幕里面表现的也不错，他们开心麻花出品的电影每一部都是精品。\n<usr>: 有句话不是说沈腾什么都不用干，就站在那里，你就想笑。\n<agent>: 哈哈哈，确实，他那张脸就会让人不由自主想笑，他一开口我就忍不住了。\n<usr>: 这次听说沈腾就是带着新作品过来的，不知道是什么类型的。\n<agent>: 我知道，他之前采访的时候说了，叫《全民狂欢》，应该也是一部喜剧电影。\n<usr>: 虽然也很期待，但还是希望能看到他能演演不同类型的电影和角色，毕竟在好的演员同类型也会看腻的。\n<agent>: 说的也对，我觉他可以试试都市剧，就演那种不着调的角色很适合他。话说你这次是来看那个明星的？\n<usr>: 我是来等迪丽热巴的，他可是我的女神，简直就是仙女下凡。\n<agent>: 我也觉得她长得很漂亮，特别是那双眼睛，很有灵气。\n<usr>: 啊，我的迪丽热巴来了，我的先过去了，拜拜。\n<agent>: 再见。'

cped 14786
'<system>: You are a Chinese personalized and emotional dialogue agent.\nContinue the multi-party conversation as the target speaker.\nTarget speaker profile: {"age": "young", "agreeableness": "high", "conscientiousness": "high", "extraversion": "low", "gender": "male", "neuroticism": "low", "openness": "high"}\nConversation conditions: scene=restaurant.\n<usr>: 几天不见 你的思想觉悟有了空前的提高啊\n<agent>: 你能不能答应我 在我离开北京这段时间里\n你不和别人相亲\n也不和别人谈恋爱\n<usr>: 我是你什么人哪 我凭什么答应你\n<agent>: 你不答应我我怎么离开呀'

ds = Dataset(source="dailydialog", split="train")

这个数据集有问题，要让 agent 分别扮演这几个人，格式 <usr>:... <agent>:...

另外，把人名都去掉

不要修改raw的 jsonl 文件，可以另外保存一份

要让 agent 分别扮演每个人，你有这样做吗



扩充数据，剧本数据

https://github.com/longyuewangdcu/tvsub


把 /home/vipuser/wby/proj_params/tinyGPT/sft/emotion read.py
整合到 /home/vipuser/wby/proj_params/tinyGPT/read_union.py 里，和 dataset, dataset2 一起训练，作为 dataset_emotion_sft
而且要加入 <bos> <eos> token
以及只预测 <agent>: 的部分
在 train2.py 中，为 dataset_emotion_sft 分配 0.9 的权重

检查一下 read_union.py 对 dataest_emotion_sft 的读取，确保
1、只训练 <agent>: 的输出位置
2、agent每句话含有 <bos> <eos>

检查一下 train2.py，确保 dataest_emotion_sft 
1、只训练 <agent>: 的输出位置
2、agent每句话含有 <bos> <eos>

在 read_union.py 的检查中，可以指定 dataset 的选择是
dataset, dataset2, dataset_emotion_sft



python3 read_union.py --datasets dataset_emotion_sft --progress-every 5000

由于加入了 dataset_emotion_sft，需要重新训练 tokenizer
尽可能和
/home/vipuser/wby/proj_params/tinyGPT/params/tokenizer2.json
保持一致
同时加入 <bos> <eos> <agent> <usr> 以及新的token
模型除了 embedding 和 head，其它部分不变，
在 train2.py 中
不严格 的加载 /home/vipuser/wby/proj_params/tinyGPT/params/model2.pt
进行训练