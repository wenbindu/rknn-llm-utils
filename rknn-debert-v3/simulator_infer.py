from transformers import DebertaV2Tokenizer
import numpy as np
from rknn.api import RKNN


# --------------------------------------------------
# Simulator-inference.
# --------------------------------------------------
tokenizer = DebertaV2Tokenizer.from_pretrained("./")
ONNX_MODEL = './deberta_nofix_opset_18.onnx'
# 加载onnx
rknn = RKNN(verbose=True)
rknn.config(
     target_platform='rk3588',   # 根据芯片型号修改
     optimization_level=3,       # 最高优化级别
     quantized_dtype='w8a8',  # 量化类型
     quantized_algorithm='normal', quantized_method='channel',
 )

rknn.load_onnx(
     model=ONNX_MODEL,
     inputs=['input_ids', 'attention_mask'],
     input_size_list=[[1, 128], [1, 128]],  # 固定输入尺寸
     outputs=['logits']
 )

rknn.build(
     do_quantization=False, 
     rknn_batch_size=1
 )

# Inference with Examples
rknn.init_runtime()

for txt in [
    "交警队出盖章文件了！hw智驾连环车祸+行车记录仪内存卡被偷最新动态！",
    "坦克500 Hi4-Z、Hi4-T到底啥区别？应该怎么选？一条视频就明白了！#新车抢先看",
    "全路况XT5遇上全满贯 @樊振东，会有怎样的精彩故事？更多内容，敬请期待！ #樊振东的选择##全新XT5 ##莫 问前路尽是坦途##凯迪拉克 樊振东# http://t.cn/A6uKBhoH ​​",
    "韩正会见马斯克！谈到特斯拉！马斯克当面表态 #美国#中国#马斯克#特斯拉#中美",
    "空间大开，热爱便没有障碍。 极氪春节贺岁档之《明日狮王》， 看#极氪MIX#打开爱的大门。 新年祝你极氪让爱满怀！ #新年新愿极氪实现# #极氪# http://t.cn/A63A74MZ ​​",
    "Model Y焕新版要来，有中国车企能干赢它吗？ #modely焕新版 #小米yu7 #电车 #国产新能源汽车 #买车",
    "梦想不会老去，岁月永远年轻。 极氪春节贺岁档之《黄色闪电》， 看#极氪007# 圆梦重庆老爸的速度与激情。 新年祝你极氪圆梦！ #新年新愿极氪实现# #极氪# http://t.cn/A63hKxcL ​​",
    "这么硬核的泛越野，36.38万！我新年的第一辆新车就它了！#开年第一车一定红#坦克500hi4z一定红",
    "特斯拉皮卡，辅助泊车撞墙了！",
    "阿达西平时开什么车？那当然是内燃机了，奥迪c51.8t手动挡#马牌xc7#马牌轮胎#奥迪c5",
    "67万全款买奔驰 半年还没提到车（一）按合同延期1天赔1万 已超150天，车主：车商反要求我们违约赔偿100万#奔驰",
    "多家企业捐款驰援灾区#西藏日喀则地震",
    "小米su7含金量还在不断加加加！！！ #娱乐评论大赏[话题]#   #万万没想到[话题]#   #小米su7[话题]#",
    "爱困不住我 我像风一样 自由到底 #捷途旅行者g331旅拍 #捷途旅行者",
    "青岛保时捷女销冠两年卖出340台保时捷，多次火上热搜备受非议，甚至被造黄谣。本人回应：做销冠并不开心 ，很累。没时间看负面新闻，清者自清。#保时捷 #女销售 #青岛 #女销冠 @抖音小助手"
]:
    M = {0:'负向', 1: '中性',  2:'正向' }
    inputs = tokenizer(
        txt,
        padding="max_length",    # 填充到最大长度
        truncation=True,         # 截断超长文本
        max_length=128,          # 与模型输入尺寸一致
        return_tensors="np"
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    outputs = rknn.inference(inputs=[input_ids, attention_mask])
    # softmax
    exp_values = np.exp(outputs[0] - np.max(outputs[0], axis=1, keepdims=True))
    softmax_array = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    idx = np.argmax(softmax_array)
    print(idx, M[idx], outputs[0], softmax_array)

