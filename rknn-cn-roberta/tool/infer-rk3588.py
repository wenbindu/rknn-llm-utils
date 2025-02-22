from rknnlite.api import RKNNLite
from transformers import AutoTokenizer
import numpy as np
import logging
import time 

logger = logging.getLogger(__name__)

rknn_model = 'cn_roberta_v1.rknn'
tokenizer = AutoTokenizer.from_pretrained("./")

rknn_lite = RKNNLite()
# load RKNN model
logger.info('--> Load RKNN model')
ret = rknn_lite.load_rknn(rknn_model)
if ret != 0:
    logger.info('Load RKNN model failed')
    exit(ret)
logger.info('done')
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
if ret != 0:
    logger.info('Init runtime environment failed')
    exit(ret)
logger.info('done')

ret=rknn_lite.list_support_target_platform(rknn_model=rknn_model)
logger.info('supported platform:{}'.format(ret))

st = time.time()
for txt in [
    "交警队出盖章文件了！hw智驾连环车祸+行车记录仪内存卡被偷最新动态！",
    "坦克500 Hi4-Z、Hi4-T到底啥区别？应该怎么选？一条视频就明白了！#新车抢先看",
]:
    M = {0:'负向', 1: '中性',  2:'正向' }
    inputs = tokenizer(txt, padding="max_length", truncation=True, max_length=128, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    outputs = rknn_lite.inference(inputs=[input_ids, attention_mask])
    # softmax
    exp_values = np.exp(outputs[0] - np.max(outputs[0], axis=1, keepdims=True))
    softmax_array = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    idx = np.argmax(softmax_array)
    print(idx, M[idx], outputs[0], softmax_array)

print(f"spend seconds: {time.time() - st}")

rknn_lite.release()