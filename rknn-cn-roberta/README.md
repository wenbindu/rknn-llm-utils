
# Sentiment-Analyse
> base: hfl/chinese-roberta-wwm-ext


## Structure
/serv
├── app
│   ├── __init__.py
│   ├── main.py          # Main Program
│   ├── exceptions.py    # Custom Exception
│   ├── schemas.py       # Pydantic Model
│   └── config.py        # Settings
├── requirements.txt
├── sentiment-api.service     # systemd service
└── logs/                # 日志目录
/tool
├── samples.csv  # data
├── train-roberta.py    # Trainer
├── torch2onnx.py       # torch model => onnx model
├── onnx2rknn.py        # onnx model => rknn model
├── infer-simulator.py  # Inference: simulator
└── infer-rk3588.py     # Inference: rk3588 

## APP

```
pip install transformers
```


## Tool

1. Train
2. TORCH2ONNX
3. ONNX2RKNN
4. Inference
   1. rk3588
   2. simulator
