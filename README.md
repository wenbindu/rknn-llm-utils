# rknn-llm-utils

## Solutions
1. rknn-debert-v3: `From microsoft/deberta-v3-base`
2. rknn-cn-roberta: `From hfl/chinese-roberta-wwm-ext`
    

## Prepare Environments

```sh
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download hfl/chinese-roberta-wwm-ext --local-dir hfl/chinese-roberta-wwm-ext --local-dir-use-symlinks False
```

## rknn-toolkit2
> v2.3.0

### Build from Dockerfile
Reference: [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
```sh
docker build -t rknn-transformer-v23 .  
```
### Dev
```sh
docker run --rm --gpus all --rm -v /hw_data/dean-ws/emotion-bert:/hw_data/dean-ws/emotion-bert --ipc=host  -it rknn-transformer-v23  bash
# install transformers when use bert
pip install transformers
pip install sentencepiece

python
```
