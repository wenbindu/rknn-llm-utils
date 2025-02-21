# rknn-llm-utils

## Solutions
1. rknn-debert-v3: `From microsoft/deberta-v3-base`

    

## Prepare Environments



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
