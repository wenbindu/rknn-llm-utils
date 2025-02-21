## Abstract
本项目主要基于 microsoft/deberta-v3-base 预训练模型，进行情感分析训练，模型导出，以及转换，以便部署到RK3588.
由于涉及的模型文件太大，所以此处只涉及代码，不包括模型文件，模型文件可以去 [Modelscope](https://huggingface.co/dean2023/deberta-v3-base-rknn/tree/main) 下载。


## Scheduler
graph TD
    A[PyTorch模型] -->|torch.onnx.export| B(ONNX模型)
    B -->|rknn.load_onnx| C{RKNN转换}
    C -->|成功| D[RKNN模型]
    C -->|失败| E[分析错误日志]
    E --> F{错误类型}
    F -->|不支持的算子| G[修改模型结构/添加自定义算子]
    F -->|量化错误| H[优化校准数据集]
    D --> I[设备端部署验证]
    I -->|性能不足| J[启用混合量化/层融合]

