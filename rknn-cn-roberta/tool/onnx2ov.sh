#!/bin/bash

# fixed shape
mo \
--input_model cn_roberta_v1.onnx \
--output_dir openvino_model/ \
--input input_ids,attention_mask \
--input_shape "[1,128],[1,128]" \
--output logits

# or dynamic shape
mo \
--input_model cn_roberta_v1.onnx \
--output_dir openvino_model_dyn/ \
--input "input_ids[1,?],attention_mask[1,?]"
