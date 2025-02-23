import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer

# Load the tokenizer and model from pretrained model.
tokenizer = AutoTokenizer.from_pretrained('./cn_roberta_model_token')
# Inference
ort_session = ort.InferenceSession('./cn_roberta_v1.onnx')

for txt in ['It is a nice day!']:
    M = {0: '负向', 1: '中性', 2: '正向'}
    inputs = tokenizer(txt, return_tensors='np')
    outputs = ort_session.run(
        None,
        {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
    )
    # softmax
    exp_values = np.exp(
        outputs[0] - np.max(outputs[0], axis=1, keepdims=True)
    )  # Or skip the softmax function.
    softmax_array = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    idx = np.argmax(softmax_array)
    # output the result.
    print(idx, M[idx], outputs[0], softmax_array)

# draw the graph of the onnx model
model = onnx.load('./cn_roberta_v1.onnx')
for input in model.graph.input:
    print(f'Input: {input.name}, Shape: {input.type.tensor_type.shape}')
for output in model.graph.output:
    print(f'Output: {output.name}, Shape: {output.type.tensor_type.shape}')
