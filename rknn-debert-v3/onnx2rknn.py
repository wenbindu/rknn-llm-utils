from rknn.api import RKNN

# --------------------------------------------------
# 不进行量化.
# --------------------------------------------------
# 初始化RKNN对象
rknn = RKNN(verbose=True)
ONNX_MODEL = './deberta.onnx'
# 配置转换参数（关键！）
rknn.config(
     target_platform='rk3588',   # 根据芯片型号修改
     optimization_level=3,       # 最高优化级别
     quantized_dtype='w8a8',  # 量化类型
     quantized_algorithm='normal', 
     quantized_method='channel',
 )

# 加载ONNX模型（指定输入节点顺序）
rknn.load_onnx(
     model=ONNX_MODEL,
     inputs=['input_ids', 'attention_mask'],
     input_size_list=[[1, 128], [1, 128]],  # 固定输入尺寸
     outputs=['logits']
 )

ret = rknn.build(
     do_quantization=False, 
     rknn_batch_size=1
 )
assert ret == 0, "Build RKNN failed"

# 导出RKNN模型
ret = rknn.export_rknn('./deberta_no_quant.rknn')
assert ret == 0, "Export RKNN failed"

rknn.release()
