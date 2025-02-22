import uvicorn
from typing import Union
from fastapi import FastAPI
from rknnlite.api import RKNNLite
from pydantic import BaseModel, Field
from typing import AsyncGenerator
import asyncio
from contextlib import asynccontextmanager
from typing import List
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from transformers import AutoTokenizer



# 数据模型
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, example="Hello World")

class ApiResponse(BaseModel):
    message: str
    data: str


class NPUCorePool:
    def __init__(self, core_ids: List[int]):
        self.available_cores = core_ids
        self.lock = threading.Lock()
        self.pool = {}
        
        # 初始化所有NPU核心
        for core_id in core_ids:
            rknn = RKNNLite()
            ret = rknn.load_rknn("/home/proembed/emotion/cn_roberta_v1.rknn")
            if ret != 0:
                raise RuntimeError(f"Core {core_id} 初始化失败")
            rknn.init_runtime(core_mask=core_id)
            tokenizer = AutoTokenizer.from_pretrained("/home/proembed/emotion/")

            self.pool[core_id] = {
                "rknn": rknn,
                "tokenizer": tokenizer,
                "in_use": False
            }
            print(f"NPU核心 {core_id} 初始化完成")

    def acquire_core(self):
        with self.lock:
            for core_id, status in self.pool.items():
                if not status["in_use"]:
                    status["in_use"] = True
                    return core_id, status["rknn"], status["tokenizer"]
            return None, None, None  # 无可用核心

    def release_core(self, core_id: int):
        with self.lock:
            self.pool[core_id]["in_use"] = False

    async def shutdown(self):
        """异步安全关闭所有核心"""
        release_tasks = []
        for core_id in self.pool:
            task = asyncio.create_task(self._release_core_async(core_id))
            release_tasks.append(task)
        await asyncio.gather(*release_tasks)

    async def _release_core_async(self, core_id: int):
        """异步释放单个核心"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.pool[core_id]["rknn"].release
            )
        except Exception as e:
            print(f"核心 {core_id} 释放异常: {str(e)}")


class InferenceWorker:
    def __init__(self, core_pool: NPUCorePool):
        self.core_pool = core_pool
        self.executor = ThreadPoolExecutor(max_workers=3)  # 每个核心一个线程
        self.M = {0:'负向', 1: '中性',  2:'正向' }

    async def predict(self, text: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._sync_predict,
            text
        )

    def _sync_predict(self, text: str):
        core_id, rknn, tokenizer = self.core_pool.acquire_core()
        if not rknn:
            raise RuntimeError("所有NPU核心忙")

        try:
            # 预处理
            inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
            # 推理
            outputs = rknn.inference(inputs=[input_ids, attention_mask])
            # 后处理
            exp_values = np.exp(outputs[0] - np.max(outputs[0], axis=1, keepdims=True))
            softmax_array = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            idx = np.argmax(softmax_array)
            return self.M[idx]
        finally:
            self.core_pool.release_core(core_id)

# 全局资源管理器
npu_pool: Union[NPUCorePool, None] = None
worker: Union[InferenceWorker , None] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """管理应用生命周期"""
    global npu_pool, worker
    
    # 初始化阶段
    try:
        print("🚀 正在初始化NPU资源...")
        npu_pool = NPUCorePool(core_ids=[
            RKNNLite.NPU_CORE_0, 
            RKNNLite.NPU_CORE_1,
            RKNNLite.NPU_CORE_2
        ])
        worker = InferenceWorker(npu_pool)
        
        # 预热所有核心（异步执行）
        async def warmup():
            tasks = [worker.predict("预热文本") for _ in range(3)]
            await asyncio.gather(*tasks)
        await warmup()

        yield  # 进入运行阶段
        
    finally:
        # 清理阶段
        print("\n🛑 正在释放NPU资源...")
        if npu_pool:
            await npu_pool.shutdown()
        print("✅ 资源释放完成")

app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: TextRequest):
    ret_str = await worker.predict(request.text)
    return ApiResponse(
        message="Success",
        data=ret_str
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )
