import io
import time

import numpy as np
import torch


def state_dict_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 2**20


def summarize(times_ms):
    t = np.asarray(times_ms)
    return {"median_ms": float(np.median(t)), "p99_ms": float(np.percentile(t, 99)),
            "mean_ms": float(t.mean())}


@torch.no_grad()
def torch_latency(model, size, dtype=torch.float32, warmup=30, runs=300, threads=1):
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    model.eval()
    x = torch.randn(1, 3, size, size, dtype=dtype)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)
    torch.set_num_threads(prev)
    return summarize(times)


def ort_latency(onnx_path, size, warmup=30, runs=300, threads=1):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1e3)
    return summarize(times)
