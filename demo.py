import os
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cpu", action="store_true", default=True, help="Use CPU (default)"
)
parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU")
parser.add_argument("--ipex", action="store_true", default=False, help="Use IPEX")
parser.add_argument(
    "--dtype", choices=["fp32", "fp16", "bf16"], default="fp32", help="Data type"
)
parser.add_argument(
    "--iters", type=int, default=100, help="Number of iterations (default is 100)"
)

args = parser.parse_args()

gpu_enabled = args.gpu

cpu_enabled = False if gpu_enabled else args.cpu
ipex_enabled = args.ipex

data_type = args.dtype
num_iters = args.iters

print(">" * 50)
print("Use CPU:", cpu_enabled)
print("Use GPU:", gpu_enabled)
print("Use IPEX:", ipex_enabled)
print("Data type:", data_type)
print("Number of iterations:", num_iters)
print("<" * 50)

if cpu_enabled:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from time import perf_counter
import numpy as np
import torch


def measure_latency(model, *args, **kwargs):
    # prepare date
    latencies = []
    # warm up
    for _ in range(3):
        _ = model(*args, **kwargs)
    # Timed run
    for _ in range(num_iters):
        start_time = perf_counter()
        _out = model(*args, **kwargs)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies, 95)
    print(">" * 30)
    print(
        f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};",
        time_p95_ms,
    )
    print("<" * 30)


test_img = "cat.jpg"
test_str1 = "猫"
test_str2 = "人类"

model = Model.from_pretrained(
    "damo/multi-modal_team-vit-large-patch14_multi-modal-similarity",
    device="cpu" if cpu_enabled else "gpu",
)
model.eval()

if data_type == "fp32":
    model.float()
elif data_type == "fp16":
    model.half()

multi_modal_similarity_pipeline = pipeline(
    task=Tasks.multi_modal_similarity, model=model
)

test_input1 = {"img": test_img, "text": test_str1}

# output1 = multi_modal_similarity_pipeline(test_input1)

# print(">" * 50)

# print("model struct: \n", multi_modal_similarity_pipeline.model)

# print("<" * 50)

if cpu_enabled and ipex_enabled:
    import intel_extension_for_pytorch as ipex

    if data_type == "bf16":
        multi_modal_similarity_pipeline.model = ipex.optimize(
            multi_modal_similarity_pipeline.model, dtype=torch.bfloat16
        )
    else:
        multi_modal_similarity_pipeline.model = ipex.optimize(
            multi_modal_similarity_pipeline.model
        )

# print(">" * 50)

# print("ipex.optimize model struct: \n", multi_modal_similarity_pipeline.model)

# print("<" * 50)

if cpu_enabled:
    if data_type == "bf16":
        with torch.no_grad(), torch.cpu.amp.autocast():
            measure_latency(multi_modal_similarity_pipeline, test_input1)
    else:
        with torch.no_grad():
            measure_latency(multi_modal_similarity_pipeline, test_input1)
elif gpu_enabled:
    if data_type == "fp16":
        with torch.no_grad(), torch.cuda.amp.autocast():
            measure_latency(multi_modal_similarity_pipeline, test_input1)
    else:
        with torch.no_grad():
            measure_latency(multi_modal_similarity_pipeline, test_input1)
