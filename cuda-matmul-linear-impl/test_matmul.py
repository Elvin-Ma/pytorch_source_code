import torch


def  test_matmul(profile = False):
    data1 = torch.randn(4, 10, 128, 32).cuda().bfloat16()
    data2 = torch.randn(32, 128).cuda().bfloat16()
    # linear = torch.nn.Linear(32, 128).cuda().bfloat16()

    # data3 = linear(data1)
    data3 = torch.matmul(data1, data2)

    if profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./mm_log"),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:

          for i in range(10):
            # data3 = linear(data1)
            data3 = torch.matmul(data1, data2)
            prof.step()

if __name__ == "__main__":
    test_matmul(profile = False)
    print(f"run test_matmul.py successfully !!!")
