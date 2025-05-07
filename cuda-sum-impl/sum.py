import torch

data = torch.randn(16,128,64,128).cuda()

data2 = torch.sum(data, 1, keepdim=True)
# data2 = torch.sum(data, 1)

print(data2.shape)
print(f"finished run test_relu.py")
