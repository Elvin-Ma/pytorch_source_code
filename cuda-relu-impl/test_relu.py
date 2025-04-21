import torch

data = torch.randn(2,3,4,5).cuda()

data2 = torch.relu(data)

print(data2)
print(f"finished run test_relu.py")