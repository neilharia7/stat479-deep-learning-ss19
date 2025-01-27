import torch

tensor_1 = torch.tensor([1.0, 2.0, 3.0])

tensor_2 = torch.tensor([4.0, 5.0, 6.0])

product = torch.matmul(tensor_1, tensor_2)

if product > 0:
	print("The angle is between the 2 tensors is < 90")
elif product < 0:
	print("The angle is between the 2 tensors is > 90")
else:
	print("The angle is between the 2 tensors is = 90")
