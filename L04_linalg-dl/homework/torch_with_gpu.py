import torch

# use mac gpu
device = torch.device('mps')

a = torch.tensor([1.0, 2.0, 3.0], device=device)  # loaded on gpu
b = torch.tensor([4.0, 5.0, 6.0])  # loaded on cpu

try:
	product = a @ b
	print(product)
except Exception as e:
	print(f"Error >> {e}")

"""
Returns
Expected all tensors to be on the same device. Found: cpu, mps:0
"""

b = b.to(device)

try:
	product = a @ b
	print(product)
except Exception as e:
	print(f"Error >> {e}")

""" 
Returns
tensor(32., device='mps:0')

"""
