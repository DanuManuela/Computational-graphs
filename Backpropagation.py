import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

# This is the parameter we want to optimize -> requires_grad=True
w = torch.tensor(1.0)

# forward pass to compute loss
y_predicted = w * x
loss = (y_predicted - y) ** 2
print('loss: ', loss.item())

# backward pass to compute gradient dLoss/dw
dloss_dw = 2 * x * (y_predicted - y)
print('grad: ', dloss_dw.item())

# update weights, this operation should not be part of the computational graph
w -= 0.01 * dloss_dw

# next forward and backward pass...
