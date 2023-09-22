import torch
import torch.nn as nn
import torch.nn.functional as F
def hinge(y_real,y_fake,critic):
    if critic:
        return  torch.mean(F.relu(1. - y_real))+torch.mean(F.relu(1 + y_fake))
    else:
        return -torch.mean(y_fake)

def least_squares(y_real, y_fake,critic):
    if critic:
        return (F.mse_loss(y_real,torch.ones_like(y_real))+F.mse_loss(y_fake,torch.zeros_like(y_fake)))
    else:
        return 0.5*(F.mse_loss(y_fake,torch.ones_like(y_fake)))

def wasserstein(y_real, y_fake,critic):
    if critic:
        return (-y_real+y_fake).mean()
    else:
        return (-y_fake).mean()


def gradient_penalty(y_pred, averaged_samples):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=averaged_samples,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.flatten(start_dim=1)
    gradients_norm = gradients.norm(2, dim=1)
    return (torch.nn.functional.relu(gradients_norm - 1) ** 2).mean()

