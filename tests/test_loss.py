from cov3d.loss import Cov3dLoss
import torch

negative = torch.as_tensor([[0,0]])
positive = torch.as_tensor([[1,0]])
mild = torch.as_tensor([[1,1]])
moderate = torch.as_tensor([[1,2]])
severe = torch.as_tensor([[1,3]])
critical = torch.as_tensor([[1,4]])

def test_loss_every_negative():
    loss_func = Cov3dLoss(severity_everything=True, severity_smoothing=0.0, presence_smoothing=0.0)
    loss = loss_func(
        torch.as_tensor([[-100.0,-20.0,-20.0,-20.0,-20.0,100.0]]),
        negative,
    )
    assert loss < 0.001
    
def test_loss_every_mild():
    loss_func = Cov3dLoss(severity_everything=True, severity_smoothing=0.0, presence_smoothing=0.0)
    loss = loss_func(
        torch.as_tensor([[100.0,100.0,-20.0,-20.0,-20.0,-20.0]]),
        mild,
    )
    assert loss < 0.001
    
    
def test_loss_every_moderate():
    loss_func = Cov3dLoss(severity_everything=True, severity_smoothing=0.0, presence_smoothing=0.0)
    loss = loss_func(
        torch.as_tensor([[100.0,-20.0,100.0,-20.0,-20.0,-20.0]]),
        moderate,
    )
    assert loss < 0.001
    
def test_loss_every_severe():
    loss_func = Cov3dLoss(severity_everything=True, severity_smoothing=0.0, presence_smoothing=0.0)
    loss = loss_func(
        torch.as_tensor([[100.0,-20.0,-20.0,100.0,-20.0,-20.0]]),
        severe,
    )
    assert loss < 0.001
    
def test_loss_every_critical():
    loss_func = Cov3dLoss(severity_everything=True, severity_smoothing=0.0, presence_smoothing=0.0)
    loss = loss_func(
        torch.as_tensor([[100.0,-20.0,-20.0,-20.0,100.0,-20.0]]),
        critical,
    )
    assert loss < 0.001

def test_loss_every_positive():
    loss_func = Cov3dLoss(severity_everything=True, severity_smoothing=0.0, presence_smoothing=0.0)
    loss = loss_func(
        torch.as_tensor([[100.0,100.0,100.0,100.0,100.0,-100.0]]),
        positive,
    )
    assert loss < 0.000001
        