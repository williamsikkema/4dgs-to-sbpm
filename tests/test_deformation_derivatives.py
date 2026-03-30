import torch

from face_mc_gs.dynamic_gs.deformation import DeformationNetwork
from face_mc_gs.dynamic_gs.gaussian_model import GaussianModel


def test_time_gradient_nonzero():
    torch.manual_seed(0)
    g = GaussianModel(16, sh_degree=0)
    d = DeformationNetwork(hidden_dim=64, num_layers=3, posenc_L=4)
    t = torch.tensor(0.0, requires_grad=True)
    xyz = g.get_xyz()
    N = xyz.shape[0]
    dxyz, _, _ = d(xyz, t.expand(N, 1))
    x = xyz + dxyz
    (gi,) = torch.autograd.grad(x[0, 0], t, retain_graph=True)
    assert gi.abs().item() >= 0.0
