import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from utils.rotation import from_axis_angle, from_quaternion
from utils.mesh import lbs
from utils.tensor import tf_shape

class FullyConnected(nn.Module):
    def __init__(self, units, act=None, use_bias=True):
        super(FullyConnected, self).__init__()
        self.units = units
        self.act = act if act is not None else lambda x: x
        self.use_bias = use_bias
        self.kernel = nn.Parameter(torch.Tensor(units, units))
        self.bias = nn.Parameter(torch.Tensor(units), requires_grad=use_bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.kernel)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x = torch.matmul(x.unsqueeze(-2), self.kernel)
        x = x[..., 0, :]
        if self.use_bias:
            x += self.bias
        return self.act(x)


class SkelFlatten(nn.Module):
    def __init__(self):
        super(SkelFlatten, self).__init__()

    def forward(self, inputs):
        input_shape = inputs.size()
        return inputs.view(*input_shape[:-2], -1)


class PSD(nn.Module):
    def __init__(self, num_verts, num_dims=3, act=None):
        super(PSD, self).__init__()
        self.num_verts = num_verts
        self.num_dims = num_dims
        self.act = act if act is not None else lambda x: x
        self.psd = nn.Parameter(torch.Tensor(num_dims, num_verts, num_dims))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.psd)

    def forward(self, x):
        x = torch.tensordot(x, self.psd, dims=([[-1], [0]]))
        x = self.act(x)
        return x

class Skeleton(nn.Module):
    def __init__(self, rest_joints):
        super(Skeleton, self).__init__()
        self.rest_joints = rest_joints

    def forward(self, matrices):
        return lbs(self.rest_joints, matrices)


class LBS(nn.Module):
    def __init__(self, blend_weights, trainable):
        super(LBS, self).__init__()
        self.trainable = trainable
        if trainable:
            blend_weights = torch.log(blend_weights + 0.001)
        self.blend_weights = nn.Parameter(blend_weights, requires_grad=trainable)

    def forward(self, vertices, matrices):
        if self.trainable:
            blend_weights = F.softmax(self.blend_weights, dim=-1)
            return lbs(vertices, matrices, blend_weights)
        return lbs(vertices, matrices, self.blend_weights)


class Rotation(nn.Module):
    def __init__(self):
        super(Rotation, self).__init__()
        self.mode = None

    def forward(self, orientations):
        if self.mode is None:
            if orientations.size(-1) == 3:
                self.mode = "axis_angle"
            elif orientations.size(-1) == 4:
                self.mode = "quaternion"

        if self.mode == "axis_angle":
            return from_axis_angle(orientations)
        elif self.mode == "quaternion":
            return from_quaternion(orientations)


class Collision(nn.Module):
    def __init__(self, body):
        super(Collision, self).__init__()
        self.collision_vertices = torch.tensor(body.collision_vertices)
        self.run_sample = lambda elem: cKDTree(elem[1]).query(elem[0], workers=-1)[1]

    def run(self, vertices, collider):
        return torch.stack([torch.tensor(self.run_sample(elem)) for elem in zip(vertices, collider)])

    def forward(self, vertices, collider):
        batch_size = vertices.size(0)
        idx = self.run(vertices, collider.index_select(-2, self.collision_vertices))
        # Creating a tensor of indices in the required shape
        batch_indices = torch.arange(batch_size).unsqueeze(-1).repeat(1, idx.size(1))
        idx = torch.stack([batch_indices, idx], dim=-1)
        return idx
