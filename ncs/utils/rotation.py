import numpy as np
import torch

from utils.tensor import tf_shape

def from_axis_angle(axis_angle):
    input_shape = axis_angle.shape
    assert (
        input_shape[-1] == 3
    ), "Rotation from axis angle error. Wrong tensor size: " + str(input_shape)
    ndims = len(axis_angle.shape)

    # Flatten batch dimensions
    axis_angle = axis_angle.reshape(-1, 3)
    # Decompose into axis and angle
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + torch.finfo(axis_angle.dtype).eps)

    # Compute Rodrigues' rotation formula
    batch_size = axis_angle.shape[0]
    zeros = torch.zeros((batch_size,), dtype=torch.float32)
    M = torch.stack(
        [
            zeros,
            -axis[:, 2],
            axis[:, 1],
            axis[:, 2],
            zeros,
            -axis[:, 0],
            -axis[:, 1],
            axis[:, 0],
            zeros,
        ],
        dim=1,
    )
    M = M.reshape(-1, 3, 3)

    rotations = (
        torch.eye(3).expand(batch_size, -1, -1)
        + torch.sin(angle)[:, None] * M
        + (1 - torch.cos(angle)[:, None]) * torch.matmul(M, M)
    )
    # Reshape back to original batch shape
    rotations = rotations.reshape(*input_shape[: ndims - 1], 3, 3)
    return rotations


def from_quaternion(quaternions):
    input_shape = quaternions.shape
    assert (
        input_shape[-1] == 4
    ), "Rotation from quaternion error. Wrong tensor size: " + str(input_shape)
    ndims = len(quaternions.shape)

    # Flatten batch dimensions
    quaternions = quaternions.reshape(-1, 4)

    # Compute rotations
    w, x, y, z = torch.unbind(quaternions, dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x



def lerp(q0, q1, r):
    r = np.expand_dims(r, [-2, -1])
    return (1 - r) * q0 + r * q1


def slerp(q0, q1, r):
    r = np.expand_dims(r, axis=[-2, -1])
    dot = (q0 * q1).sum(-1, keepdims=True)
    dot = np.clip(dot, -1, 1)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    w0 = np.where(sin_omega, np.sin((1 - r) * omega) / sin_omega, 1 - r)
    w1 = np.where(sin_omega, np.sin(r * omega) / sin_omega, r)
    return w0 * q0 + w1 * q1


def axis_angle_to_quat(rotvec):
    angle = np.linalg.norm(rotvec, axis=-1)[..., None] + np.finfo(float).eps
    axis = rotvec / angle
    sin = np.sin(angle / 2)
    w = np.cos(angle / 2)
    return np.concatenate((w, sin * axis), axis=-1)


def quat_to_axis_angle(quat):
    angle = 2 * np.arccos(quat[..., 0:1])
    axis = quat[..., 1:] * (1 / (np.sin(angle / 2) + np.finfo(float).eps))
    return angle * axis
