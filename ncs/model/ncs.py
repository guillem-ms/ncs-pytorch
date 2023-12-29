import os
from utils.tensor import compute_nth_derivative
import torch
import torch.nn as nn
from loss.losses import *
from loss.metrics import *
from model.body import Body

from model.cloth import Garment
from .layers import *
from global_vars import BODY_DIR
from ncs.utils.augmentations import motion_augmentation

class NCS(nn.Module):
    def __init__(self, config):
        super(NCS, self).__init__()
        self.config = config
        folder = os.path.join(BODY_DIR, config.body.model)
        body_model = os.path.join(folder, "body.npz")
        garment_obj = os.path.join(folder, config.garment.name)
        
        # Read body
        print("Reading body model...")
        self.body = Body(body_model, input_joints=config.body.input_joints)  # Body class needs to be defined or imported
        
        # Read garment
        print("Reading garment...")
        self.garment = Garment(garment_obj)  # Garment class needs to be defined or imported
        
        print("Computing cloth blend weights...")
        self.garment.transfer_blend_weights(self.body)
        
        print("Smoothing cloth blend weights...")
        self.garment.smooth_blend_weights(iterations=config.garment.blend_weights_smoothing_iterations)

        # Build model (if any layers or operations need to be defined)
        self.build_model() 
        # Losses/Metrics
        self.build_losses_and_metrics()

    def build_model(self):
        self.build_lbs()
        self.build_encoder()
        self.build_decoder()

    def build_lbs(self):
        self.rot = Rotation(name="Rotation")
        self.skeleton = Skeleton(self.body.joints, name="Skeleton")
        self.lbs_body = LBS(self.body.blend_weights, trainable=False, name="LBS/Body")
        self.lbs_cloth = LBS(
            self.garment.blend_weights,
            trainable=self.config.blend_weights_trainable,
            name="LBS/Cloth",
        )

    def build_encoder(self):
        self.static_encoder = [
            SkelFlatten(),
            FullyConnected(64, act=F.relu, name="stc_enc/fc0"),
            FullyConnected(128, act=F.relu, name="stc_enc/fc0"),
            FullyConnected(256, act=F.relu, name="stc_enc/fc0"),
            FullyConnected(512, act=F.relu, name="stc_enc/fc0"),
        ]
        self.dynamic_encoder = [
            FullyConnected(32, act=F.relu, use_bias=False, name="dyn_enc/fc0"),
            FullyConnected(32, act=F.relu, use_bias=False, name="dyn_enc/fc1"),
            SkelFlatten(),
            FullyConnected(512, act=F.relu, use_bias=False, name="dyn_enc/fc2"),
            FullyConnected(512, act=F.relu, use_bias=False, name="dyn_enc/fc3"),
            nn.GRU(512, 512, batch_first=True, bias=False)

        ]

    def build_decoder(self):
        self.decoder = [
            FullyConnected(512, act=F.relu, name="dec/fc0"),
            FullyConnected(512, act=F.relu, name="dec/fc1"),
            FullyConnected(512, act=F.relu, name="dec/fc2"),
            PSD(self.garment.num_verts, name="dec/PSD"),
        ]

    def build_losses_and_metrics(self):
        # Losses and Metrics
        self.loss_metric = MyMetric(name="Loss")
        # Cloth model
        if self.config.cloth_model == "mass-spring":
            self.cloth_loss = EdgeLoss(self.garment)
            self.edge_metric = MyMetric(name="Edge")
        elif self.config.cloth_model == "baraff98":
            self.cloth_loss = ClothLoss(self.garment)
            self.stretch_metric = MyMetric(name="Stretch")
            self.shear_metric = MyMetric(name="Shear")
        elif self.config.cloth_model == "stvk":
            self.cloth_loss = StVKLoss(
                self.garment,
                self.config.loss.cloth.lambda_,
                self.config.loss.cloth.mu,
            )
            self.strain_metric = MyMetric(name="Strain")
        # Bending
        self.bending_loss = BendingLoss(self.garment)
        self.bending_metric = MyMetric(name="Bending")
        # Collision
        self.collision = Collision(self.body, name="Collision")
        self.collision_loss = CollisionLoss(
            self.body, collision_threshold=self.config.loss.collision_threshold
        )
        self.collision_metric = MyMetric(name="Collision")
        # Gravity
        self.gravity_loss = GravityLoss(
            self.garment.vertex_area,
            density=self.config.loss.density,
            gravity=self.config.gravity,
        )
        self.gravity_metric = MyMetric(name="Gravity")
        # Intertia
        self.inertia_loss = InertiaLoss(
            self.config.time_step,
            self.garment.vertex_area,
            density=self.config.loss.density,
        )
        self.inertia_metric = MyMetric(name="Inertia")
        # Pinning (if)
        if self.garment.pinning:
            self.pinning_loss = PinningLoss(self.garment, self.config.pin_blend_weights)

    def compute_losses_and_metrics(self, body, vertices, unskinned, training):
        loss = self.compute_static_losses_and_metrics(body, vertices[:, -1], unskinned)
        if training and self.config.motion_augmentation:
            vertices = vertices[self.config.motion_augmentation :]
        loss += self.compute_dynamic_losses_and_metrics(vertices)
        return loss

    def compute_static_losses_and_metrics(self, body, vertices, unskinned):
        # Cloth
        if self.config.cloth_model == "mass-spring":
            cloth_loss, edge_error = self.cloth_loss(vertices)
            cloth_loss *= self.config.loss.cloth.edge
        elif self.config.cloth_model == "baraff98":
            stretch_loss, stretch_error, shear_loss, shear_error = self.cloth_loss(
                vertices
            )
            cloth_loss = (
                self.config.loss.cloth.stretch * stretch_loss
                + self.config.loss.cloth.shear * shear_loss
            )
        elif self.config.cloth_model == "stvk":
            cloth_loss, strain_error = self.cloth_loss(vertices)
        # Bending
        bending_loss, bending_error = self.bending_loss(vertices)
        # Collision
        collision_indices = self.collision(vertices, body)
        collision_loss, collision_error = self.collision_loss(
            vertices, body, collision_indices
        )
        # Gravity
        gravitational_potential = self.gravity_loss(vertices)
        # Pinning
        if self.garment.pinning:
            pinning_loss = self.pinning_loss(unskinned, self.cloth_blend_weights)
        # Combine loss
        loss = (
            cloth_loss
            + self.config.loss.bending * bending_loss
            + self.config.loss.collision_weight * collision_loss
            + gravitational_potential
        )
        if self.garment.pinning:
            loss += self.config.loss.pinning * pinning_loss
        # Update metrics
        self.loss_metric.update_state(loss)
        if self.config.cloth_model == "mass-spring":
            self.edge_metric.update_state(edge_error)
        elif self.config.cloth_model == "baraff98":
            self.stretch_metric.update_state(stretch_error)
            self.shear_metric.update_state(shear_error)
        elif self.config.cloth_model == "stvk":
            self.strain_metric.update_state(strain_error)
        self.bending_metric.update_state(bending_error)
        self.collision_metric.update_state(collision_error)
        self.gravity_metric.update_state(gravitational_potential)
        return loss

    def compute_dynamic_losses_and_metrics(self, vertices):
        inertia_loss, inertia_error = self.inertia_loss(vertices)
        self.inertia_metric.update_state(inertia_error)
        return inertia_loss

    def train_step(self, inputs):
        self.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Zero out gradients to avoid accumulation
        body, vertices, unskinned = self(inputs)  # Forward pass
        loss = self.compute_losses_and_metrics(body, vertices, unskinned)  # Compute loss
        loss.backward()  # Backward pass
        self.optimizer.step()  # Update parameters
        metrics_dict = {m.name: m.result() for m in self.metrics}
        # Reset metrics at the end of each training step
        for m in self.metrics:
            m.reset()
        return metrics_dict

    def test_step(self, inputs):
        self.eval()
        with torch.no_grad():
            body, vertices, unskinned = self(inputs)  # Forward pass
            self.compute_losses_and_metrics(body, vertices, unskinned)
        metrics_dict = {m.name: m.result() for m in self.metrics}
        # Reset metrics at the end of each test step
        for m in self.metrics:
            m.reset()
        return metrics_dict

    @property
    def metrics(self):
        if self.config.cloth_model == "mass-spring":
            cloth_metrics = [self.edge_metric]
        elif self.config.cloth_model == "baraff98":
            cloth_metrics = [self.stretch_metric, self.shear_metric]
        elif self.config.cloth_model == "stvk":
            cloth_metrics = [self.strain_metric]
        return [
            self.loss_metric,
            *cloth_metrics,
            self.bending_metric,
            self.collision_metric,
            self.gravity_metric,
            self.inertia_metric,
        ]

    @property
    def cloth_blend_weights(self):
        return self.lbs_cloth.blend_weights

    def predict(self, inputs, w):
        poses, trans = inputs
        assert poses.dim() == 3, "Pose sequence has wrong dimensions. Should be (T, J, 3/4)."
        poses, trans = poses.unsqueeze(0), trans.unsqueeze(0)  # Add a new dimension
        X, matrices = self.call_inputs(poses, trans)
        deformations = self.call_network(X, w=w, training=False, predict=True)
        body = self.lbs_body(self.body.vertices, matrices)  # Compute body LBS
        # Compute garment LBS
        unskinned = self.garment.vertices + deformations
        matrices = torch.index_select(matrices, -3, self.body.input_joints)
        garment = self.lbs_cloth(unskinned, matrices)
        return body[0], garment[0], unskinned[0]

    def forward(self, inputs, w=None):
        poses, trans = inputs
        X, matrices = self.call_inputs(poses, trans)
        deformations = self.call_network(X, w=w, training=self.training)
        body = self.lbs_body(self.body.vertices, matrices[:, -1])  # Compute body LBS
        # Compute garment LBS
        unskinned = self.garment.vertices + deformations
        matrices = torch.index_select(matrices, -3, self.body.input_joints)
        garment = self.lbs_cloth(unskinned, matrices[:, -3:])
        return body, garment, unskinned[:, -1]

    def call_inputs(self, poses, trans):
        rotations = self.rot(poses)  # Compute local rotation matrices
        # Compute global transformation matrices
        matrices = self.body.forward_kinematics(rotations, trans)
        matrices_inv = torch.transpose(matrices[..., :3], -2, -1)
        X = torch.reshape(rotations[..., :2], (*rotations.shape[:-2], 6))  # 6D descriptors
        # Unposed gravity
        gravity_normalized = self.gravity_loss.gravity / torch.norm(self.gravity_loss.gravity)
        Z = torch.matmul(matrices_inv, gravity_normalized[:, None])[..., 0]
        X = torch.cat((X, Z), dim=-1)
        J = self.skeleton(matrices)  # Compute joint locations
        # Compute joint temporal derivatives
        dX = compute_nth_derivative(X, 1, self.config.time_step)[:, 1:]
        dJ = compute_nth_derivative(J, 2, self.config.time_step)
        dJ = torch.matmul(matrices_inv[:, 2:], dJ[..., None])[..., 0]  # Unpose accelerations
        X = torch.cat((X[:, 2:], dX, dJ), dim=-1)
        X = torch.index_select(X, -2, self.body.input_joints)  # Gather relevant joints only
        return X, matrices[:, 2:]

    def call_network(self, x, w, training, predict=False):
        x_static = x[..., :9]   # Extracting the first 9 elements from the last dimension
        x_dynamic = x[..., 9:21]  # Extracting the next 12 elements from the last dimension
        # Static encoder
        if not predict:
            x_static = x_static[:, -3:]
        for l in self.static_encoder:
            x_static = l(x_static)
        # Dynamic encoder
        for l in self.dynamic_encoder:
            x_dynamic = l(x_dynamic)
        if not predict:
            x_dynamic = x_dynamic[:, -3:]
        if training and self.config.motion_augmentation:
            x_static =  motion_augmentation(x_static, self.config.motion_augmentation)
            x_dynamic = motion_augmentation(x_dynamic, self.config.motion_augmentation, shuffle=True)
        if w is not None:
            x_dynamic = w * x_dynamic
        x = x_static + x_dynamic
        for l in self.decoder:
            x = l(x)
        return x