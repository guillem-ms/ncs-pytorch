import torch
from utils.mesh import vertex_normals, face_normals

# Mass-spring model
class EdgeLoss:
    def __init__(self, garment):
        self.edges = garment.edges
        self.edge_lengths_true = garment.edge_lengths

    def __call__(self, vertices):
        edges = torch.gather(vertices, 1, self.edges[:, 0].unsqueeze(1)) - torch.gather(
            vertices, 1, self.edges[:, 1].unsqueeze(1)
        )
        edge_lengths = torch.norm(edges, dim=-1)
        edge_difference = edge_lengths - self.edge_lengths_true
        loss = edge_difference**2
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        error = torch.abs(edge_difference)
        error = torch.mean(error)
        return loss, error


# Baraff '98 cloth model (squared)
class ClothLoss:
    def __init__(self, garment):
        self.faces = garment.faces
        self.face_areas = garment.face_areas
        self.total_area = garment.surf_area
        self.uv_matrices = garment.uv_matrices

    def __call__(self, vertices):
        dX = torch.stack(
            [
                torch.gather(vertices, 1, self.faces[:, 1].unsqueeze(1))
                - torch.gather(vertices, 1, self.faces[:, 0].unsqueeze(1)),
                torch.gather(vertices, 1, self.faces[:, 2].unsqueeze(1))
                - torch.gather(vertices, 1, self.faces[:, 0].unsqueeze(1)),
            ],
            dim=2,
        )

        w = torch.einsum("abcd,bce->abed", dX, self.uv_matrices)
        
        stretch = torch.norm(w, dim=-1) - 1
        stretch_loss = self.face_areas[:, None] * stretch**2
        stretch_loss = torch.sum(stretch_loss, dim=[1, 2])
        stretch_loss = torch.mean(stretch_loss)
        stretch_error = (
            self.face_areas[:, None] * torch.abs(stretch) * (0.5 / self.total_area)
        )
        stretch_error = torch.mean(torch.sum(stretch_error, dim=-1))

        shear = torch.sum(torch.mul(w[:, :, 0], w[:, :, 1]), dim=-1)
        shear_loss = shear**2
        shear_loss *= self.face_areas
        shear_loss = torch.sum(shear_loss, dim=1)
        shear_loss = torch.mean(shear_loss)
        shear_error = self.face_areas * torch.abs(shear) * (1 / self.total_area)
        shear_error = torch.mean(torch.sum(shear_error, dim=-1))

        return stretch_loss, stretch_error, shear_loss, shear_error


# Saint-Venant Kirchhoff
class StVKLoss:
    def __init__(self, garment, l, m):
        self.faces = garment.faces
        self.face_areas = garment.face_areas
        self.total_area = garment.surf_area
        self.uv_matrices = garment.uv_matrices
        self.l = l
        self.m = m

    def __call__(self, vertices):
        dX = torch.stack(
            [
                torch.gather(vertices, 1, self.faces[:, 1].unsqueeze(1))
                - torch.gather(vertices, 1, self.faces[:, 0].unsqueeze(1)),
                torch.gather(vertices, 1, self.faces[:, 2].unsqueeze(1))
                - torch.gather(vertices, 1, self.faces[:, 0].unsqueeze(1)),
            ],
            dim=-1,
        )
        F = dX @ self.uv_matrices
        Ft = torch.transpose(F, -2, -1)
        G = 0.5 * (torch.matmul(Ft, F) - torch.eye(2).expand_as(Ft))
        S = self.m * G + (0.5 * self.l * torch.einsum("...ii", G)).unsqueeze(-1).unsqueeze(-1) * torch.eye(2).expand_as(G)
        loss = torch.einsum("...ii", torch.matmul(torch.transpose(S, -2, -1), G))
        loss *= self.face_areas
        loss = torch.mean(loss, dim=0)
        loss = torch.sum(loss)
        error = loss / self.total_area

        return loss, error


class BendingLoss:
    def __init__(self, garment):
        self.faces = garment.faces
        self.face_adjacency = garment.face_adjacency
        face_areas = garment.face_areas[garment.face_adjacency].sum(-1)
        edge_lengths = garment.face_adjacency_edge_lengths
        self.stiffness_scaling = edge_lengths**2 / (8 * face_areas)
        self.angle_true = garment.face_dihedral

    def __call__(self, vertices):
        mesh_face_normals = face_normals(vertices, self.faces)
        normals0 = torch.gather(mesh_face_normals, 1, self.face_adjacency[:, 0].unsqueeze(1))
        normals1 = torch.gather(mesh_face_normals, 1, self.face_adjacency[:, 1].unsqueeze(1))
        cos = torch.einsum("abc,abc->ab", normals0, normals1)
        sin = torch.norm(torch.cross(normals0, normals1), dim=-1)
        angle = torch.atan2(sin, cos) - self.angle_true
        loss = angle**2
        error = torch.abs(angle)
        loss *= self.stiffness_scaling
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        error = torch.mean(error)

        return loss, error


# Fast estimation of SDF
class CollisionLoss:
    def __init__(self, body, collision_threshold=0.004):
        self.body_faces = body.faces
        self.collision_vertices = torch.tensor(body.collision_vertices)
        self.collision_threshold = collision_threshold

    def __call__(self, vertices, body_vertices, indices):
        # Compute body vertex normals
        body_vertex_normals = vertex_normals(body_vertices, self.body_faces)
        # Gather collision vertices
        body_vertices = torch.gather(body_vertices, 1, self.collision_vertices.unsqueeze(1))
        body_vertex_normals = torch.gather(body_vertex_normals, 1, self.collision_vertices.unsqueeze(1))

        # Compute loss
        cloth_to_body = vertices - body_vertices.gather(1, indices)
        body_vertex_normals = body_vertex_normals.gather(1, indices)
        normal_dist = torch.einsum("abc,abc->ab", cloth_to_body, body_vertex_normals)
        loss = torch.minimum(normal_dist - self.collision_threshold, torch.tensor(0.0)) ** 2
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        error = (normal_dist < 0.0).float()
        error = torch.mean(error)

        return loss, error


class GravityLoss:
    def __init__(self, vertex_area, density=0.15, gravity=[0, 0, -9.81]):
        self.vertex_mass = density * vertex_area[:, None]
        self.gravity = torch.tensor(gravity, dtype=torch.float32)


    def __call__(self, vertices):
        loss = -self.vertex_mass * vertices * self.gravity
        loss = torch.sum(loss, dim=[1, 2])
        loss = torch.mean(loss)
        return loss


class InertiaLoss:
    def __init__(self, dt, vertex_area, density=0.15):
        self.dt = dt
        self.vertex_mass = density * vertex_area
        self.total_mass = torch.sum(self.vertex_mass)

    def __call__(self, vertices):
        x0, x1, x2 = vertices.unbind(dim=1)
        x_proj = 2 * x1 - x0
        x_proj = x_proj.detach()
        dx = x2 - x_proj
        loss = (0.5 / self.dt**2) * self.vertex_mass[:, None] * dx**2
        loss = torch.mean(loss, dim=0)
        loss = torch.sum(loss)
        error = self.vertex_mass * torch.norm(dx, dim=-1)
        error = torch.sum(error, dim=-1) / self.total_mass
        error = torch.mean(error)
        return loss, error


class PinningLoss:
    def __init__(self, garment, pin_blend_weights=False):
        self.indices = garment.pinning_vertices
        self.vertices = garment.vertices[self.indices]
        self.pin_blend_weights = pin_blend_weights
        if pin_blend_weights:
            self.blend_weights = garment.blend_weights[self.indices]

    def __call__(self, unskinned, blend_weights):
        loss = torch.gather(unskinned, -2, self.indices.unsqueeze(-2)) - self.vertices
        loss = loss**2
        loss = torch.sum(loss, dim=[1, 2])
        loss = torch.mean(loss)
        if self.pin_blend_weights:
            _loss = torch.gather(blend_weights, -2, self.indices.unsqueeze(-2)) - self.blend_weights
            _loss = _loss**2
            _loss = torch.mean(_loss, 0)
            loss += 1e2 * torch.sum(_loss)
        return loss
