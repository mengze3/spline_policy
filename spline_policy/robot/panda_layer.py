import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from torch import Tensor


@dataclass
class EEfTarget:
    position: Tensor
    rotation: Tensor


class PandaLayer(torch.nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        tcp_offset: int = 0.105,
    ):
        super().__init__()
        self.device = device
        self.tcp_offset = tcp_offset
        self.dtype = torch.float64

        self.theta_min = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        ).to(self.device)
        self.theta_max = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        ).to(self.device)
        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min - self.theta_mid) * 0.9 + self.theta_mid
        self.theta_max_soft = (self.theta_max - self.theta_mid) * 0.9 + self.theta_mid
        self.dof = len(self.theta_min)

        self.A0 = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.A1 = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.A2 = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.A3 = torch.tensor(0.0825, dtype=self.dtype, device=self.device)
        self.A4 = torch.tensor(-0.0825, dtype=self.dtype, device=self.device)
        self.A5 = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.A6 = torch.tensor(0.088, dtype=self.dtype, device=self.device)
        self.A7 = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    def get_transformations_each_link(self, pose, theta):
        batch_size = theta.shape[0]
        self.device = theta.device

        # set params
        A = torch.tensor(
            [self.A0, self.A1, self.A2, self.A3, self.A4, self.A5, self.A6, self.A7]
        ).to(
            dtype=self.dtype, device=self.device
        )  # (8,)
        alpha = torch.tensor(
            [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0],
        ).to(
            dtype=self.dtype, device=self.device
        )  # (8,)
        D = torch.tensor([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]).to(
            dtype=self.dtype, device=self.device
        )  # (8,)
        theta_joints = theta  # (batch_size, 7)
        theta_fixed = (
            -np.pi
            / 4
            * torch.ones(batch_size, 1).to(dtype=self.dtype, device=self.device)
        )  # (batch_size, 1)
        theta_all = torch.cat([theta_joints, theta_fixed], dim=1)  # (batch_size, 8)
        # expand params
        A = A.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, 8)
        alpha = alpha.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, 8)
        D = D.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, 8)

        # calculate transforms
        transformations = self.forward_kinematics(
            A, alpha, D, theta_all
        )  # (batch_size, 8, 4, 4)

        pose_to_Tw0 = pose.to(dtype=self.dtype)  # (batch_size, 4, 4)
        pose_to_T01 = torch.matmul(
            pose_to_Tw0, transformations[:, 0]
        )  # (batch_size, 4, 4)
        pose_to_T12 = torch.matmul(
            pose_to_T01, transformations[:, 1]
        )  # (batch_size, 4, 4)
        pose_to_T23 = torch.matmul(
            pose_to_T12, transformations[:, 2]
        )  # (batch_size, 4, 4)
        pose_to_T34 = torch.matmul(
            pose_to_T23, transformations[:, 3]
        )  # (batch_size, 4, 4)
        pose_to_T45 = torch.matmul(
            pose_to_T34, transformations[:, 4]
        )  # (batch_size, 4, 4)
        pose_to_T56 = torch.matmul(
            pose_to_T45, transformations[:, 5]
        )  # (batch_size, 4, 4)
        pose_to_T67 = torch.matmul(
            pose_to_T56, transformations[:, 6]
        )  # (batch_size, 4, 4)
        pose_to_T78 = torch.matmul(
            pose_to_T67, transformations[:, 7]
        )  # (batch_size, 4, 4)

        return [
            pose_to_Tw0,
            pose_to_T01,
            pose_to_T12,
            pose_to_T23,
            pose_to_T34,
            pose_to_T45,
            pose_to_T56,
            pose_to_T67,
            pose_to_T78,
        ]

    def forward_kinematics(self, A, alpha, D, theta):
        batch_size, num_joints = theta.shape

        # expand params
        theta = theta.view(batch_size, num_joints, 1)  # (batch_size, num_joints, 1)
        alpha = alpha.view(batch_size, num_joints, 1)  # (batch_size, num_joints, 1)
        D = D.view(batch_size, num_joints, 1)  # (batch_size, num_joints, 1)
        A = A.view(batch_size, num_joints, 1)  # (batch_size, num_joints, 1)

        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)
        zeros = torch.zeros_like(c_theta)
        ones = torch.ones_like(c_theta)

        transformations = torch.cat(
            [
                c_theta,
                -s_theta,
                zeros,
                A,
                s_theta * c_alpha,
                c_theta * c_alpha,
                -s_alpha,
                -s_alpha * D,
                s_theta * s_alpha,
                c_theta * s_alpha,
                c_alpha,
                c_alpha * D,
                zeros,
                zeros,
                zeros,
                ones,
            ],
            dim=2,
        ).reshape(
            batch_size, num_joints, 4, 4
        )  # (batch_size, num_joints, 4, 4)

        return transformations

    def get_eef(self, pose, theta, link=-1):
        poses = self.get_transformations_each_link(
            pose, theta
        )  # List: (dof + 1) * [(B, 4, 4)]

        # compute the pos for the gripper center
        if self.tcp_offset is not None:
            T78 = poses[-1]
            T_offset = (torch.eye(4).unsqueeze(0).repeat(T78.shape[0], 1, 1)).to(
                device=self.device, dtype=self.dtype
            )
            T_offset[:, 2, 3] = self.tcp_offset
            T_gripper_center = torch.bmm(T78, T_offset)
            pos = T_gripper_center[:, :3, 3]
            rot = T_gripper_center[:, :3, :3]
        else:
            pos = poses[link][:, :3, 3]
            rot = poses[link][:, :3, :3]

        return pos, rot, poses

    def nullspace_policy(self, main_task, auxiliary_task):

        # reshape from B, D to B, _, D
        J1, J2 = main_task.unsqueeze(1), auxiliary_task.unsqueeze(1)

        # Ensure the input tensors have the correct dimensions
        assert J1.shape == J2.shape, "J1 and J2 must have the same shape"

        B, D = J1.shape[0], J1.shape[-1]

        # Compute the pseudoinverse of J1
        pinvJ1 = torch.linalg.pinv(J1)
        # Compute the nullspace projector of J1
        NJ1 = torch.eye(D).to(device=self.device, dtype=self.dtype).unsqueeze(0).expand(
            B, -1, -1
        ) - torch.bmm(pinvJ1, J1)

        # Project J2 onto the nullspace of J1
        J2NJ1 = torch.bmm(J2, NJ1)

        # reshape from B, _, D to B, D
        J2NJ1, J1 = J2NJ1.squeeze(1), J1.squeeze(1)
        return J2NJ1 + J1

    def get_pertubed_eef(self, pose, theta, link=-1, delta=1e-3):
        B, dof = theta.shape
        device = theta.device

        # q_new = q + dq
        theta_perturbed = theta.unsqueeze(1).repeat(1, dof, 1)  # (B, dof, dof)
        indices = torch.arange(dof, device=device)
        theta_perturbed[:, indices, indices] += delta
        theta_perturbed = theta_perturbed.view(-1, dof)  # (B * dof, dof)
        pose_expanded = (
            pose.unsqueeze(1).repeat(1, dof, 1, 1).view(-1, 4, 4)
        )  # (B * dof, 4, 4)

        # combine for batch calculation
        theta_combined = torch.cat(
            [theta, theta_perturbed], dim=0
        )  # (B + B * dof, dof)
        pose_combined = torch.cat([pose, pose_expanded], dim=0)  # (B + B * dof, 4, 4)

        # get eef
        pos_combined, rot_combined, trans_list = self.get_eef(
            pose_combined, theta_combined, link=link
        )  # (B + B * dof, 3), (B + B * dof, 3, 3), (list:(dof+1)*[B*(1 + dof)], 4, 4)

        pos_initial = pos_combined[:B]  # (B, 3)
        rot_initial = rot_combined[:B]  # (B, 3, 3)
        pos_perturbed = pos_combined[B:].view(B, dof, 3)  # (B, dof, 3)
        rot_perturbed = rot_combined[B:].view(B, dof, 3, 3)  # (B, dof, 3, 3)

        eef_cur = EEfTarget(pos_initial, rot_initial)
        eef_pertubed = EEfTarget(pos_perturbed, rot_perturbed)
        return eef_cur, eef_pertubed, trans_list

    def compute_dpos(self, pos_initial: Tensor, pos_final: Tensor, delta: float) -> Tensor:
        return (pos_final - pos_initial) / delta  # (B, 3)

    def compute_drot(self, rot_initial: Tensor, rot_final: Tensor, delta: float) -> Tensor:
        delta_R = rot_final @ rot_initial.transpose(1, 2)  # (B, 3, 3)
        trace = delta_R.diagonal(dim1=1, dim2=2).sum(dim=1)  # (B,)
        theta = (
            torch.acos(((trace - 1) / 2).clamp(-1.0, 1.0)).clamp(min=1e-6).unsqueeze(-1)
        )  # rotation angular: (B, 1)
        axis = torch.stack(
            [
                delta_R[:, 2, 1] - delta_R[:, 1, 2],
                delta_R[:, 0, 2] - delta_R[:, 2, 0],
                delta_R[:, 1, 0] - delta_R[:, 0, 1],
            ],
            dim=1,
        ) / (
            2 * theta
        )  # rotation axis: (B, 3)
        return (theta * axis) / delta  # (B, 3)

    def compute_end_effector_velocity(
        self,
        pos_initial: Tensor,
        rot_initial: Tensor,
        pos_final: Tensor,
        rot_final: Tensor,
        timestamps: Tensor,
    ) -> Tensor:
        # Note: v and omega represent relative value without unit
        # calculate linear velocity
        v = self.compute_dpos(pos_initial, pos_final, timestamps)  # (B, 3)

        # calculate angular velocity
        omega = self.compute_drot(rot_initial, rot_final, timestamps)

        delta_x = torch.cat((v, omega), dim=1)  # (B, 6)

        return delta_x

    def get_numerical_jacobian(
        self, pos_initial, rot_initial, pos_perturbed, rot_perturbed, delta=1e-3
    ):
        # Reshape for dof
        B, dof, dim = pos_perturbed.shape
        pos_perturbed = pos_perturbed.reshape(-1, dim)  # (B * dof, 3)
        rot_perturbed = rot_perturbed.reshape(-1, dim, dim)  # (B * dof, 3, 3)
        pos_initial = pos_initial.repeat_interleave(dof, dim=0)   # (B*dof, 3)
        rot_initial = rot_initial.repeat_interleave(dof, dim=0)   # (B*dof, 3, 3)

        # calculate linear velocity jocobian
        J_pos = self.compute_dpos(pos_initial, pos_perturbed, delta)  # (B * dof, 3)
        J_pos = J_pos.view(B, dof, dim).transpose(1, 2)  # (B, 3, dof)

        # calculate angular velocity Jacobian
        J_rot = self.compute_drot(rot_initial, rot_perturbed, delta)  # (B * dof, 3)
        J_rot = J_rot.view(B, dof, dim).transpose(1, 2)  # (B, 3, dof)

        J_full = torch.cat([J_pos, J_rot], dim=1)  # (B, 6, dof)
        return J_full

    def is_jacobian_singular(self, J, threshold=1e-3) -> bool:
        # Compute singular values of the Jacobian
        U, S, Vh = torch.linalg.svd(J)  # Full SVD
        min_singular_value = S.min(dim=1).values
        is_singular = (min_singular_value < threshold).any()

        if is_singular:
            print("Singular Jacobian")

        return is_singular

    def damped_inverse(self, J, lambda_=1e-2) -> Tensor:
        (
            B,
            _,
            dof,
        ) = J.shape  # Extract batch size, task space dimension, and degrees of freedom
        I = (
            torch.eye(dof, device=J.device).unsqueeze(0).expand(B, dof, dof)
        )  # Identity matrix (B, dof, dof)
        J_T = J.transpose(1, 2)  # Transpose of Jacobian (B, dof, dim)

        # Damped least squares formula: (J^T * J + lambda^2 * I)^-1 * J^T
        damped_inverse = torch.linalg.inv(J_T @ J + lambda_**2 * I) @ J_T
        return damped_inverse

    def map_taskspace_to_jointspace(
        self, task_vec: Tensor, eef_cur, eef_pertubed, delta: float = 1e-3
    ) -> Tensor:
        """Map a task-space velocity into configuration space via the
        numerical Jacobian and its damped pseudo-inverse: q_dot = J^dagger * v.

        Args:
            task_vec: (B, D) task-space velocity, D <= 6 (position-only
                vectors are zero-padded on the angular components).
            eef_cur, eef_pertubed: current/perturbed end-effector poses, e.g.
                from `get_pertubed_eef(...)`.
        """
        J = self.get_numerical_jacobian(
            eef_cur.position, eef_cur.rotation, eef_pertubed.position, eef_pertubed.rotation, delta
        )  # (B, 6, dof)

        task_dim = task_vec.shape[-1]
        if task_dim < J.shape[1]:
            pad = torch.zeros(
                *task_vec.shape[:-1],
                J.shape[1] - task_dim,
                device=task_vec.device,
                dtype=task_vec.dtype,
            )
            task_vec = torch.cat([task_vec, pad], dim=-1)

        damped_inverse_J = self.damped_inverse(J)
        return torch.matmul(damped_inverse_J, task_vec.unsqueeze(-1)).squeeze(-1)

    def check_feasibility(self, theta: Tensor) -> Tuple[bool, Tensor]:
        feasible = torch.all((theta >= self.theta_min) & (theta <= self.theta_max))
        theta_clipped = torch.clamp(theta, self.theta_min_soft, self.theta_max_soft)
        if not torch.equal(theta, theta_clipped):
            print("Warning: reaching the joint limit")
        return feasible, theta_clipped

    def get_dof(self) -> int:
        return self.dof
