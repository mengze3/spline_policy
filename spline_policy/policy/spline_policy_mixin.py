from typing import Dict, Optional
import torch

from diffusion_policy.planning.quadratic_spline import (
    QuadraticSpline,
    dynamical_system_single_step,
)


class SplinePolicyMixin:
    """Shared Spline Policy glue: everything that only depends on
    `self.poly` (a QuadraticSpline) rather than on a specific denoising backbone.

    Concrete policies (diffusion/flow-matching x lowdim/image) keep their own model,
    scheduler/flow-matcher, obs encoding and denoising loop, and mix this in for:
    flow-field construction and the spline encode/decode glue used by
    `compute_loss`.

    Plain mixin: no `__init__`/`super()` chaining. Concrete classes call
    `self._init_spline(...)` from their own `__init__` after `super().__init__()`.
    """

    def _init_spline(
        self,
        poly: QuadraticSpline,
        lambda_dist: float = 0.5,
        step_size: float = 3.0,
    ):
        self.poly = poly
        self.lambda_dist = lambda_dist
        self.step_size = step_size

    # ========= flow-field realization ============
    def generate_dynamical_system(self, policy_dict: Dict[str, torch.Tensor]):
        Da = self.action_dim
        action_points = policy_dict["action"]  # (B, N, Da)
        action_points = action_points[..., :Da].to(self.device)  # (B, N, Da)
        w_params = self.poly.params
        action_pred, vec_pred = dynamical_system_single_step(
            self.poly,
            action_points,
            w_params,
            lambda_dist=self.lambda_dist,
            step_size=self.step_size,
        )  # (B, N, Da)
        return {"action": action_pred, "gradient_field": vec_pred}

    # ========= null-space control integration ============
    def null_space_project(
        self, main_task: torch.Tensor, auxiliary_task: torch.Tensor
    ) -> torch.Tensor:
        """Follow `auxiliary_task` wherever it doesn't conflict with the
        higher-priority `main_task`: project auxiliary_task onto the null
        space of main_task, then add main_task back in."""
        J1, J2 = main_task.unsqueeze(1), auxiliary_task.unsqueeze(1)
        B, D = J1.shape[0], J1.shape[-1]
        pinv_J1 = torch.linalg.pinv(J1)
        null_space = torch.eye(D, device=J1.device, dtype=J1.dtype).unsqueeze(0).expand(B, -1, -1) - torch.bmm(pinv_J1, J1)
        projected_auxiliary = torch.bmm(J2, null_space)
        return projected_auxiliary.squeeze(1) + main_task

    # ========= flow-field uncertainty propagation ============
    def propagate_flow_uncertainty(
        self,
        query_points: torch.Tensor,
        w_samples: torch.Tensor,
        lambda_dist: Optional[float] = None,
        step_size: Optional[float] = None,
    ):
        """Monte Carlo uncertainty propagation from a batch of sampled spline
        parameters: mean/variance of the decoded control points, the decoded
        trajectory, and the flow field they induce.

        Args:
            query_points: (N, Da) spatial grid to evaluate the flow field at.
            w_samples: (M, nbSeg+2, Da) sampled spline parameters, one
                single-spline (B=1) parameter set per Monte Carlo draw.
        Returns:
            ctrl_mean, ctrl_var: (K, Da) decoded control-point mean/variance.
            traj_mean, traj_var: (N_traj, Da) decoded trajectory mean/variance.
            field_mean, field_var: (N, Da) flow-vector mean/variance at each
                query point (reduce var e.g. via `.sum(-1)` for a scalar
                heatmap).

        Note: like `_decode_prediction`, this leaves `self.poly.params` set to
        `w_samples` (last write wins) -- call it on a dedicated spline
        instance if `self.poly.params` must stay at some other nominal fit.
        """
        lambda_dist = self.lambda_dist if lambda_dist is None else lambda_dist
        step_size = self.step_size if step_size is None else step_size
        device = query_points.device
        M = w_samples.shape[0]
        w_samples = w_samples.to(device)

        # decode every sample's trajectory + control points in one batched call each
        traj_samples = self._decode_prediction(w_samples.clone(), is_continue=False)  # (M, N_traj, Da)
        ctrl_samples = self.decode_control_points(w_samples)  # (M, K, Da)

        # p only needs a leading dim of 1 -- it broadcasts against w_samples'
        # batch dim throughout sdf_batch and dynamical_system_single_step's
        # own p_next = p + vec*step_size, giving (M, N, Da) directly.
        p = query_points.unsqueeze(0)  # (1, N, Da)
        w_flat = w_samples.reshape(M, -1)
        _, vec_samples = dynamical_system_single_step(
            self.poly, p, w_flat, lambda_dist=lambda_dist, step_size=step_size
        )  # (M, N, Da)

        return (
            ctrl_samples.mean(0), ctrl_samples.var(0),
            traj_samples.mean(0), traj_samples.var(0),
            vec_samples.mean(0), vec_samples.var(0),
        )

    # ========= training-time spline target/decode glue ============
    def _build_spline_target(self, action: torch.Tensor, horizon: int):
        """Upsample demo actions to `horizon` (dense loss target), then downsample
        to `poly.nbSeg + 2` as the seed for the implicit control-point target."""
        trajectory = torch.nn.functional.interpolate(
            action.permute(0, 2, 1), size=horizon, mode="linear", align_corners=True
        ).permute(0, 2, 1)
        ctrpls_num = self.poly.nbSeg + 2
        sample_trajectory = torch.nn.functional.interpolate(
            trajectory.clone().permute(0, 2, 1),
            size=ctrpls_num,
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)
        return trajectory, sample_trajectory

    def encode_trajectory(
        self, waypoints: torch.Tensor, sample_type: str = "uniform"
    ) -> torch.Tensor:
        """Fit raw waypoints (B, N, Da) to spline params, shaped (B, nbSeg+2, Da)
        ready for `_decode_prediction`."""
        w_batch, _ = self.poly.encode_trajectory(waypoints, sample_type=sample_type)
        return w_batch.reshape(waypoints.shape[0], self.poly.nbSeg + 2, self.poly.nbDim)

    def _decode_prediction(
        self,
        w_batch: torch.Tensor,
        receding_horizon: float = 1.0,
        is_continue: bool = False,
        last_state: Optional[torch.Tensor] = None,
        N: Optional[int] = None,
        return_control_points: bool = False,
    ) -> torch.Tensor:
        pred_traj, w_decode = self.poly.encode_trajectory_given_w(
            w_batch=w_batch,
            N=N,
            receding_horizon=receding_horizon,
            is_continue=is_continue,
            last_state=last_state,
        )
        return (pred_traj, w_decode) if return_control_points else pred_traj

    def decode_control_points(self, w_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Flattened Bezier control points (B, nbSeg*nbFct, Da) for `w_batch`, or
        for the most recently decoded params (`self.poly.params`) if not given."""
        w = self.poly.params if w_batch is None else w_batch
        B = w.shape[0]
        return self.poly.decode_w(w).reshape(B, -1, self.poly.nbDim)

