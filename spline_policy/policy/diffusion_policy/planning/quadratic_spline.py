import torch
import math
from typing import Tuple


class QuadraticSpline:
    def __init__(self, nbFct=3, nbSeg=10, nbDim=2, nPoints=64, device="cuda"):
        self.device = device
        self.nbFct = nbFct
        if nbFct != 3:
            raise ValueError("Only quadratic splines are supported!")
        self.nbSeg = nbSeg
        self.nbDim = nbDim
        self.nPoints = nPoints
        self.BC = self._compute_BC()  # (ctrl_num: nbFct * nbSeg, param_w: nbSeg + 2)
        self._params = None

    def binomial(self, n: int, i: int) -> int:
        if n >= 0 and i >= 0:
            return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
        return 0

    def block_diag(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            (A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]),
            dtype=A.dtype,
            device=self.device,
        )
        out[: A.shape[0], : A.shape[1]] = A
        out[A.shape[0] :, A.shape[1] :] = B
        return out

    # basis and contrained matrix
    def _compute_BC(self) -> torch.Tensor:
        B0 = torch.zeros((self.nbFct, self.nbFct), device=self.device)
        for n in range(1, self.nbFct + 1):
            for i in range(1, self.nbFct + 1):
                B0[self.nbFct - i, n - 1] = (
                    (-1) ** (self.nbFct - i - n)
                    * (-self.binomial(self.nbFct - 1, i - 1))
                    * self.binomial(
                        self.nbFct - 1 - (i - 1), self.nbFct - 1 - (n - 1) - (i - 1)
                    )
                )
        B = torch.kron(
            torch.eye(self.nbSeg, device=self.device), B0
        )  # (nbDim*nbFct, nbDim*nbFct)

        C0 = torch.tensor([[1.0], [1.0], [2.0]], device=self.device)
        C = torch.eye(self.nbFct - 1, device=self.device)
        for _ in range(self.nbSeg - 1):
            C = self.block_diag(C, C0)
        C = self.block_diag(C, torch.eye(self.nbFct - 2, device=self.device))
        C[1:-1:6, 1] = 1
        C[4:-1:6, 1] = -1
        id = 4
        for n in range(self.nbSeg - 2):
            C[id:-1:6, n + 2] = 2
            C[id + 3 : -1 : 6, n + 2] = -2
            id += 3
        self.C = C  # (nbDim*nbFct, nbSeg+2)
        return B @ C  # (nbDim*nbFct, nbSeg+2)

    def computePsiList1D(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = torch.zeros((1, self.nbFct), device=self.device)  # (1, nbFct)
        dT = torch.zeros((1, self.nbFct), device=self.device)  # (1, nbFct)
        phi = torch.zeros(
            (len(t), self.BC.shape[1]), device=self.device
        )  # (num_points, param_w)
        dphi = torch.zeros_like(phi, device=self.device)  # (num_points, param_w)

        for k in range(len(t)):
            tt = (
                torch.remainder(t[k], 1.0 / self.nbSeg) * self.nbSeg
            )  # get relative time in [0, 1] for current seg
            id_float = torch.round(t[k] * self.nbSeg - tt)  # get id for current seg
            id = id_float.long()

            if id < 0:
                tt = tt + id  # id is negative here
                id = 0
            if id > (self.nbSeg - 1):
                tt = tt + (id - (self.nbSeg - 1))
                id = self.nbSeg - 1

            p1 = torch.arange(self.nbFct)  # [0, 1, ..., nbFct-1]
            T[0, :] = tt**p1
            dT[0, 1:] = p1[1:] * (tt ** (p1[1:] - 1)) * self.nbSeg
            idl = (id * self.nbFct + p1).long()

            phi[k, :] = T @ self.BC[idl, :]
            dphi[k, :] = dT @ self.BC[idl, :]

        Psi = torch.kron(
            phi, torch.eye(self.nbDim, device=self.device)
        )  # (nbDim*num_points, nbDim*param_w)
        dPsi = torch.kron(dphi, torch.eye(self.nbDim, device=self.device))
        return Psi, dPsi, phi  # Psi: (nbDim*num_points, nbDim*param_w)

    def encode_trajectory(
        self, data: torch.Tensor, sample_type: str = "uniform",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory parameters and reconstruct the trajectory.

        Args:
            data (torch.Tensor): Batched control points of shape (B, N, D),
                                where B is the batch size, N is the number of waypoints,
                                and D is the spatial dimension (e.g., 2 or 3).

        Returns:
            w_batch (torch.Tensor): Encoded trajectory parameters of shape (B, D * param_w),
                                    where param_w is the number of basis coefficients per dimension.
            trajectory (torch.Tensor): Reconstructed trajectories of shape (B, N, D).
        """
        B, N, D = data.shape  # (B, N, D)

        # 1. compute pos params of t for w params
        i = torch.arange(N)
        if sample_type == "uniform":
            t = torch.linspace(0, 1, N)
        elif sample_type == "chebyshev":
            # Chebyshev nodes mapped to [0,1]
            t = 0.5 * (1 - torch.cos((i + 0.5) * torch.pi / N))
        Psi, dPsi, phi = self.computePsiList1D(t)  # Psi: (N*D, param_w*D)
        Psi_pinv = torch.linalg.pinv(Psi.float())  # (param_w*D, N*D)

        # 2. solve w params
        # (B, N, D) -> (B, N*D)
        data_flat = data.reshape(B, -1).float()  # (B, N*D)
        # (param_w*D, N*D) x (B, N*D) -> (B, D*param_w)
        w_batch = torch.einsum("ij,bj->bi", Psi_pinv, data_flat)  # (B, D*param_w)

        # 3. calculate trajectory
        # (N*D, param_w*D) x (B, param_w*D) -> (B, N*D)
        traj_flat = torch.einsum("ij,bj->bi", Psi, w_batch)  # (B, N*D)
        trajectory = traj_flat.reshape(B, N, D)  # (B, N, D)

        return w_batch, trajectory

    def get_boundary_condition(self, receding_horizon: float):
        """
        Extract control points of the final segment that was used in the current receding horizon.

        Args:
            receding_horizon (float): Time span of the MPC rollout, in [0, 1].

        Returns:
            ctrl_pts (torch.Tensor): Shape (B, nbFct=3, nbDim), control points of the last used segment.
            seg_idx (int): The segment index corresponding to the final used segment (shared across batch).
        """
        w_batch = self.params
        B = w_batch.shape[0]

        # 1. Decode into control points: (B, nbSeg, nbFct, nbDim)
        w_decode = self.decode_w(w_batch)

        # 2. Determine how many full segments are covered in receding horizon
        segment_len = 1.0 / self.nbSeg
        seg_idx = int((receding_horizon + 1e-3) / segment_len)
        # 3. Clamp segment index to valid range
        seg_idx = min(seg_idx - 1, self.nbSeg - 1)  # last used segment, zero-based

        # 4. Extract segment control points: (B, nbFct=3, nbDim)
        ctrl_pts = w_decode[:, seg_idx, :, :]  # (B, 3, D)

        return ctrl_pts

    def set_boundary_condition(
        self, last_ctrl_pts: torch.Tensor, w_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce strict C1 continuity by adding two new spline segments at the start
        and shifting the original spline, following user-defined logic.

        Args:
            last_ctrl_pts (torch.Tensor): (B, 3, nbDim), previous spline segment's control points.
            w_batch (torch.Tensor): (B, nbSeg+2, nbDim), spline parameters.

        Returns:
            w_batch_new (torch.Tensor): updated spline parameters with new first two segments.
        """
        B, nbFct, nbDim = last_ctrl_pts.shape
        nbSeg = self.nbSeg

        # Step 1 - Compute first segment's P0 and P1
        # P0 = w_prev^3
        P0_last = last_ctrl_pts[:, 2, :]  # (B, D)

        # P1 = 2*w_prev^3 - w_prev^2
        P1_last = 2.0 * last_ctrl_pts[:, 2, :] - last_ctrl_pts[:, 1, :]  # (B, D)

        # Step 2 - Inverse compute previous segment's second point from new spline
        # decode current w_batch → control points
        w_decode = self.decode_w(w_batch)  # (B, nbSeg, 3, nbDim)

        # original second control point of w_batch
        P1_new = w_decode[:, 1, 1, :]  # (B, D)

        # Step 3 - Compute midpoint of the two second points
        P0_new = 0.5 * (P1_last + P1_new)  # (B, D)
        w_decode[:, 1, 0, :] = P0_new

        # Step 4 - Construct new segments
        seg0_new = torch.stack([P0_last, P1_last, P0_new], dim=1)  # (B, 3, D)

        # Shift original spline segments to make room for 2 new segments
        w_decode_new = torch.zeros_like(w_decode)

        # Place new seg0 and seg1
        w_decode_new[:, 0, :, :] = seg0_new
        w_decode_new[:, 1:, :, :] = w_decode[:, 1:, :, :]

        # Encode back to spline parameters
        w_decode_flat = w_decode_new.reshape(B, -1)

        # Build kron matrix
        kron_matrix = torch.kron(self.C, torch.eye(nbDim, device=self.device)).float()

        # Least squares solve
        w_batch_new_flat = torch.linalg.lstsq(kron_matrix, w_decode_flat.T).solution.T

        w_batch_new = w_batch_new_flat.reshape(B, -1, nbDim)

        return w_batch_new

    def encode_trajectory_given_w(
        self,
        w_batch: torch.Tensor,
        N: int = None,
        receding_horizon: float = 0.5,
        last_state: torch.Tensor = None,
        sample_type: str = "uniform",
        is_continue: bool = True,
    ) -> torch.Tensor:
        """Reconstruct trajectory from encoded parameters.

        Args:
            w_batch (torch.Tensor): Encoded trajectory parameters of shape (B, param_w, D),
                                    where B is the batch size, D is the spatial dimension,
                                    and param_w is the number of basis coefficients per dimension.
            N (int, optional): Number of trajectory points to evaluate. Defaults to 200.

        Returns:
            trajectory (torch.Tensor): Reconstructed trajectories of shape (B, N, D),
                                    where N is the number of points and D is the spatial dimension.
        """
        B, M, D = w_batch.shape
        if N is None:
            N = self.nPoints

        # 1. compute pos params of t for params
        i = torch.arange(N)
        if sample_type == "uniform":
            t = torch.linspace(0, 1, N)
        elif sample_type == "chebyshev":
            # Chebyshev nodes mapped to [0,1]
            t = 0.5 * (1 - torch.cos((i + 0.5) * torch.pi / N))
        # enable receding horizon fashion
        segment_len = 1.0 / self.nbSeg
        seg_num = int(receding_horizon // segment_len)
        receding_horizon = seg_num * segment_len
        t = t * receding_horizon

        # calculate coefficient matrix
        Psi, dPsi, phi = self.computePsiList1D(t)  # Psi: (N*D, param_w*D)

        # 2. set boundary condition
        if is_continue:
            if self.params is not None:
                last_ctrl_pts = self.get_boundary_condition(receding_horizon)
                w_batch = self.set_boundary_condition(last_ctrl_pts, w_batch)
            elif last_state is not None:
                w_batch[:, 0, :] = last_state
        w_batch = w_batch.reshape(B, -1)  # (B, param_w*D)

        # 3. calculate trajectory
        # (N*D, param_w*D) x (B, param_w*D) -> (B, N*D)
        traj_flat = torch.einsum("ij,bj->bi", Psi, w_batch)  # (B, N*D)
        trajectory = traj_flat.reshape(B, N, self.nbDim)  # (B, N, D)

        # 4. save params
        self.params = w_batch

        w_decode = self.decode_w(w_batch.clone())
        w_decode = w_decode[:, :seg_num, :, :]

        return trajectory, w_decode

    @property
    def params(self) -> torch.Tensor:
        return self._params

    @params.setter
    def params(self, w_batch: torch.Tensor):
        self._params = w_batch

    @staticmethod
    def quadratic_bezier_curve(
        t, control_pts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        P0, P1, P2 = control_pts  # P: (B, N, nbDim)
        t = t.unsqueeze(-1)  # (B, N, 1)
        return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

    @staticmethod
    def quadratic_bezier_curve_grad(
        t, control_pts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        P0, P1, P2 = control_pts  # P: (B, N, nbDim)
        t = t.unsqueeze(-1)  # (B, N, 1)
        return 2 * (1 - t) * (P1 - P0) + 2 * t * (P2 - P1)

    def decode_w(self, w_batch: torch.Tensor) -> torch.Tensor:
        """Decode the compressed trajectory parameter vector into segment-wise coefficients.

        Args:
            w (torch.Tensor): Encoded trajectory parameters of shape (B, (nbSeg + 2) * nbDim),
                            where B is the batch size, nbSeg is the number of segments,
                            and nbDim is the spatial dimension.

        Returns:
            w_decode (torch.Tensor): Decoded trajectory coefficients of shape (B, nbSeg, nbFct, nbDim),
                                    where nbFct is the number of basis functions per segment.
        """
        B = w_batch.shape[0]
        w_batch = w_batch.reshape(B, -1)  # (B, param_w*D)
        kron_matrix = torch.kron(
            self.C, torch.eye(self.nbDim, device=self.device)
        ).float()  # ((nbSeg * nbFct) * nbDim, (nbSeg + 2) * nbDim)
        # ((nbSeg * nbFct) * nbDim, (nbSeg + 2) * nbDim) x (B, (nbSeg+2) * nbDim) -> (B, nbSeg * nbFct * nbDim)
        w_decode = torch.einsum(
            "ij,bj->bi", kron_matrix, w_batch.float()
        )  # (B, nbSeg * nbFct * nbDim)
        w_decode = w_decode.reshape(
            B, self.nbSeg, self.nbFct, self.nbDim
        )  # (B, nbSeg, nbFct, nbDim)
        return w_decode

    def solve_cubic(self, a, b, c):
        """Find the signed distance from a point to a quadratic bezier curve.
        Supports batch operations in PyTorch.
        """
        p = b - (a**2) / 3
        q = a * (2 * (a**2) - 9 * b) / 27 + c
        R = (0.5 * q) ** 2 + (p / 3) ** 3
        batch_size = a.shape[0] if a.dim() > 0 else 1
        results = torch.zeros((batch_size, 3), device=a.device, dtype=a.dtype)

        # Case where R >= 0
        mask_real = R >= 0.0
        if mask_real.any():
            s = torch.sqrt(R[mask_real])
            s1 = -0.5 * q[mask_real] + s
            s2 = -0.5 * q[mask_real] - s
            v = torch.sign(s1) * torch.abs(s1) ** (1 / 3)
            w = torch.sign(s2) * torch.abs(s2) ** (1 / 3)
            results[mask_real.squeeze()] = (
                v.unsqueeze(-1) + w.unsqueeze(-1) - a[mask_real].unsqueeze(-1) / 3
            ).repeat(1, 3)

        # Case where R < 0 (3 real solutions)
        mask_complex = ~mask_real
        if mask_complex.any():
            S = 2 * torch.sqrt(-p[mask_complex] / 3)
            theta = torch.acos(
                1.5
                * q[mask_complex]
                / p[mask_complex]
                * torch.sqrt(-3.0 / p[mask_complex])
            )
            angles = (
                theta.unsqueeze(-1)
                + 2 * torch.arange(3, device=a.device).float() * torch.pi
            ) / 3
            results[mask_complex.squeeze()] = (
                S.unsqueeze(-1) * torch.cos(angles) - a[mask_complex].unsqueeze(-1) / 3
            )
        return results

    def quadractic_bezier_curve_batch(
        self, t_batch: torch.Tensor, W: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate batched quadratic Bézier curves at given time steps.

        Args:
            t_batch (torch.Tensor): Normalized time steps of shape (B, N),
                                    where B is the batch size and N is the number of evaluation points.
                                    Each entry lies in [0, nbSeg), where the integer part indicates
                                    the segment index, and the fractional part indicates relative position.
            W (torch.Tensor): Compressed Bézier parameters of shape (B, (nbSeg + 2) * nbDim),
                            to be decoded into control points.

        Returns:
            curve (torch.Tensor): Evaluated points on the Bézier curve of shape (B, N, nbDim),
                                corresponding to the input time steps.
        """
        B, N = t_batch.shape
        device = t_batch.device

        # 1. Decode weights to get control points
        w_decode = self.decode_w(W)  # (B, nbSeg, nbFct=3, nbDim)
        w_decode = w_decode.reshape(
            B, self.nbFct * self.nbSeg, self.nbDim
        )  # (B, nbFct*nbSeg, nbDim)

        # 2. Compute segment indices
        seg_index = t_batch.int()  # (B, N)
        relative_t = t_batch - seg_index  # (B, N)

        # 3. Clamp segment index to ensure +2 doesn’t overflow
        max_valid_seg = (self.nbSeg - 1) * self.nbFct
        seg_index = torch.clamp(self.nbFct * seg_index, max=max_valid_seg)  # (B, N)

        # 4. Gather control points (P0, P1, P2)
        B_idx, N_idx = torch.meshgrid(
            torch.arange(B, device=device),
            torch.arange(N, device=device),
            indexing="ij",
        )
        P0 = w_decode[B_idx, seg_index]  # (B, N, D)
        P1 = w_decode[B_idx, seg_index + 1]  # (B, N, D)
        P2 = w_decode[B_idx, seg_index + 2]  # (B, N, D)
        ctrl_points = (P0, P1, P2)

        # 5. Calculate gradient
        curve = self.quadratic_bezier_curve(relative_t, ctrl_points)  # (B, N, D)

        return curve

    def quadractic_bezier_curve_grad_batch(self, t_batch, W):
        """Evaluate the gradient (velocity) of batched quadratic Bézier curves at given time steps.

        Args:
            t_batch (torch.Tensor): Normalized time steps of shape (B, N),
                                    where B is the batch size and N is the number of sample points.
                                    Each value should lie in [0, nbSeg), where the integer part
                                    specifies the segment index, and the fractional part is the relative time.
            W (torch.Tensor): Compressed Bézier parameters of shape (B, (nbSeg + 2) * nbDim),
                            which are decoded into control points.

        Returns:
            curve_grad (torch.Tensor): Gradient (velocity) of the Bézier curve at each time step,
                                    with shape (B, N, nbDim). The velocity is set to zero for time
                                    steps beyond the valid trajectory range.
        """
        B, N = t_batch.shape
        device = t_batch.device

        # 1. Decode weights to get control points
        w_decode = self.decode_w(W)  # (B, nbSeg, nbFct=3, nbDim)
        w_decode = w_decode.reshape(
            B, self.nbFct * self.nbSeg, self.nbDim
        )  # (B, nbFct*nbSeg, nbDim)

        # 2. Compute segment indices
        seg_index = t_batch.int()  # (B, N)
        relative_t = t_batch - seg_index  # (B, N)

        # 3. Clamp segment index to ensure +2 doesn’t overflow
        max_valid_seg = (self.nbSeg - 1) * self.nbFct
        seg_index = torch.clamp(self.nbFct * seg_index, max=max_valid_seg)  # (B, N)

        # 4. Gather control points (P0, P1, P2)
        B_idx, N_idx = torch.meshgrid(
            torch.arange(B, device=device),
            torch.arange(N, device=device),
            indexing="ij",
        )
        P0 = w_decode[B_idx, seg_index]  # (B, N, D)
        P1 = w_decode[B_idx, seg_index + 1]  # (B, N, D)
        P2 = w_decode[B_idx, seg_index + 2]  # (B, N, D)
        ctrl_points = (P0, P1, P2)

        # 5. Calculate gradient
        curve_grad = self.quadratic_bezier_curve_grad(
            relative_t, ctrl_points
        )  # (B, N, D)

        # 6. Set end velocity
        curve_grad[t_batch >= (self.nbSeg - 1e-2)] = 0  # (B, N, D)

        return curve_grad

    def quadratic_bezier_curve_sdf_batch(self, W, p):
        """Find the signed distance from points to quadratic bezier curves.

        Args:
            W: Batched quadratic bezier control points with shape (B, 3, D)
            p: Points to evaluate with shape (B, N, D)

        Returns:
            distance: Shape (B, N)
            grad: Shape (B, N, D)
            tmin: Shape (B, N)
        """
        # Split control points
        P0, P1, P2 = W.unbind(dim=1)  # each (B, D)

        # Prepare for broadcasting
        d = P0.unsqueeze(1) - p  # (B, N, D)
        p1 = 2 * (P1 - P0)  # (B, D)
        p2 = P0 - 2 * P1 + P2  # (B, D)

        # Compute cubic equation coefficients
        B_coeff = 1.5 * torch.sum(p1.unsqueeze(1) * p2.unsqueeze(1), dim=-1)  # (B, 1)
        C_coeff = torch.sum(d * p2.unsqueeze(1), dim=-1) + 0.5 * torch.sum(
            p1.unsqueeze(1) * p1.unsqueeze(1), dim=-1
        )  # (B, N)
        D_coeff = 0.5 * torch.sum(d * p1.unsqueeze(1), dim=-1)  # (B, N)

        # Normalization factor
        a = torch.sum(p2 * p2, dim=-1)  # (B,)
        eps = 1e-8
        inv_a = 1.0 / (a.unsqueeze(-1).unsqueeze(-1) + eps)  # (B, 1, 1)
        # print(f"inv_a: {inv_a.shape}, B_coeff: {B_coeff.shape}, C_coeff: {C_coeff.shape}, D_coeff: {D_coeff.shape}")
        # Stack and normalize coefficients
        coefs = (
            torch.stack([B_coeff.expand(-1, p.shape[1]), C_coeff, D_coeff], dim=-1)
            * inv_a
        )  # (B, N, 3)

        # Solve cubic equation
        B, N = coefs.shape[:2]
        coefs_flat = coefs.reshape(-1, 3)  # (B*N, 3)
        t_candidates = torch.clamp(
            self.solve_cubic(*coefs_flat.T), 0.0, 1.0 - 1e-3
        )  # (B*N, 3)
        t_candidates = t_candidates.reshape(B, N, 3)  # (B, N, 3)

        # CORRECTED Curve evaluation: P(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
        t = t_candidates.unsqueeze(-1)  # (B, N, 3, 1)
        one_minus_t = 1 - t

        # Compute each term
        term0 = one_minus_t * one_minus_t * P0.unsqueeze(1).unsqueeze(2)  # (B, N, 3, D)
        term1 = 2 * one_minus_t * t * P1.unsqueeze(1).unsqueeze(2)
        term2 = t * t * P2.unsqueeze(1).unsqueeze(2)

        p_curve = term0 + term1 + term2  # (B, N, 3, D)

        # Compute distances to original points
        distances = torch.norm(p_curve - p.unsqueeze(2), dim=-1)  # (B, N, 3)

        # Find best t for each point
        min_dist, best_idx = torch.min(distances, dim=-1)  # both (B, N)

        # Gather results
        batch_idx = torch.arange(B, device=W.device).view(B, 1, 1).expand(-1, N, 1)
        point_idx = torch.arange(N, device=W.device).view(1, N, 1).expand(B, -1, 1)
        tmin = t_candidates.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
        p_closest = p_curve[batch_idx, point_idx, best_idx.unsqueeze(-1)].squeeze(
            -2
        )  # (B, N, D)

        # Compute gradient (normal vector)
        grad = p_closest - p  # (B, N, D)

        return min_dist, grad, tmin

    def sdf_batch(self, p: torch.Tensor, w_batch: torch.Tensor):
        """Compute batched signed distance fields (SDF) from points to Bézier curves.

        Args:
            p (torch.Tensor): Query points of shape (N, nbDim), where N is the number of points.
            w_batch (torch.Tensor): Encoded Bézier trajectory parameters of shape (B, (nbSeg + 2) * nbDim),
                                    where B is the batch size.

        Returns:
            dist (torch.Tensor): Minimum distance from each point to the trajectory, shape (B, N).
            grad (torch.Tensor): Normalized gradient (direction toward closest point), shape (B, N, nbDim).
            t (torch.Tensor): Absolute time (including segment offset) of the closest point on the curve,
                            shape (B, N).
        """
        B = w_batch.shape[0]
        w_batch = w_batch.reshape(B, -1)  # (B, (nbSeg + 2) * nbDim)
        _, N, _ = p.shape  # (B, N, nbDim)

        # 1. construct control points
        w_decode_batch = self.decode_w(w_batch)  # (B, nbSeg, nbFct, nbDim)
        w_decode_batch = w_decode_batch.reshape(
            B * self.nbSeg, self.nbFct, self.nbDim
        )  # (B * nbSeg, 3, nbDim)

        # 2. calculate sdf for each segments in spline, respect to sampling points in space
        dist, grad, tmin = self.quadratic_bezier_curve_sdf_batch(w_decode_batch, p)
        # dist: (B * nbSeg, N), grad: (B * nbSeg, N, nbDim), tmin: (B * nbSeg, N)
        dist = dist.view(B, self.nbSeg, N)
        grad = grad.view(B, self.nbSeg, N, self.nbDim)
        tmin = tmin.view(B, self.nbSeg, N)

        # 3. get the sdf of sampling points in space
        dist, dist_idx = torch.min(dist, dim=1)  # (B, N), (B, N)
        B_idx = torch.arange(B, device=self.device).view(B, 1)  # (B, 1)
        N_idx = torch.arange(N, device=self.device).view(1, N)  # (1, N)
        grad = grad[B_idx, dist_idx, N_idx]  # (B, N, nbDim)
        grad = torch.nn.functional.normalize(grad, dim=-1)

        # 4. relative time -> absolute time
        segment_offset = torch.arange(self.nbSeg, device=self.device).view(
            1, self.nbSeg, 1
        )  # (1, nbSeg, 1)
        tmin = tmin + segment_offset  # (B, nbSeg, N)
        t = tmin[B_idx, dist_idx, N_idx]  # (B, N)
        return dist, grad, t  # (B, N), (B, N, nbDim), (B, N)


def dynamical_system_single_step(curve, p, w, lambda_dist=0.03, step_size=3.0):
    """Compute one step of a guidance-based dynamical system toward a Bézier trajectory.

    Args:
        curve: A curve object that provides `sdf_batch` and `quadractic_bezier_curve_grad_batch` methods.
        p (torch.Tensor): Current positions of shape (B, N, nbDim), where N is number of points.
        w (torch.Tensor): Encoded trajectory parameters of shape (B, (nbSeg + 2) * nbDim).
        lambda_dist (float, optional): Weighting factor for the distance-based barrier function. Default is 0.5.
        step_size (float, optional): Step size for the update. Default is 0.1.

    Returns:
        p_next (torch.Tensor): Updated positions after one step, shape (B, N, nbDim).
        vec (torch.Tensor): Normalized direction vector used for the update, shape (B, N, nbDim).
    """
    # 1. compute sdf from points
    dist, grad, t = curve.sdf_batch(p, w)  # (B, N), (B, N, nbDim), (B, N)
    # 2. compute gradient in curve
    curve_grad = curve.quadractic_bezier_curve_grad_batch(t, w) * 0.1  # (B, N, nbDim)
    # 3. Compute barrier function
    barrier = 1.0 / (1 + lambda_dist * dist + 1e-6)
    # 4. Combine trajectory and gradient fields
    vec = curve_grad * barrier.unsqueeze(-1) + grad * (1 - barrier).unsqueeze(-1)
    vec = torch.nn.functional.normalize(vec, dim=-1)  # (B, N, nbDim)
    p_next = p + vec * step_size
    return p_next, vec



if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    import torch
    import torch.nn as nn
    import math
    from typing import Tuple
    
    # --- Step 1: Define the Module to be Tested ---
    class TrajectoryEncoder(nn.Module):
        def __init__(self, nbSeg=10, nbDim=2, nPoints=64, 
                     receding_horizon=0.5, sample_type="uniform", is_continue=True, device="cpu"):
            super().__init__()
            # Store the QuadraticSpline instance
            # This instance holds the pre-computed matrices like BC and C.
            self.spline = QuadraticSpline(nbSeg=nbSeg, nbDim=nbDim, nPoints=nPoints, device=device)
            
            # Store fixed parameters for the forward call
            self.nPoints_eval = nPoints
            self.receding_horizon = receding_horizon
            self.sample_type = sample_type
            self.is_continue = is_continue

        def forward(self, w_batch: torch.Tensor, last_state: torch.Tensor = None):
            """
            The forward method wraps the original `encode_trajectory_given_w` function.
            It takes tensors as input that are expected to change with each call.
            
            Args:
                w_batch (torch.Tensor): Encoded trajectory parameters, shape (B, param_w, D).
                last_state (torch.Tensor, optional): Initial state for boundary conditions. Shape (B, D).
            """
            # Call the method from the spline instance
            trajectory, w_decode = self.spline.encode_trajectory_given_w(
                w_batch=w_batch,
                N=self.nPoints_eval,
                receding_horizon=self.receding_horizon,
                last_state=last_state,
                sample_type=self.sample_type,
                is_continue=self.is_continue
            )
            return trajectory, w_decode

    # --- Step 2: Define Test Parameters ---
    B = 256               # Batch size
    nbDim = 3             # Spatial dimension (e.g., 3D)
    nbSeg = 6             # Number of spline segments
    nPoints_eval = 16     # Number of points to evaluate on the trajectory
    receding_horizon = 1.0
    # Use CUDA if available, otherwise fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # is_continue=False is chosen to avoid the lstsq path for this specific test
    is_continue_test = False 

    # param_w is the number of parameters per dimension, which is nbSeg + 2
    param_w = nbSeg + 2
    
    print("--- Test Configuration ---")
    print(f"Batch Size (B): {B}")
    print(f"Dimension (nbDim): {nbDim}")
    print(f"Segments (nbSeg): {nbSeg}")
    print(f"Evaluation Points (nPoints_eval): {nPoints_eval}")
    print(f"Device: {device}")
    print("-" * 26)

    # --- Step 3: Instantiate the Module ---
    model = TrajectoryEncoder(
        nbSeg=nbSeg,
        nbDim=nbDim,
        nPoints=nPoints_eval,
        receding_horizon=receding_horizon,
        is_continue=is_continue_test,
        device=device
    )
    model.eval()

    # --- Step 4: Create Dummy Input Data ---
    # `w_batch` is the main input, representing the parameters for B trajectories
    w_batch_dummy = torch.randn(B, param_w, nbDim, device=device)

    # `last_state` is an optional input. For is_continue=False, this path is not taken,
    # so we can safely set it to None.
    last_state_dummy = None

    # The inputs must be a tuple matching the forward method's signature
    dummy_inputs = (w_batch_dummy, last_state_dummy)

    # --- Step 5: Run the FLOPs Analysis ---
    # FlopCountAnalysis traces the model to count FLOPs.
    # Note: It may print warnings for unsupported operations.
    print("\nRunning FlopCountAnalysis... (Warnings for unsupported ops are expected)")
    flops_analyzer = FlopCountAnalysis(model, dummy_inputs)

    # --- Step 6: Print the Results ---
    total_flops = flops_analyzer.total()

    print("\n--- FLOPs Test Report (using fvcore) ---")
    print(f"Total FLOPs (GFLOPs): {total_flops / 1e9:.6f}")
    print(f"Total FLOPs (MFLOPs): {total_flops / 1e6:.4f}")
    
    print("\n--- Analysis by Operator ---")
    # This shows which operations contributed to the total FLOPs count.
    print(flops_analyzer.by_operator())
    
    print("\n--- Analysis by Module ---")
    # This shows FLOPs distribution among sub-modules (if any).
    print(flops_analyzer.by_module())