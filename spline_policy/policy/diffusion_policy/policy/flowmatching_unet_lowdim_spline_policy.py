from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from policy.spline_policy_mixin import SplinePolicyMixin
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.planning.quadratic_spline import QuadraticSpline
from diffusion_policy.flow_matcher.conditional_flow_matcher import (
    ConditionalFlowMatcher,
)


class FlowmatchingUnetLowdimPolicy(SplinePolicyMixin, BaseLowdimPolicy):
    def __init__(
        self,
        model: ConditionalUnet1D,
        flow_matcher: ConditionalFlowMatcher,
        poly: QuadraticSpline,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_local_cond=False,
        obs_as_global_cond=False,
        pred_action_steps_only=False,
        oa_step_convention=False,
        lambda_dist=0.5,
        step_size=3.0,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.flow_matcher = flow_matcher
        self._init_spline(poly, lambda_dist=lambda_dist, step_size=step_size)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = flow_matcher.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.flow_matcher

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        w_params = trajectory.clone()

        return w_params

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        is_continue: bool = False,
        receding_horizon: float = None,
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict  # not implemented yet
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.poly.nbSeg + 2
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        # run sampling
        nwparams = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        wparams = self.normalizer["action"].unnormalize(
            nwparams
        )  # (B, param_num, action_dim)
        # set boundary condition
        agent_pos_data = obs_dict.get("agent_pos")
        if agent_pos_data is not None:
            agent_obs = agent_pos_data[:, -1]

        else:
            agent_obs = None
        # recover new trajectory
        if not self.pred_action_steps_only:
            start = To
            if self.oa_step_convention:
                start = To - 1
            sample_num = self.n_action_steps + start
        else:
            sample_num = self.n_action_steps

        sample, w_decode = self._decode_prediction(
            wparams,
            receding_horizon=(
                sample_num / self.horizon
                if receding_horizon is None
                else receding_horizon
            ),
            is_continue=is_continue,
            last_state=agent_obs,
            N=sample_num,
            return_control_points=True,
        )

        action_pred = sample[..., :Da]  # (B, horizon, action_dim)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred, "ctrl_points": w_decode}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]
        B, data_horizon = action.shape[0], action.shape[1]
        horizon = self.horizon

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        # B, horizon, D -> B, ctrpls_num, D
        trajectory, sample_trajectory = self._build_spline_target(action, horizon)

        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:, self.n_obs_steps :, :] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, : self.n_obs_steps, :].reshape(obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                sample_trajectory = action[:, start:end]
        else:
            sample_trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(sample_trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(sample_trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(sample_trajectory.shape, device=sample_trajectory.device)
        bsz = sample_trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.rand(bsz, device=trajectory.device)
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
            sample_trajectory, noise, timesteps
        )

        # compute loss mask
        loss_mask = ~condition_mask  # not implemented yet

        # apply conditioning
        xt[condition_mask] = sample_trajectory[condition_mask]

        # Predict the noise residual
        pred = self.model(xt, timesteps, local_cond=local_cond, global_cond=global_cond)

        # 'sample': learning control points by e2e
        pred_traj = self._decode_prediction(pred, receding_horizon=1.0, is_continue=False)
        loss = F.mse_loss(pred_traj, trajectory, reduction="none")  # TODO: use mask
        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()
