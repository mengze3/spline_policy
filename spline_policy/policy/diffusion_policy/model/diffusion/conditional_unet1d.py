import os, sys

tool_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if tool_dir not in sys.path:
    sys.path.append(tool_dir)

from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [
                    # down encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    # up encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        **kwargs,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x


if __name__ == "__main__":
    import torch
    import time
    from thop import profile
    import matplotlib.pyplot as plt

    input_dim = 514
    local_cond_dim = None
    global_cond_dim = 10
    diffusion_step_embed_dim = 128
    kernel_size = 5
    n_groups = 8
    down_dims = [128, 256, 512]
    cond_predict_scale = True

    T_list = [4, 8, 16, 32, 64]
    batch_size = 256
    denoise_step = 10

    model = (
        ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            kernel_size=kernel_size,
            n_groups=n_groups,
            down_dims=down_dims,
            cond_predict_scale=cond_predict_scale,
        )
        .cuda()
        .eval()
    )

    def measure_runtime(T):
        x = torch.randn(batch_size, T, input_dim).cuda()
        local_cond = None
        global_cond = torch.randn(batch_size, global_cond_dim).cuda()
        timestep = torch.randint(0, 100, (batch_size,), device="cuda")

        with torch.no_grad():
            for _ in range(100):
                _ = model(x, timestep, local_cond=local_cond, global_cond=global_cond)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(denoise_step):
                _ = model(x, timestep, local_cond=local_cond, global_cond=global_cond)
        torch.cuda.synchronize()
        t1 = time.time()
        return t1 - t0, x, timestep, local_cond, global_cond

    # Assuming T_list, measure_runtime, and model are defined
    macs_list = []
    params_list = []
    runtime_ms_list = []
    flops_list = [] # New list to store FLOPs

    print("--- Performance Analysis ---")
    for T in T_list:
        runtime, x, t, lc, gc = measure_runtime(T)
        runtime_ms_list.append(1000 * runtime)

        # Prepare single-item inputs for profiling
        single_x = x[0:1]
        single_t = t[0:1]
        single_gc = gc[0:1]
        
        # Use thop.profile to get MACs and Params
        macs, params = profile(
            model, inputs=(single_x, single_t, lc, single_gc), verbose=False
        )
        
        # --- The Conversion Happens Here ---
        # 1. Convert MACs to FLOPs
        flops = macs * 2
        
        # 2. Convert to standard units (Giga for FLOPs/MACs, Mega for Params)
        gmacs = macs / 1e9
        gflops = flops / 1e9 # Convert FLOPs to GFLOPs
        mparams = params / 1e6
        
        # 3. Append to lists for plotting or further analysis
        macs_list.append(gmacs)
        flops_list.append(gflops) # Store GFLOPs
        params_list.append(mparams)

        # 4. Print results, showing both MACs and FLOPs for clarity
        print(
            f"T = {T:<4} | Runtime: {1000*runtime:.2f} ms | "
            f"MACs: {gmacs:.4f} G | "
            f"FLOPs: {gflops:.4f} G | " # Added FLOPs to the printout
            f"Params: {mparams:.2f} M"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(T_list, runtime_ms_list, marker="o")
    axes[0].set_title("Runtime vs T")
    axes[0].set_xlabel("T")
    axes[0].set_ylabel("Runtime (ms)")
    axes[0].grid(False)

    axes[1].plot(T_list, macs_list, marker="s", color="orange")
    axes[1].set_title("MACs vs T")
    axes[1].set_xlabel("T")
    axes[1].set_ylabel("MACs (G)")
    axes[1].grid(False)

    axes[2].plot(T_list, params_list, marker="^", color="green")
    axes[2].set_title("Params vs T")
    axes[2].set_xlabel("T")
    axes[2].set_ylabel("Params (M)")
    axes[2].grid(False)

    plt.tight_layout()
    plt.show()
