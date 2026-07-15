if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace as _BaseWorkspace,
)
from diffusion_policy.policy.diffusion_unet_image_spline_policy import (
    DiffusionUnetImagePolicy,
)


class TrainDiffusionUnetImageSplineWorkspace(_BaseWorkspace):
    """Workspace that fine-tunes the spline-based image diffusion policy."""

    def __init__(self, cfg: DictConfig, output_dir: Optional[str] = None):
        cfg_copy = copy.deepcopy(cfg)
        if not cfg_copy.get("action_dim"):
            shape_meta_obj = OmegaConf.to_object(cfg_copy.shape_meta)
            cfg_copy.action_dim = int(shape_meta_obj["action"]["shape"][0])
        super().__init__(cfg_copy, output_dir=output_dir)
        assert isinstance(self.model, DiffusionUnetImagePolicy)
        if self.ema_model is not None:
            assert isinstance(self.ema_model, DiffusionUnetImagePolicy)
