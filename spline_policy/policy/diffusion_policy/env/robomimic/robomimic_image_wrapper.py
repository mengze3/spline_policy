from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key="agentview_image",
    ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1, 1
            if key.endswith("image"):
                min_value, max_value = 0, 1
            elif key.endswith("quat"):
                min_value, max_value = -1, 1
            elif key.endswith("qpos"):
                min_value, max_value = -1, 1
            elif key.endswith("pos"):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")

            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img


def test():
    
    import os
    from omegaconf import OmegaConf
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import cv2
    import collections
    from termcolor import cprint
    
    def create_env(env_meta, shape_meta, enable_render=True):
        modality_mapping = collections.defaultdict(list)
        for key, attr in shape_meta['obs'].items():
            modality_mapping[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=enable_render,
            use_image_obs=enable_render, 
        )
        return env

    cfg_path = os.path.expanduser(
        "spline_policy/config/train/task/can_render_image.yaml"
    )
    dataset_path = os.path.expanduser(
        "spline_policy/data/robomimic/datasets/can/ph/image_abs.hdf5"
    )
    
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg["shape_meta"]
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    channel, height, width = shape_meta["obs"]["agentview_image"]["shape"]
    env_meta["env_kwargs"]["camera_heights"], env_meta["env_kwargs"]["camera_widths"] = height, width
    cprint(f"env_meta: {env_meta}", color="red")
    cprint(f"shape_meta: {shape_meta}", color="yellow")

    env = create_env(env_meta=env_meta, shape_meta=shape_meta)

    wrapper = RobomimicImageWrapper(env=env, shape_meta=shape_meta)
    wrapper.seed(0)
    obs = wrapper.reset()
    img_rgb = wrapper.render()

    # Convert RGB image (from wrapper) to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    output_filename = f"spline_policy/policy/diffusion_policy/env/render_figs/{env_meta['env_name']}_robot_view.png"
    cv2.imwrite(output_filename, img_bgr)
    print(f"Success! Image saved to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    test()
