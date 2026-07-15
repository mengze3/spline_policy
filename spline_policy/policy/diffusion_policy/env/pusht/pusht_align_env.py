import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTAlignEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=True,
        render_size=512,
        reset_to_state=None,
        obs_horizon=2,
        sim_action_space=None,
        seed=None,
        object_init=None,
        sim_hz=100,
    ):
        self._seed = seed
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = sim_hz
        self.obs_horizon = obs_horizon
        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["video.frames_per_second"]
        # legcay set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float64),
            high=np.array([ws, ws, ws, ws, np.pi * 2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64,
        )
        self.obs_history = list()

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )
        self.sim_action_space = sim_action_space

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
        self.object_init = object_init

        self.reset()

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array(
                [
                    rs.randint(50, 450),
                    rs.randint(50, 450),
                    rs.randint(100, 400),
                    rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                ]
            )
            if self.object_init is not None:
                object_init = np.array(self.object_init)
                object_init = self.sim2env(object_init)
                state[2:] = object_init
        self._set_state(state)
        self._set_state(state)

        observation = self._get_obs()
        return observation

    def sync_sim(self, sim_obs):
        env_obs = self.sim2env(sim_obs)  # align sim and env
        self._set_state(env_obs)  # set state in env
        self.step(action=None)  # run state in env

    def sim2env(self, sim_actions):
        x_min, x_max = self.sim_action_space[0]
        y_min, y_max = self.sim_action_space[1]
        ws = float(self.window_size)
        sim_actions = np.asarray(sim_actions)
        if sim_actions.ndim == 1:
            sim_actions = np.array([sim_actions])

        if sim_actions.shape[1] == 5:
            # agent + block
            env_points = np.zeros_like(sim_actions)
            env_points[:, 0] = np.interp(
                sim_actions[:, 0], [x_min, x_max], [0, ws]
            )  # agent_x
            env_points[:, 1] = np.interp(
                sim_actions[:, 1], [y_min, y_max], [0, ws]
            )  # agent_y
            env_points[:, 2] = np.interp(
                sim_actions[:, 2], [x_min, x_max], [0, ws]
            )  # block_x
            env_points[:, 3] = np.interp(
                sim_actions[:, 3], [y_min, y_max], [0, ws]
            )  # block_y
            env_points[:, 4] = sim_actions[:, 4]  #  block yaw
        elif sim_actions.shape[1] == 2:
            # agent
            env_points = np.zeros_like(sim_actions)
            env_points[:, 0] = np.interp(sim_actions[:, 0], [x_min, x_max], [0, ws])
            env_points[:, 1] = np.interp(sim_actions[:, 1], [y_min, y_max], [0, ws])
        elif sim_actions.shape[1] == 3:
            # block
            env_points = np.zeros_like(sim_actions)
            env_points[:, 0] = np.interp(sim_actions[:, 0], [x_min, x_max], [0, ws])
            env_points[:, 1] = np.interp(sim_actions[:, 1], [y_min, y_max], [0, ws])
            env_points[:, 2] = sim_actions[:, 2]  #  block yaw

        return env_points.squeeze()

    def env2sim(self, env_points):
        # env_points: (horizon, 5), 5 means agent pos + block pos + block yaw
        x_min, x_max = self.sim_action_space[0]
        y_min, y_max = self.sim_action_space[1]
        ws = float(self.window_size)
        env_points = np.asarray(env_points)
        if env_points.ndim == 1:
            env_points = np.array([env_points])

        # Initialize sim_actions with the same shape as env_points
        sim_actions = np.zeros_like(env_points)
        # Map agent_x (env_points[:, 0]) from [0, ws] to [x_min, x_max]
        sim_actions[:, 0] = np.interp(env_points[:, 0], [0, ws], [x_min, x_max])
        # Map agent_y (env_points[:, 1]) from [0, ws] to [y_min, y_max]
        sim_actions[:, 1] = np.interp(env_points[:, 1], [0, ws], [y_min, y_max])
        # Map block_x (env_points[:, 2]) from [0, ws] to [x_min, x_max]
        sim_actions[:, 2] = np.interp(env_points[:, 2], [0, ws], [x_min, x_max])
        # Map block_y (env_points[:, 3]) from [0, ws] to [y_min, y_max]
        sim_actions[:, 3] = np.interp(env_points[:, 3], [0, ws], [y_min, y_max])
        # Keep block_yaw (env_points[:, 4]) unchanged
        sim_actions[:, 4] = env_points[:, 4]

        return sim_actions.squeeze()

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            action = self.sim2env(action)
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                    Vec2d(0, 0) - self.agent.velocity
                )
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()
        observation = np.reshape(observation, (1, -1))  # (1, 5)
        # preserve observation steps
        if len(self.obs_history) >= self.obs_horizon:
            self.obs_history.pop(0)  # romove oldest observation
        self.obs_history.append(observation)

        # Zero-pad if history is not long enough
        if len(self.obs_history) < self.obs_horizon:
            # Compute how many zeros to pad
            pad_length = self.obs_horizon - len(self.obs_history)
            # Pad with zeros at the beginning
            padded_history = [observation] * pad_length + self.obs_history
            combined_obs = np.concatenate(padded_history, axis=0)
        else:
            combined_obs = np.concatenate(self.obs_history, axis=0)

        # Ensure shape is always (obs_horizon, 5)
        assert combined_obs.shape == (
            self.obs_horizon,
            5,
        ), f"Expected shape {(self.obs_horizon, 5)}, got {combined_obs.shape}"

        env_obs = combined_obs.copy()
        sim_obs = self.env2sim(combined_obs)

        return env_obs, sim_obs, reward, done, info

    def render(self, mode, traj=None):
        return self._render_frame(mode, traj)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), self.screen
            )
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position)
            + tuple(self.block.position)
            + (self.block.angle % (2 * np.pi),)
        )
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _render_frame(self, mode, traj):
        if traj is not None:
            traj = self.sim2env(traj)

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(
                    goal_body.local_to_world(v), draw_options.surface
                )
                for v in shape.get_vertices()
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        # Note: OpenCV uses image coordinates:
        #   - Origin (0, 0) is at the **top-left corner** of the image
        #   - The x-axis goes to the right → (increasing columns)
        #   - The y-axis goes downward ↓ (increasing rows)
        #
        #   Coordinate system diagram:
        #
        #       (0,0) ───────────→ x (cols)
        #         │
        #         │
        #         ↓
        #         y (rows)
        #
        #   So a pixel img[y, x] refers to the pixel at row y and column x
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if (
                self.render_action
                and (self.latest_action is not None)
                and traj is not None
            ):
                traj = np.array(traj)
                coord = traj[:, :2].astype(np.int32)
                marker_size = 8

                for idx, c in enumerate(coord):
                    color_intensity = int(255 * (idx / len(coord)))
                    color = (255 - color_intensity, 0, color_intensity)
                    cv2.circle(img, tuple(c), marker_size, color=color, thickness=-1)
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], rotation=self.goal_pose[2]
        )
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2], rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(matrix=tf_img_obj.params @ tf_obj_new.params)
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0])
            + list(tf_img_new.translation)
            + [tf_img_new.rotation]
        )
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color("LightGreen")
        self.goal_pose = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.90  # 90% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color(
            "LightGray"
        )  # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color("LightSlateGray")
        self.space.add(body, shape)
        return body

    def add_tee(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
