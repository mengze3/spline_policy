from collections import defaultdict
import numpy as np


class Logger:

    # Configuration for the "pusht" task
    _PUSHT_OBS_DIM = 5
    _PUSHT_POS_SLICE = slice(0, 2)

    # Configuration for the "lift" task
    _LIFT_OBS_DIM = 19
    _LIFT_POS_SLICE = slice(11, 14)

    # Configuration for the "can" or "square" task
    _CAN_OBS_DIM = _SQUARE_OBS_DIM = 23
    _CAN_POS_SLICE = _SQUARE_POS_SLICE = slice(15, 18)

    # Configuration for the "tool_transport" task
    _TOOL_TRANSPORT_OBS_DIM = 59
    _TOOL_TRANSPORT_SLICE = slice(41, 44)

    def __init__(self):
        self.observations = list()

    def log(self, obs):
        agent_obs, vel_obs = None, None
        if isinstance(obs, dict):
            
            # print("robot0_eef_pos.get", obs.get("robot0_eef_pos"))
            # print("agent_pos.get", obs.get("agent_pos"))
            if obs.get("agent_pos") is not None:
                agent_obs = obs.get("agent_pos")
            elif obs.get("robot0_eef_pos") is not None:
                agent_obs = obs.get("robot0_eef_pos")
            elif obs.get("pos_agent") is not None and obs.get("vel_agent") is not None:
                agent_obs = obs.get("pos_agent")
                vel_obs = obs.get("vel_agent")
            else:
                agent_obs = None   
            
            obs_dict = {"pos_agent": agent_obs, "vel_agent": vel_obs}
            self.observations.append(obs_dict)
            return
        elif isinstance(obs, np.ndarray):
            obs_len = len(obs)
            obs_dict = None

            # Infer the task type based on the length of the observation array.
            if obs_len == self._PUSHT_OBS_DIM:
                obs_dict = {"pos_agent": obs[self._PUSHT_POS_SLICE]}
            elif obs_len == self._LIFT_OBS_DIM:
                obs_dict = {"pos_agent": obs[self._LIFT_POS_SLICE]}
            elif obs_len == self._CAN_OBS_DIM:
                obs_dict = {"pos_agent": obs[self._CAN_POS_SLICE]}
            elif obs_len == self._TOOL_TRANSPORT_OBS_DIM:
                obs_dict = {"pos_agent": obs[self._TOOL_TRANSPORT_SLICE]}
            else:
                raise ValueError(
                    f"Unsupported observation array length: {obs_len}. "
                    f"Expected {self._PUSHT_OBS_DIM} or {self._LIFT_OBS_DIM} or {self._CAN_POS_SLICE} or {self._TOOL_TRANSPORT_OBS_DIM}."
                )
            # Append the successfully created dictionary to our list.
            self.observations.append(obs_dict)
        else:
            raise TypeError(f"Unsupported info type: {type(obs)}")

    def get_all(self):
        return self.observations

    def clear(self):
        self.observations = []

    def to_dict_of_lists(self):
        output = defaultdict(list)
        for item in self.infos:
            for k, v in item.items():
                output[k].append(v)
        return dict(output)


# TODO: not tested
class MultiLogger:
    def __init__(self, loggers):
        self.loggers = loggers  # List[Logger]

    def get_all(self):
        all_info = []
        for logger in self.loggers:
            all_info.extend(logger.get_all())
        return all_info

    def to_dict_of_lists(self):
        merged = defaultdict(list)
        for logger in self.loggers:
            for item in logger.get_all():
                for k, v in item.items():
                    merged[k].append(v)
        return dict(merged)

    def clear(self):
        for logger in self.loggers:
            logger.clear()


def compute_motion_metrics(observations, dt=1):
    result = {}

    # Step 1: position
    pos_seq = np.array([obs["pos_agent"] for obs in observations])  # (T, D)

    # Step 2: velocity
    vel_seq = np.diff(pos_seq, axis=0) / dt  # (T-1, D)
    vel_norm = np.linalg.norm(vel_seq, axis=1)  # (T-1,)
    result["mean_vel"] = vel_norm.mean()
    result["max_vel"] = vel_norm.max()
    result["vel"] = vel_norm

    # Step 3: acceleration
    acc_seq = np.diff(vel_seq, axis=0) / dt  # (T-2, 2)
    acc_norm = np.linalg.norm(acc_seq, axis=1)  # (T-2,)
    result["mean_accel"] = acc_norm.mean()
    result["max_accel"] = acc_norm.max()
    result["accel"] = acc_norm

    # Step 4: jerkness
    jerk_seq = np.diff(acc_seq, axis=0) / dt  # (T-3, D)
    jerk_norm = np.linalg.norm(jerk_seq, axis=1)  # (T-3,)
    result["mean_jerk"] = jerk_norm.mean()
    result["max_jerk"] = jerk_norm.max()
    result["jerk"] = jerk_norm

    return result
