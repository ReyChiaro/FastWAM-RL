import numpy as np
from dataclasses import dataclass


@dataclass
class Observation:

    cameras: dict[str, np.ndarray]
    proprio_states: np.ndarray

    is_success: bool
    reward: float


class BaseEnvAdapter:

    def reset(self):
        pass

    def step(self) -> Observation:
        pass

    def close(self):
        pass
