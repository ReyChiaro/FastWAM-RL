from simulation.adapters.base import BaseEnvAdapter, Observation

try:
    from libero.libero import benchmark, get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore
except ModuleNotFoundError as exc:
    if exc.name in {"robosuite", "bddl"}:
        raise ModuleNotFoundError(f"Missing dependency '{exc.name}'. Activate the LIBERO runtime first.") from exc
    raise


class LiberoEnvAdapter(BaseEnvAdapter):
    r"""
    Wrapper of libero environment.
    """

    def reset(self):
        return super().reset()

    def step(self) -> Observation:
        return super().step()

    def close(self):
        return super().close()
