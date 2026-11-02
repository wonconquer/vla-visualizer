"""LeRobot 0.6.1 environment boundary. Only reset uses an official initial state."""

import json
import os
from pathlib import Path

from vla_visualizer.config import path_setting
from vla_visualizer.simulation.compatibility import SUITE, TASK_ID, TASK_NAME, verify_assets, assets_dir


class LiberoEnvironment:
    def __init__(self):
        self.env = None

    def reset(self, seed: int):
        # Set project-owned config before importing LIBERO (its default config prompts).
        import importlib.util
        verify_assets()
        spec = importlib.util.find_spec("libero")
        package = Path(spec.origin).parent / "libero"
        config = path_setting("VLA_LIBERO_CONFIG", ".cache/libero/config")
        config.mkdir(parents=True, exist_ok=True)
        os.environ["LIBERO_CONFIG_PATH"] = str(config)
        # JSON is valid YAML; no user-home writes or implicit asset downloads.
        (config / "config.yaml").write_text(json.dumps({
            "assets": str(assets_dir()), "bddl_files": str(package / "bddl_files"),
            "init_states": str(package / "init_files"), "datasets": str(config / "unused_datasets"),
        }))
        import libero.libero as libero_runtime
        # hf-libero 0.1.4's get_assets_path() ignores config.yaml's assets key.
        # Its documented local cache slot must be seeded before environment imports;
        # otherwise it attempts a full mutable-main download. No installed file changes.
        libero_runtime._assets_path_cache = str(assets_dir())
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv

        suite = benchmark.get_benchmark_dict()[SUITE](task_order_index=0)
        if suite.get_task(TASK_ID).name != TASK_NAME:
            raise RuntimeError("Installed LIBERO task ID/order differs from the reviewed source.")
        self.env = LiberoEnv(
            task_suite=suite, task_id=TASK_ID, task_suite_name=SUITE,
            obs_type="pixels_agent_pos", observation_height=256, observation_width=256,
            init_states=True, episode_index=0, n_envs=1, num_steps_wait=10,
            control_freq=20, control_mode="relative", hard_reset=True,
        )
        observation, info = self.env.reset(seed=seed)
        controller = self.env._env.robots[0].controller
        import numpy as np
        if not controller.use_delta or not np.allclose(controller.output_max, [0.05]*3 + [0.5]*3):
            raise RuntimeError("OSC_POSE relative action scaling differs from the reviewed source.")
        self.controller_metadata = {
            "class": type(controller).__name__, "use_delta": bool(controller.use_delta),
            "input_min": np.asarray(controller.input_min).tolist(),
            "input_max": np.asarray(controller.input_max).tolist(),
            "output_min": np.asarray(controller.output_min).tolist(),
            "output_max": np.asarray(controller.output_max).tolist(),
            "physics_timestep_seconds": float(self.env._env.sim.model.opt.timestep),
            "control_frequency_hz": 20, "warmup_control_steps": 10,
            "official_initial_state_index": 0,
            "asset_lookup": "project manifest / explicit hf-libero runtime cache path",
        }
        return observation, info

    def step(self, action):
        return self.env.step(action)

    def simulator_time(self) -> float:
        return float(self.env._env.sim.data.time)

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
