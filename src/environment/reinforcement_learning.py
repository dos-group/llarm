from gymnasium import Env
from gymnasium.envs.registration import register
import numpy as np
from gymnasium import spaces

class ReinforcementLearning(Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, max_steps=10000, seed=None):
        super().__init__()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._step_count = 0
        self._rng = np.random.default_rng(seed)
        self._state = None

    def seed(self, seed=None):
        # Optional (älterer Stil); Gymnasium nutzt üblicherweise reset(seed=...)
        self._rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._state = self._rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
        info = {}
        return self._state.copy(), info

    def step(self, action):
        assert self.action_space.contains(action), "Ungültige Aktion"

        # einfache Zustandsdynamik: kleine Zufallsbewegung
        noise = self._rng.normal(loc=0.0, scale=0.05, size=(3,))
        # Aktion beeinflusst die Richtung der Verschiebung der ersten Komponente
        drift = 0.02 if action == 1 else -0.02
        self._state = np.clip(
            self._state + noise + np.array([drift, 0.0, 0.0]), -1.0, 1.0
        ).astype(np.float32)

        self._step_count += 1
        reward = 1.0
        terminated = False                   # kein natürliches Terminalkriterium
        truncated = self._step_count >= self.max_steps  # Zeitlimit

        info = {}
        if self.render_mode == "human":
            self.render()

        return self._state.copy(), reward, terminated, truncated, info

    def render(self):
        # Minimal: einfach den Zustand ausgeben
        print(f"[SmokeEnv] step={self._step_count} state={self._state}")

    def close(self):
        pass

environment = ReinforcementLearning()

obs, info = environment.reset()

done = False
while False and not done:
    action = environment.action_space.sample()
    obs, reward, terminated, truncated, info = environment.step(action)
    environment.render()

    done = terminated or truncated

verbs = [
    "Lift",
    "Pick",
    "Take",
    "Grab",
    "Raise",
    "Hoist",
    "Fetch",
    "Collect",
    "Retrieve",
    "Grasp",
    "Pick up",
    "Lift up",
    "Grab up",
    "Raise up",
    "Snatch",
    "Seize",
    "Get",
    "Acquire",
    "Carry",
    "Hold",
    "Pick the",
    "Pick up",
    "Grab the",
    "Lift the",
    "Raise the",
]

items = [
    "block",
    "cube",
    "item",
    "object",
    "",
]

tags = [
    "red",
    "green",
    "blue",
]

for verb in verbs:
    for item in items:
        for tag in tags:
            from functools import reduce
            a = reduce(
                lambda carry, item: " ".join([carry, item]).strip(),
                [verb, item, tag],
                "",
            )
            print(a)

exit()

from .base import Base
from threading import Thread
from ..utility import PyBulletContext

with PyBulletContext(True):
    base = Base()

    def handle_input():
        while True:
            message = input("> ")

            if message == "exit":
                break
            elif message == "break":
                breakpoint()
                continue
            elif message == "reset":
                base.reset()
                continue
            else:
                base.evaluate(message)


    thread = Thread(target=handle_input)
    thread.start()

    while thread.is_alive():
        base.update()
