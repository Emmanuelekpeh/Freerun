"""Lightweight online-learning brain for hybrid bots.

Pure numpy -- no PyTorch/TensorFlow required.
Learns from live gameplay via reward-weighted regression.
Shared singleton across all game rooms so every public match
feeds experience into the same evolving policy.
"""

import math
import os
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "brain.npz")
META_FILE = os.path.join(WEIGHTS_DIR, "brain_meta.npz")

OBS_DIM = 34
ACT_DIM = 4           # dx, dy, dash_prob, break_prob
HIDDEN1 = 64
HIDDEN2 = 32

BUFFER_CAP = 10_000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_EVERY = 200      # ticks between gradient steps
SAVE_EVERY = 50        # training steps between disk saves

BLEND_START = 0.80     # alpha starts here (mostly scripted)
BLEND_MIN = 0.30       # alpha floor (scripted always has influence)
BLEND_DECAY = 0.0002   # per training step


# ─── Numpy Neural Net ────────────────────────────────────────────────

class NumpyNet:
    """2-hidden-layer feedforward net.  ReLU activations, no dependencies."""

    def __init__(self):
        self.W1 = (np.random.randn(OBS_DIM, HIDDEN1).astype(np.float32)
                    * np.sqrt(2.0 / OBS_DIM))
        self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
        self.W2 = (np.random.randn(HIDDEN1, HIDDEN2).astype(np.float32)
                    * np.sqrt(2.0 / HIDDEN1))
        self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
        self.W3 = (np.random.randn(HIDDEN2, ACT_DIM).astype(np.float32)
                    * np.sqrt(2.0 / HIDDEN2))
        self.b3 = np.zeros(ACT_DIM, dtype=np.float32)

        self._x = None
        self._z1 = None
        self._h1 = None
        self._z2 = None
        self._h2 = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._h1 = np.maximum(0, self._z1)
        self._z2 = self._h1 @ self.W2 + self.b2
        self._h2 = np.maximum(0, self._z2)
        return self._h2 @ self.W3 + self.b3

    def backward(self, d_out: np.ndarray, lr: float):
        n = max(d_out.shape[0], 1)

        dW3 = self._h2.T @ d_out / n
        db3 = d_out.mean(axis=0)
        dh2 = d_out @ self.W3.T
        dz2 = dh2 * (self._z2 > 0).astype(np.float32)

        dW2 = self._h1.T @ dz2 / n
        db2 = dz2.mean(axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self._z1 > 0).astype(np.float32)

        dW1 = self._x.T @ dz1 / n
        db1 = dz1.mean(axis=0)

        for g in (dW1, db1, dW2, db2, dW3, db3):
            np.clip(g, -1.0, 1.0, out=g)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

    def save(self, path: str):
        np.savez_compressed(path,
                            W1=self.W1, b1=self.b1,
                            W2=self.W2, b2=self.b2,
                            W3=self.W3, b3=self.b3)

    def load(self, path: str):
        d = np.load(path)
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.W3, self.b3 = d["W3"], d["b3"]


# ─── Replay Buffer ───────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_CAP):
        self.cap = capacity
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.acts = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.rews = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.idx = 0

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float):
        self.obs[self.idx] = obs
        self.acts[self.idx] = act
        self.rews[self.idx] = rew
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n: int):
        ix = np.random.randint(0, self.size, size=min(n, self.size))
        return self.obs[ix], self.acts[ix], self.rews[ix]


# ─── Observation Encoder ─────────────────────────────────────────────

def encode_obs(obs: dict) -> np.ndarray:
    """Observation dict → fixed 34-dim float32 vector."""
    v = np.zeros(OBS_DIM, dtype=np.float32)
    v[0] = obs.get("self_x", 0) / 1000.0
    v[1] = obs.get("self_y", 0) / 1000.0
    v[2] = obs.get("self_vx", 0) / 10.0
    v[3] = obs.get("self_vy", 0) / 10.0
    v[4] = 1.0 if obs.get("is_it") else 0.0
    v[5] = obs.get("tag_cooldown", 0) / 2.0
    v[6] = obs.get("dash_cooldown", 0) / 5.0
    v[7] = obs.get("dash_charges", 0) / 3.0
    v[8] = obs.get("break_cooldown", 0) / 3.0
    v[9] = min(obs.get("explored_count", 0), 200) / 200.0

    nearest = obs.get("nearest", [])
    for i in range(4):
        base = 10 + i * 4
        if i < len(nearest):
            n = nearest[i]
            v[base]     = n.get("dx", 0) / 500.0
            v[base + 1] = n.get("dy", 0) / 500.0
            v[base + 2] = 1.0 if n.get("is_it") else 0.0
            v[base + 3] = n.get("dash_cd", 0) / 5.0

    rays = obs.get("raycasts", [])
    for i in range(8):
        v[26 + i] = (rays[i] / 96.0) if i < len(rays) else 1.0

    return v


# ─── Bot Brain (singleton) ───────────────────────────────────────────

class BotBrain:
    """Shared learning brain for all hybrid bots on the server."""

    _instance = None

    @classmethod
    def shared(cls) -> "BotBrain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.net = NumpyNet()
        self.buffer = ReplayBuffer()
        self.train_steps = 0
        self.tick_counter = 0

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        if os.path.exists(WEIGHTS_FILE):
            try:
                self.net.load(WEIGHTS_FILE)
                if os.path.exists(META_FILE):
                    m = np.load(META_FILE)
                    self.train_steps = int(m["train_steps"])
            except Exception:
                pass

    @property
    def alpha(self) -> float:
        return max(BLEND_MIN, BLEND_START - BLEND_DECAY * self.train_steps)

    # ── Inference ──────────────────────────────────────────────────

    def get_action(self, obs_vec: np.ndarray):
        """Returns (dx, dy, dash_prob, break_prob)."""
        raw = self.net.forward(obs_vec.reshape(1, -1))[0]
        dx = float(np.tanh(raw[0]))
        dy = float(np.tanh(raw[1]))
        dash_p = float(1.0 / (1.0 + math.exp(-max(-10, min(10, raw[2])))))
        brk_p  = float(1.0 / (1.0 + math.exp(-max(-10, min(10, raw[3])))))
        return dx, dy, dash_p, brk_p

    # ── Experience collection ──────────────────────────────────────

    def record(self, obs_vec: np.ndarray, act_vec: np.ndarray, reward: float):
        self.buffer.add(obs_vec, act_vec, reward)

    # ── Training step ──────────────────────────────────────────────

    def maybe_train(self):
        self.tick_counter += 1
        if self.tick_counter % TRAIN_EVERY != 0:
            return
        if self.buffer.size < BATCH_SIZE * 2:
            return

        obs_b, act_b, rew_b = self.buffer.sample(BATCH_SIZE)

        mean_r = rew_b.mean()
        std_r  = rew_b.std() + 1e-8
        advantages = (rew_b - mean_r) / std_r
        weights = np.maximum(advantages, 0).reshape(-1, 1)

        raw = self.net.forward(obs_b)
        pred_move = np.tanh(raw[:, :2])
        sig_raw = np.clip(raw[:, 2:], -10, 10)
        pred_db = 1.0 / (1.0 + np.exp(-sig_raw))
        pred = np.concatenate([pred_move, pred_db], axis=1)

        errors = pred - act_b
        d_full = 2.0 * errors * weights

        dtanh = (1.0 - pred_move ** 2).astype(np.float32)
        dsig  = (pred_db * (1.0 - pred_db)).astype(np.float32)
        d_out = np.concatenate([d_full[:, :2] * dtanh,
                                d_full[:, 2:] * dsig], axis=1)

        self.net.backward(d_out, LEARNING_RATE)
        self.train_steps += 1

        if self.train_steps % SAVE_EVERY == 0:
            self.save()

    # ── Persistence ────────────────────────────────────────────────

    def save(self):
        try:
            self.net.save(WEIGHTS_FILE)
            np.savez_compressed(META_FILE, train_steps=np.array(self.train_steps))
        except Exception:
            pass
