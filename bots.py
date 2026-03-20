import math
import os
import random
from typing import Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ScriptedBot:
    """Rule-based bot brain. Returns (dx, dy, dash, do_break)."""

    def __init__(self):
        self.aggression = random.uniform(0.5, 1.0)
        self.jitter_angle = 0.0
        self.jitter_timer = 0.0
        self.wander_angle = random.uniform(0, 2 * math.pi)

    def get_action(self, observation: dict) -> Tuple[float, float, bool, bool]:
        if not observation or not observation.get("nearest"):
            dx, dy = self._wander()
            return dx, dy, False, False

        self.jitter_timer -= 0.05
        if self.jitter_timer <= 0:
            self.jitter_angle = random.uniform(-0.4, 0.4)
            self.jitter_timer = random.uniform(0.3, 0.8)

        if observation["is_it"]:
            return self._chase(observation)
        return self._flee(observation)

    def _wander(self) -> Tuple[float, float]:
        self.wander_angle += random.uniform(-0.2, 0.2)
        return (math.cos(self.wander_angle) * 0.4, math.sin(self.wander_angle) * 0.4)

    def _chase(self, obs: dict) -> Tuple[float, float, bool, bool]:
        targets = [n for n in obs["nearest"] if not n["is_it"]]
        if not targets:
            dx, dy = self._wander()
            return dx, dy, False, False

        t = targets[0]
        dx, dy = t["dx"], t["dy"]
        dist = math.sqrt(dx * dx + dy * dy) or 0.01

        pred_dx = dx + t["vx"] * 3.0 * self.aggression
        pred_dy = dy + t["vy"] * 3.0 * self.aggression
        pred_dist = math.sqrt(pred_dx * pred_dx + pred_dy * pred_dy) or 0.01

        base_angle = math.atan2(pred_dy, pred_dx)
        angle = base_angle + self.jitter_angle * 0.3
        mag = min(1.0, self.aggression + 0.2)
        move_dx = math.cos(angle) * mag
        move_dy = math.sin(angle) * mag

        dash = (
            dist < 80
            and obs.get("dash_cooldown", 1) <= 0
            and random.random() < 0.3
        )

        do_break = (
            obs.get("break_cooldown", 1) <= 0
            and self._walls_ahead(obs, move_dx, move_dy)
            and random.random() < 0.6
        )
        return move_dx, move_dy, dash, do_break

    def _flee(self, obs: dict) -> Tuple[float, float, bool, bool]:
        threats = [n for n in obs["nearest"] if n["is_it"]]
        if not threats:
            dx, dy = self._wander()
            return dx, dy, False, False

        threat = threats[0]
        dx, dy = threat["dx"], threat["dy"]
        dist = math.sqrt(dx * dx + dy * dy) or 0.01

        flee_x = -dx / dist
        flee_y = -dy / dist

        avoid_x, avoid_y = self._avoid_walls(obs)

        move_x = flee_x * 0.7 + avoid_x * 0.3
        move_y = flee_y * 0.7 + avoid_y * 0.3

        angle = math.atan2(move_y, move_x) + self.jitter_angle
        mag = min(1.0, math.sqrt(move_x ** 2 + move_y ** 2) + 0.1)
        move_dx = math.cos(angle) * mag
        move_dy = math.sin(angle) * mag

        dash = (
            dist < 60
            and obs.get("dash_cooldown", 1) <= 0
            and random.random() < 0.4
        )

        do_break = (
            obs.get("break_cooldown", 1) <= 0
            and dist < 100
            and self._walls_ahead(obs, move_dx, move_dy)
            and random.random() < 0.5
        )
        return move_dx, move_dy, dash, do_break

    @staticmethod
    def _walls_ahead(obs: dict, move_dx: float, move_dy: float) -> bool:
        walls = obs.get("walls_nearby", [])
        if not walls:
            return False
        sx, sy = obs["self_x"], obs["self_y"]
        move_mag = math.sqrt(move_dx * move_dx + move_dy * move_dy)
        if move_mag < 0.1:
            return False
        ndx, ndy = move_dx / move_mag, move_dy / move_mag
        for wx, wy, ww, wh in walls:
            cx = wx + ww * 0.5
            cy = wy + wh * 0.5
            to_x = cx - sx
            to_y = cy - sy
            d = math.sqrt(to_x * to_x + to_y * to_y)
            if d > 56:
                continue
            if ndx * to_x + ndy * to_y > 0:
                return True
        return False

    @staticmethod
    def _avoid_walls(obs: dict) -> Tuple[float, float]:
        walls = obs.get("walls_nearby", [])
        if not walls:
            return (0.0, 0.0)
        sx, sy = obs["self_x"], obs["self_y"]
        ax, ay = 0.0, 0.0
        for wx, wy, ww, wh in walls:
            cx = wx + ww * 0.5
            cy = wy + wh * 0.5
            ddx = sx - cx
            ddy = sy - cy
            dist = math.sqrt(ddx * ddx + ddy * ddy) or 0.01
            if dist < 64:
                force = (64 - dist) / 64
                ax += ddx / dist * force
                ay += ddy / dist * force
        mag = math.sqrt(ax * ax + ay * ay)
        if mag > 0:
            ax /= mag
            ay /= mag
        return (ax, ay)


class RLBot:
    """Bot driven by a trained PyTorch policy. Falls back to ScriptedBot if no model."""

    def __init__(self, policy_path: str = None):
        self._fallback = ScriptedBot()
        self.policy = None

        if not TORCH_AVAILABLE:
            return

        path = policy_path or os.path.join(os.path.dirname(__file__), "best_policy.pt")
        if os.path.exists(path):
            try:
                from training import PolicyNetwork, encode_observation
                self._encode = encode_observation
                self.policy = PolicyNetwork()
                self.policy.load_state_dict(torch.load(path, weights_only=True))
                self.policy.eval()
            except Exception:
                self.policy = None

    def get_action(self, observation: dict) -> Tuple[float, float, bool, bool]:
        if self.policy is None:
            return self._fallback.get_action(observation)

        import torch as th
        obs_vec = self._encode(observation)
        obs_t = th.from_numpy(obs_vec).unsqueeze(0)
        with th.no_grad():
            move_mu, log_std, dash_logit, break_logit, _ = self.policy(obs_t)
        dx = float(th.tanh(move_mu[0, 0]))
        dy = float(th.tanh(move_mu[0, 1]))
        dash = float(th.sigmoid(dash_logit[0, 0])) > 0.5
        do_break = float(th.sigmoid(break_logit[0, 0])) > 0.5
        return dx, dy, dash, do_break
