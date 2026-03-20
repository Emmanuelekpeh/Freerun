import math
import os
import random
from typing import Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


PERSONALITY_TYPES = ("hunter", "strategist", "trickster", "timid")


class ScriptedBot:
    """Rule-based bot brain with personality. Returns (dx, dy, dash, do_break)."""

    def __init__(self, personality: str = None):
        self.personality = personality or random.choice(PERSONALITY_TYPES)
        self.jitter_angle = 0.0
        self.jitter_timer = 0.0
        self.wander_angle = random.uniform(0, 2 * math.pi)
        self._locked_target_id = None
        self._target_lock_timer = 0.0
        self._patience_timer = 0.0

        if self.personality == "hunter":
            self.aggression = random.uniform(0.8, 1.0)
            self.tagback_aversion = 0.9
            self.target_switch_chance = 0.02
            self.dash_eagerness = 0.4
            self.patience = random.uniform(0.0, 1.5)
        elif self.personality == "strategist":
            self.aggression = random.uniform(0.5, 0.75)
            self.tagback_aversion = 0.95
            self.target_switch_chance = 0.08
            self.dash_eagerness = 0.15
            self.patience = random.uniform(2.0, 5.0)
        elif self.personality == "trickster":
            self.aggression = random.uniform(0.6, 0.85)
            self.tagback_aversion = 0.7
            self.target_switch_chance = 0.15
            self.dash_eagerness = 0.5
            self.patience = random.uniform(0.5, 2.0)
        else:  # timid
            self.aggression = random.uniform(0.3, 0.55)
            self.tagback_aversion = 0.4
            self.target_switch_chance = 0.05
            self.dash_eagerness = 0.2
            self.patience = random.uniform(3.0, 8.0)

    def get_action(self, observation: dict) -> Tuple[float, float, bool, bool]:
        if not observation or not observation.get("nearest"):
            dx, dy = self._wander()
            return dx, dy, False, False

        self.jitter_timer -= 0.05
        if self.jitter_timer <= 0:
            self.jitter_angle = random.uniform(-0.4, 0.4)
            self.jitter_timer = random.uniform(0.3, 0.8)

        self._target_lock_timer = max(0, self._target_lock_timer - 0.05)
        self._patience_timer = max(0, self._patience_timer - 0.05)

        orb_action = self._seek_orb(observation)
        if orb_action is not None:
            return orb_action

        if observation["is_it"]:
            return self._chase(observation)
        return self._flee(observation)

    def _wander(self) -> Tuple[float, float]:
        self.wander_angle += random.uniform(-0.2, 0.2)
        return (math.cos(self.wander_angle) * 0.4, math.sin(self.wander_angle) * 0.4)

    def _seek_orb(self, obs: dict):
        orbs = obs.get("nearby_orbs", [])
        if not orbs:
            return None
        if obs.get("dash_charges", 0) >= 3:
            return None
        if obs.get("break_cooldown", 1) > 0:
            return None

        threats = [n for n in obs.get("nearest", []) if n["is_it"]]
        if threats:
            closest_threat = min(
                math.sqrt(t["dx"] ** 2 + t["dy"] ** 2) for t in threats
            )
            if closest_threat < 120:
                return None

        best = min(orbs, key=lambda o: o[0] ** 2 + o[1] ** 2)
        odx, ody = best
        odist = math.sqrt(odx * odx + ody * ody)
        if odist > 140:
            return None

        if odist < 50:
            ndx = odx / (odist or 1)
            ndy = ody / (odist or 1)
            return ndx * 0.6, ndy * 0.6, False, True

        ndx = odx / odist
        ndy = ody / odist
        return ndx * 0.8, ndy * 0.8, False, False

    # ── Target scoring ────────────────────────────────────────────────

    def _score_target(self, obs: dict, target: dict) -> float:
        dx, dy = target["dx"], target["dy"]
        dist = math.sqrt(dx * dx + dy * dy) or 1.0

        score = 1000.0 / (dist + 10.0)

        if target.get("id") == obs.get("last_tagged_by"):
            score *= (1.0 - self.tagback_aversion)

        if target.get("dash_cd", 0) > 1.0:
            score *= 1.5
        if target.get("is_dashing"):
            score *= 0.3

        my_vx = obs.get("self_vx", 0)
        my_vy = obs.get("self_vy", 0)
        speed = math.sqrt(my_vx * my_vx + my_vy * my_vy)
        if speed > 0.5:
            dot = (my_vx * dx + my_vy * dy) / (speed * dist)
            score *= (1.0 + dot * 0.4)

        if target.get("id") == self._locked_target_id and self._target_lock_timer > 0:
            score *= 1.6

        return score

    def _pick_target(self, obs: dict) -> dict:
        candidates = [n for n in obs["nearest"] if not n["is_it"]]
        if not candidates:
            return None

        if self._target_lock_timer > 0 and random.random() > self.target_switch_chance:
            for c in candidates:
                if c.get("id") == self._locked_target_id:
                    return c

        scored = [(self._score_target(obs, c), c) for c in candidates]
        scored.sort(key=lambda x: -x[0])

        if self.personality == "trickster" and len(scored) > 1:
            if random.random() < 0.3:
                pick = scored[1][1]
            else:
                pick = scored[0][1]
        else:
            pick = scored[0][1]

        self._locked_target_id = pick.get("id")
        self._target_lock_timer = random.uniform(1.5, 4.0)
        return pick

    # ── Chase ─────────────────────────────────────────────────────────

    def _chase(self, obs: dict) -> Tuple[float, float, bool, bool]:
        target = self._pick_target(obs)
        if not target:
            dx, dy = self._wander()
            return dx, dy, False, False

        dx, dy = target["dx"], target["dy"]
        dist = math.sqrt(dx * dx + dy * dy) or 0.01

        if self._patience_timer > 0 and dist < 120:
            return self._stalk(obs, target)

        tvx, tvy = target.get("vx", 0), target.get("vy", 0)
        pred_factor = min(3.0, dist / 40.0) * self.aggression
        pred_dx = dx + tvx * pred_factor
        pred_dy = dy + tvy * pred_factor
        pred_dist = math.sqrt(pred_dx * pred_dx + pred_dy * pred_dy) or 0.01

        base_angle = math.atan2(pred_dy, pred_dx)

        if self.personality == "strategist" and dist > 80:
            flank = math.pi * 0.15 * (1 if random.random() > 0.5 else -1)
            base_angle += flank

        angle = base_angle + self.jitter_angle * 0.3
        mag = min(1.0, self.aggression + 0.2)
        move_dx = math.cos(angle) * mag
        move_dy = math.sin(angle) * mag

        can_dash = obs.get("dash_cooldown", 1) <= 0 or obs.get("dash_charges", 0) > 0
        dash = (
            dist < 90
            and can_dash
            and random.random() < self.dash_eagerness
            and not target.get("is_dashing")
        )

        if dist < 60 and self.patience > 0 and random.random() < 0.04:
            self._patience_timer = self.patience

        do_break = (
            obs.get("break_cooldown", 1) <= 0
            and self._walls_ahead(obs, move_dx, move_dy)
            and random.random() < 0.6
        )
        return move_dx, move_dy, dash, do_break

    def _stalk(self, obs: dict, target: dict) -> Tuple[float, float, bool, bool]:
        """Maintain distance, circling the target before striking."""
        dx, dy = target["dx"], target["dy"]
        dist = math.sqrt(dx * dx + dy * dy) or 0.01
        nx, ny = dx / dist, dy / dist

        perp_x, perp_y = -ny, nx
        if random.random() < 0.5:
            perp_x, perp_y = ny, -nx

        if dist < 50:
            self._patience_timer = 0
            move_dx = nx * 1.0
            move_dy = ny * 1.0
            can_d = obs.get("dash_cooldown", 1) <= 0 or obs.get("dash_charges", 0) > 0
            dash = can_d and random.random() < self.dash_eagerness
            return move_dx, move_dy, dash, False

        toward = 0.2 if dist > 100 else -0.1
        move_dx = perp_x * 0.7 + nx * toward
        move_dy = perp_y * 0.7 + ny * toward
        mag = math.sqrt(move_dx * move_dx + move_dy * move_dy) or 1
        move_dx = move_dx / mag * 0.6
        move_dy = move_dy / mag * 0.6
        return move_dx, move_dy, False, False

    # ── Flee ──────────────────────────────────────────────────────────

    def _flee(self, obs: dict) -> Tuple[float, float, bool, bool]:
        threats = [n for n in obs["nearest"] if n["is_it"]]
        if not threats:
            dx, dy = self._wander()
            return dx, dy, False, False

        flee_x, flee_y = 0.0, 0.0
        for t in threats:
            tdx, tdy = t["dx"], t["dy"]
            d = math.sqrt(tdx * tdx + tdy * tdy) or 1.0
            weight = 1.0 / (d + 1.0)
            flee_x -= (tdx / d) * weight
            flee_y -= (tdy / d) * weight

        mag = math.sqrt(flee_x * flee_x + flee_y * flee_y)
        if mag > 0:
            flee_x /= mag
            flee_y /= mag

        avoid_x, avoid_y = self._avoid_walls(obs)
        move_x = flee_x * 0.7 + avoid_x * 0.3
        move_y = flee_y * 0.7 + avoid_y * 0.3

        angle = math.atan2(move_y, move_x) + self.jitter_angle
        mag = min(1.0, math.sqrt(move_x ** 2 + move_y ** 2) + 0.1)
        move_dx = math.cos(angle) * mag
        move_dy = math.sin(angle) * mag

        closest = threats[0]
        cdist = math.sqrt(closest["dx"] ** 2 + closest["dy"] ** 2)
        can_dash_f = obs.get("dash_cooldown", 1) <= 0 or obs.get("dash_charges", 0) > 0
        dash = (
            cdist < 65
            and can_dash_f
            and random.random() < 0.45
        )

        do_break = (
            obs.get("break_cooldown", 1) <= 0
            and cdist < 100
            and self._walls_ahead(obs, move_dx, move_dy)
            and random.random() < 0.5
        )
        return move_dx, move_dy, dash, do_break

    # ── Helpers ───────────────────────────────────────────────────────

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
