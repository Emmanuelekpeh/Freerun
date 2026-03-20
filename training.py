"""
RL training pipeline for Freerun bots.
PPO on CPU with population-based training and ELO ranking.

Run directly:  python training.py
Metrics written to training_metrics.json, best policy to best_policy.pt
"""

import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np

from engine import (
    GameEngine, TICK_DT, MAX_SPEED, TAG_COOLDOWN, DASH_COOLDOWN,
    BREAK_COOLDOWN, WORLD_BOUNDARY, TILE_SIZE, NUM_RAYCASTS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Hyperparameters ──────────────────────────────────────────────────
OBS_DIM = 6 + 5 * 5 + NUM_RAYCASTS          # 39
HIDDEN_DIM = 128
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64

EPISODE_LENGTH = 600                          # 30 seconds of game time at 20Hz
NUM_MATCH_AGENTS = 6
INITIAL_POOL_SIZE = 8
ELO_K = 32
EVOLVE_INTERVAL = 20
SAVE_INTERVAL = 5

# Reward shaping
R_SURVIVE_TICK = 0.01
R_IT_TICK = -0.01
R_TAG_MADE = 2.0
R_GOT_TAGGED = -1.0
R_DASH_PENALTY = -0.03
R_BREAK_PENALTY = -0.02

# Tier boundaries
TIERS = {
    "Bronze":  (-9999, 950),
    "Silver":  (950, 1050),
    "Gold":    (1050, 1150),
    "Diamond": (1150, 9999),
}


# ─── Observation Encoder ──────────────────────────────────────────────

def encode_observation(obs: dict) -> np.ndarray:
    feats = [
        obs["self_vx"] / MAX_SPEED,
        obs["self_vy"] / MAX_SPEED,
        1.0 if obs["is_it"] else 0.0,
        max(0, obs["tag_cooldown"]) / TAG_COOLDOWN,
        max(0, obs.get("dash_cooldown", 0)) / DASH_COOLDOWN,
        max(0, obs.get("break_cooldown", 0)) / BREAK_COOLDOWN,
    ]

    nearest = obs.get("nearest", [])
    for i in range(5):
        if i < len(nearest):
            n = nearest[i]
            feats.extend([
                n["dx"] / WORLD_BOUNDARY,
                n["dy"] / WORLD_BOUNDARY,
                n["vx"] / MAX_SPEED,
                n["vy"] / MAX_SPEED,
                1.0 if n["is_it"] else 0.0,
            ])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    raycasts = obs.get("raycasts", [96.0] * NUM_RAYCASTS)
    for r in raycasts:
        feats.append(r / 96.0)

    return np.array(feats, dtype=np.float32)


# ─── Policy Network ──────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.move_mean = nn.Linear(hidden, 2)
        self.move_log_std = nn.Parameter(torch.zeros(2))
        self.dash_logit = nn.Linear(hidden, 1)
        self.break_logit = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        move_mu = torch.tanh(self.move_mean(h))
        dash_logit = self.dash_logit(h)
        break_logit = self.break_logit(h)
        value = self.value_head(h)
        return move_mu, self.move_log_std, dash_logit, break_logit, value


# ─── Experience Buffer ────────────────────────────────────────────────

class ExperienceBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.move_actions: List[np.ndarray] = []
        self.dash_actions: List[float] = []
        self.break_actions: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.move_log_probs: List[float] = []
        self.dash_log_probs: List[float] = []
        self.break_log_probs: List[float] = []
        self.dones: List[float] = []

    def store(self, obs, move_act, dash_act, break_act, reward, value,
              move_lp, dash_lp, break_lp, done):
        self.obs.append(obs)
        self.move_actions.append(move_act)
        self.dash_actions.append(dash_act)
        self.break_actions.append(break_act)
        self.rewards.append(reward)
        self.values.append(value)
        self.move_log_probs.append(move_lp)
        self.dash_log_probs.append(dash_lp)
        self.break_log_probs.append(break_lp)
        self.dones.append(1.0 if done else 0.0)

    def __len__(self):
        return len(self.obs)

    def compute_gae(self, last_value: float):
        values = self.values + [last_value]
        advantages = []
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + GAMMA * values[t + 1] * (1.0 - self.dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1.0 - self.dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, self.values)]
        return advantages, returns

    def to_tensors(self, advantages, returns):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32),
            torch.tensor(np.array(self.move_actions), dtype=torch.float32),
            torch.tensor(self.dash_actions, dtype=torch.float32),
            torch.tensor(self.break_actions, dtype=torch.float32),
            torch.tensor(self.move_log_probs, dtype=torch.float32),
            torch.tensor(self.dash_log_probs, dtype=torch.float32),
            torch.tensor(self.break_log_probs, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
        )

    def clear(self):
        self.__init__()


# ─── RLAgent: wraps policy + ELO + optimizer ─────────────────────────

class RLAgent:
    def __init__(self, policy: PolicyNetwork = None, elo: float = 1000.0):
        self.policy = policy or PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.elo = elo
        self.games_played = 0
        self.total_reward = 0.0
        self.tags_made = 0
        self.times_tagged = 0

    def act(self, obs_vec: np.ndarray):
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0)
        with torch.no_grad():
            move_mu, log_std, dash_logit, break_logit, value = self.policy(obs_t)

        move_std = log_std.exp()
        move_dist = D.Normal(move_mu, move_std)
        move_action = move_dist.sample().clamp(-1, 1)
        move_lp = move_dist.log_prob(move_action).sum(-1).item()

        dash_dist = D.Bernoulli(logits=dash_logit.squeeze(-1))
        dash_action = dash_dist.sample()
        dash_lp = dash_dist.log_prob(dash_action).item()

        break_dist = D.Bernoulli(logits=break_logit.squeeze(-1))
        break_action = break_dist.sample()
        break_lp = break_dist.log_prob(break_action).item()

        return {
            "dx": move_action[0, 0].item(),
            "dy": move_action[0, 1].item(),
            "dash": dash_action.item() > 0.5,
            "do_break": break_action.item() > 0.5,
            "move_act": move_action[0].numpy(),
            "dash_act": dash_action.item(),
            "break_act": break_action.item(),
            "move_lp": move_lp,
            "dash_lp": dash_lp,
            "break_lp": break_lp,
            "value": value.item(),
        }

    def get_value(self, obs_vec: np.ndarray) -> float:
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0)
        with torch.no_grad():
            _, _, _, _, value = self.policy(obs_t)
        return value.item()

    @property
    def tier(self) -> str:
        for name, (lo, hi) in TIERS.items():
            if lo <= self.elo < hi:
                return name
        return "Bronze"

    def clone(self, elo_offset: float = -50) -> "RLAgent":
        child = RLAgent(elo=max(800, self.elo + elo_offset))
        child.policy.load_state_dict(self.policy.state_dict())
        with torch.no_grad():
            for param in child.policy.parameters():
                param.add_(torch.randn_like(param) * 0.02)
        child.optimizer = optim.Adam(child.policy.parameters(), lr=LR)
        return child


# ─── PPO Update ───────────────────────────────────────────────────────

def ppo_update(agent: RLAgent, buffer: ExperienceBuffer):
    if len(buffer) < 16:
        return

    last_obs = buffer.obs[-1]
    last_val = agent.get_value(last_obs)
    advantages, returns = buffer.compute_gae(last_val)
    (obs_t, moves_t, dashes_t, breaks_t,
     old_move_lp, old_dash_lp, old_break_lp, adv_t, ret_t) = \
        buffer.to_tensors(advantages, returns)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    old_total_lp = old_move_lp + old_dash_lp + old_break_lp
    n = len(buffer)

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(n)
        for start in range(0, n, MINIBATCH_SIZE):
            end = min(start + MINIBATCH_SIZE, n)
            idx = indices[start:end]

            b_obs = obs_t[idx]
            b_moves = moves_t[idx]
            b_dashes = dashes_t[idx]
            b_breaks = breaks_t[idx]
            b_old_lp = old_total_lp[idx]
            b_adv = adv_t[idx]
            b_ret = ret_t[idx]

            move_mu, log_std, dash_logit, break_logit, values = agent.policy(b_obs)
            move_std = log_std.exp()
            move_dist = D.Normal(move_mu, move_std)
            new_move_lp = move_dist.log_prob(b_moves).sum(-1)
            move_entropy = move_dist.entropy().sum(-1).mean()

            dash_dist = D.Bernoulli(logits=dash_logit.squeeze(-1))
            new_dash_lp = dash_dist.log_prob(b_dashes)
            dash_entropy = dash_dist.entropy().mean()

            break_dist = D.Bernoulli(logits=break_logit.squeeze(-1))
            new_break_lp = break_dist.log_prob(b_breaks)
            break_entropy = break_dist.entropy().mean()

            new_total_lp = new_move_lp + new_dash_lp + new_break_lp
            ratio = (new_total_lp - b_old_lp).exp()

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((values.squeeze(-1) - b_ret) ** 2).mean()
            entropy = move_entropy + dash_entropy + break_entropy
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.policy.parameters(), MAX_GRAD_NORM)
            agent.optimizer.step()


# ─── Population Manager ──────────────────────────────────────────────

class Population:
    def __init__(self, size: int = INITIAL_POOL_SIZE):
        self.agents: List[RLAgent] = [RLAgent() for _ in range(size)]

    def sample_match(self, n: int = NUM_MATCH_AGENTS) -> List[RLAgent]:
        return random.sample(self.agents, min(n, len(self.agents)))

    def update_elo(self, winner: RLAgent, loser: RLAgent):
        ea = 1.0 / (1.0 + 10.0 ** ((loser.elo - winner.elo) / 400.0))
        winner.elo += ELO_K * (1.0 - ea)
        loser.elo -= ELO_K * ea

    def evolve(self):
        self.agents.sort(key=lambda a: a.elo, reverse=True)
        half = len(self.agents) // 2
        survivors = self.agents[:half]
        children = [a.clone() for a in survivors]
        self.agents = survivors + children

    def tier_counts(self) -> dict:
        counts = {name: 0 for name in TIERS}
        for a in self.agents:
            counts[a.tier] += 1
        return counts

    def best(self) -> RLAgent:
        return max(self.agents, key=lambda a: a.elo)


# ─── Training Session ────────────────────────────────────────────────

class TrainingSession:
    def __init__(self):
        self.population = Population()
        self.episode_count = 0
        self.start_time = time.time()
        self.metrics_path = os.path.join(SCRIPT_DIR, "training_metrics.json")
        self.policy_path = os.path.join(SCRIPT_DIR, "best_policy.pt")
        self.running = True
        self._reward_history: List[float] = []
        self._elo_history: List[dict] = []

    def run_episode(self) -> float:
        engine = GameEngine()
        agents = self.population.sample_match(NUM_MATCH_AGENTS)

        agent_map: Dict[str, RLAgent] = {}
        buffers: Dict[str, ExperienceBuffer] = {}
        for i, agent in enumerate(agents):
            pid = f"t{i}"
            engine.add_player(pid, f"A{i}", is_bot=True)
            agent_map[pid] = agent
            buffers[pid] = ExperienceBuffer()

        ep_rewards: Dict[str, float] = {pid: 0.0 for pid in agent_map}

        for tick in range(EPISODE_LENGTH):
            action_cache: Dict[str, dict] = {}

            for pid, agent in agent_map.items():
                obs = engine.get_observation(pid)
                if obs is None:
                    continue
                obs_vec = encode_observation(obs)
                info = agent.act(obs_vec)
                action_cache[pid] = {"obs_vec": obs_vec, **info}

                engine.set_input(pid, info["dx"], info["dy"])
                if info["dash"]:
                    engine.trigger_dash(pid)
                if info["do_break"]:
                    engine.trigger_break(pid)

            engine.tick()

            done = (tick == EPISODE_LENGTH - 1)

            for pid, agent in agent_map.items():
                if pid not in action_cache:
                    continue
                ac = action_cache[pid]
                p = engine.players.get(pid)
                reward = 0.0

                if p is not None:
                    reward = R_IT_TICK if p.is_it else R_SURVIVE_TICK

                if ac["dash"]:
                    reward += R_DASH_PENALTY
                if ac["do_break"]:
                    reward += R_BREAK_PENALTY

                for ev in engine.events:
                    if ev["type"] == "tag":
                        if ev["tagger_id"] == pid:
                            reward += R_TAG_MADE
                            agent.tags_made += 1
                            tagged_agent = agent_map.get(ev["tagged_id"])
                            if tagged_agent:
                                self.population.update_elo(agent, tagged_agent)
                        elif ev["tagged_id"] == pid:
                            reward += R_GOT_TAGGED
                            agent.times_tagged += 1

                ep_rewards[pid] += reward

                buffers[pid].store(
                    ac["obs_vec"], ac["move_act"], ac["dash_act"], ac["break_act"],
                    reward, ac["value"], ac["move_lp"], ac["dash_lp"], ac["break_lp"],
                    done,
                )

        for pid, agent in agent_map.items():
            ppo_update(agent, buffers[pid])
            agent.games_played += 1
            agent.total_reward += ep_rewards[pid]

        self.episode_count += 1
        avg_reward = sum(ep_rewards.values()) / max(1, len(ep_rewards))
        self._reward_history.append(avg_reward)

        best = self.population.best()
        self._elo_history.append({
            "episode": self.episode_count,
            "avg_elo": sum(a.elo for a in self.population.agents) / len(self.population.agents),
            "top_elo": best.elo,
            "tier_counts": self.population.tier_counts(),
        })

        if self.episode_count % EVOLVE_INTERVAL == 0:
            self.population.evolve()

        if self.episode_count % SAVE_INTERVAL == 0:
            self._save()

        return avg_reward

    def _save(self):
        best = self.population.best()
        torch.save(best.policy.state_dict(), self.policy_path)

        recent = self._reward_history[-100:]
        metrics = {
            "running": self.running,
            "episode": self.episode_count,
            "elapsed_s": round(time.time() - self.start_time, 1),
            "avg_reward_recent": round(sum(recent) / max(1, len(recent)), 4),
            "reward_history": [round(r, 4) for r in self._reward_history[-200:]],
            "elo_snapshots": self._elo_history[-50:],
            "population": [
                {
                    "elo": round(a.elo, 1),
                    "tier": a.tier,
                    "games": a.games_played,
                    "tags": a.tags_made,
                }
                for a in sorted(self.population.agents, key=lambda a: a.elo, reverse=True)
            ],
        }
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def run(self, max_episodes: int = 10000):
        print(f"Training started: pool={INITIAL_POOL_SIZE}, match={NUM_MATCH_AGENTS}, "
              f"ep_len={EPISODE_LENGTH}, lr={LR}")
        try:
            while self.running and self.episode_count < max_episodes:
                avg_r = self.run_episode()
                if self.episode_count % 10 == 0:
                    best = self.population.best()
                    tiers = self.population.tier_counts()
                    print(
                        f"[Ep {self.episode_count:>5}] "
                        f"avg_r={avg_r:+.3f}  "
                        f"top_elo={best.elo:.0f} ({best.tier})  "
                        f"tiers: {tiers}"
                    )
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        finally:
            self.running = False
            self._save()
            print(f"Saved best policy to {self.policy_path}")
            print(f"Saved metrics to {self.metrics_path}")


if __name__ == "__main__":
    session = TrainingSession()
    session.run()
