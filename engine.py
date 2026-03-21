import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ─── World Constants ─────────────────────────────────────────────────
CHUNK_SIZE = 512
TILE_SIZE = 32
TILES_PER_CHUNK = CHUNK_SIZE // TILE_SIZE

# ─── Physics Constants ───────────────────────────────────────────────
PLAYER_RADIUS = 12
ACCELERATION = 0.5
FRICTION = 0.92
MAX_SPEED = 7.0
IT_SPEED_BONUS = 1.15

# ─── Dash Constants ──────────────────────────────────────────────────
DASH_SPEED = 20.0
DASH_COOLDOWN = 3.5
DASH_TICKS = 3
MAX_DASH_CHARGES = 3
DASH_ORB_CHANCE = 0.005

# ─── Break Constants ─────────────────────────────────────────────────
BREAK_RADIUS = 48
BREAK_COOLDOWN = 1.0

# ─── Tag Constants ───────────────────────────────────────────────────
TAG_RADIUS = 30
TAG_COOLDOWN = 3.0
WORLD_BOUNDARY = 1200

# ─── Ping Constants ──────────────────────────────────────────────────
PING_AFTER_TICKS = 160           # 8 seconds at 20Hz
PING_REPEAT_TICKS = 300          # 15 seconds between repeats

# ─── Tick ────────────────────────────────────────────────────────────
TICK_RATE = 20
TICK_DT = 1.0 / TICK_RATE

# ─── World Gen ───────────────────────────────────────────────────────
SPAWN_CLEAR_RADIUS = 120
NOISE_SCALE_PRIMARY = 0.09
NOISE_SCALE_DETAIL = 0.22
WALL_THRESHOLD = 0.08
DETAIL_THRESHOLD = 0.4
NUM_RAYCASTS = 8
RAYCAST_MAX = 96
RAYCAST_STEP = 16


# ─── Pure Python Value Noise ────────────────────────────────────────

def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, t):
    return a + t * (b - a)


def _hash_coord(x, y, seed):
    h = (x * 374761393 + y * 668265263 + seed * 1274126177) & 0xFFFFFFFF
    h = (((h ^ (h >> 13)) & 0xFFFFFFFF) * 1103515245 + 12345) & 0x7FFFFFFF
    return (h & 0xFFFF) / 32768.0 - 1.0


def noise2d(x, y, seed=0, octaves=3, persistence=0.5):
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for _ in range(octaves):
        sx = x * frequency
        sy = y * frequency
        ix = int(math.floor(sx))
        iy = int(math.floor(sy))
        fx = sx - ix
        fy = sy - iy
        u = _fade(fx)
        v = _fade(fy)
        n00 = _hash_coord(ix, iy, seed)
        n10 = _hash_coord(ix + 1, iy, seed)
        n01 = _hash_coord(ix, iy + 1, seed)
        n11 = _hash_coord(ix + 1, iy + 1, seed)
        nx0 = _lerp(n00, n10, u)
        nx1 = _lerp(n01, n11, u)
        total += _lerp(nx0, nx1, v) * amplitude
        max_val += amplitude
        amplitude *= persistence
        frequency *= 2.0
    return total / max_val if max_val else 0.0


# ─── Data Classes ────────────────────────────────────────────────────

@dataclass
class Player:
    id: str
    name: str
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    is_it: bool = False
    is_bot: bool = False
    input_dx: float = 0.0
    input_dy: float = 0.0
    tag_cooldown: float = 0.0
    dash_cooldown: float = 0.0
    dash_ticks_left: int = 0
    dash_charges: int = 0
    break_cooldown: float = 0.0
    it_ticks: int = 0
    last_tagged_by: str = ""
    tagged_by_timer: float = 0.0


class Chunk:
    __slots__ = ("cx", "cy", "wall_grid", "dash_tiles", "_walls_cache")

    def __init__(self, cx: int, cy: int):
        self.cx = cx
        self.cy = cy
        self.wall_grid: List[List[bool]] = [
            [False] * TILES_PER_CHUNK for _ in range(TILES_PER_CHUNK)
        ]
        self.dash_tiles: set = set()
        self._walls_cache: Optional[List[Tuple[float, float, float, float]]] = None

    def get_walls(self) -> List[Tuple[float, float, float, float]]:
        if self._walls_cache is None:
            walls = []
            base_x = self.cx * CHUNK_SIZE
            base_y = self.cy * CHUNK_SIZE
            for ty in range(TILES_PER_CHUNK):
                for tx in range(TILES_PER_CHUNK):
                    if self.wall_grid[ty][tx]:
                        walls.append((
                            base_x + tx * TILE_SIZE,
                            base_y + ty * TILE_SIZE,
                            TILE_SIZE,
                            TILE_SIZE,
                        ))
            self._walls_cache = walls
        return self._walls_cache

    def get_dash_tile_positions(self) -> List[Tuple[float, float, float, float]]:
        base_x = self.cx * CHUNK_SIZE
        base_y = self.cy * CHUNK_SIZE
        return [
            (base_x + tx * TILE_SIZE, base_y + ty * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            for (tx, ty) in self.dash_tiles
        ]


# ─── Game Engine ─────────────────────────────────────────────────────

INFESTATION_RESET_TICKS = 80      # 4 seconds between rounds


class GameEngine:
    def __init__(self, seed: int = None, game_mode: str = "classic"):
        self.seed = seed if seed is not None else random.randint(0, 999999)
        self.game_mode = game_mode
        self.players: Dict[str, Player] = {}
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.tick_count: int = 0
        self.events: List[dict] = []
        self.round_number: int = 1
        self.round_reset_countdown: int = -1

    # ── Player Management ────────────────────────────────────────────

    def add_player(self, player_id: str, name: str, is_bot: bool = False) -> Player:
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(30, 80)
        p = Player(
            id=player_id,
            name=name,
            x=math.cos(angle) * dist,
            y=math.sin(angle) * dist,
            is_bot=is_bot,
        )
        self.players[player_id] = p
        if not any(pl.is_it for pl in self.players.values()):
            p.is_it = True
            p.tag_cooldown = TAG_COOLDOWN
        return p

    def remove_player(self, player_id: str):
        if player_id not in self.players:
            return
        was_it = self.players[player_id].is_it
        del self.players[player_id]
        if self.game_mode == "classic":
            if was_it and self.players:
                new_it = random.choice(list(self.players.values()))
                new_it.is_it = True
                new_it.tag_cooldown = TAG_COOLDOWN
        else:
            if not any(p.is_it for p in self.players.values()) and self.players:
                p0 = random.choice(list(self.players.values()))
                p0.is_it = True
                p0.tag_cooldown = TAG_COOLDOWN

    def set_input(self, player_id: str, dx: float, dy: float):
        if player_id not in self.players:
            return
        mag = math.sqrt(dx * dx + dy * dy)
        if mag > 1.0:
            dx /= mag
            dy /= mag
        self.players[player_id].input_dx = dx
        self.players[player_id].input_dy = dy

    def trigger_dash(self, player_id: str):
        if player_id not in self.players:
            return
        p = self.players[player_id]
        if p.dash_ticks_left > 0:
            return
        has_charge = p.dash_charges > 0
        if not has_charge and p.dash_cooldown > 0:
            return
        ix, iy = p.input_dx, p.input_dy
        imag = math.sqrt(ix * ix + iy * iy)
        if imag < 0.1:
            vmag = math.sqrt(p.vx * p.vx + p.vy * p.vy)
            if vmag < 0.1:
                return
            ix, iy = p.vx / vmag, p.vy / vmag
        else:
            ix, iy = ix / imag, iy / imag
        p.vx = ix * DASH_SPEED
        p.vy = iy * DASH_SPEED
        p.dash_ticks_left = DASH_TICKS
        if has_charge:
            p.dash_charges -= 1
        else:
            p.dash_cooldown = DASH_COOLDOWN

    def trigger_break(self, player_id: str):
        if player_id not in self.players:
            return
        p = self.players[player_id]
        if p.break_cooldown > 0:
            return

        p.break_cooldown = BREAK_COOLDOWN
        destroyed = []
        orbs_collected = []

        center_tx = int(math.floor(p.x / TILE_SIZE))
        center_ty = int(math.floor(p.y / TILE_SIZE))
        scan = int(math.ceil(BREAK_RADIUS / TILE_SIZE)) + 1

        for dtx in range(-scan, scan + 1):
            for dty in range(-scan, scan + 1):
                wtx = center_tx + dtx
                wty = center_ty + dty
                wx = wtx * TILE_SIZE + TILE_SIZE * 0.5
                wy = wty * TILE_SIZE + TILE_SIZE * 0.5
                dx = wx - p.x
                dy = wy - p.y
                if dx * dx + dy * dy > BREAK_RADIUS * BREAK_RADIUS:
                    continue
                if not self._is_wall_at(wtx, wty):
                    continue

                cx = wtx // TILES_PER_CHUNK
                cy = wty // TILES_PER_CHUNK
                chunk = self.chunks.get((cx, cy))
                if chunk:
                    local_tx = wtx % TILES_PER_CHUNK
                    local_ty = wty % TILES_PER_CHUNK
                    chunk.wall_grid[local_ty][local_tx] = False
                    chunk._walls_cache = None
                    destroyed.append([wtx * TILE_SIZE, wty * TILE_SIZE, TILE_SIZE, TILE_SIZE])

                    if (local_tx, local_ty) in chunk.dash_tiles:
                        chunk.dash_tiles.discard((local_tx, local_ty))
                        orbs_collected.append([wtx * TILE_SIZE, wty * TILE_SIZE])

        if orbs_collected and p.dash_charges < MAX_DASH_CHARGES:
            gained = min(len(orbs_collected), MAX_DASH_CHARGES - p.dash_charges)
            p.dash_charges += gained
            self.events.append({
                "type": "dash_orb",
                "player_id": player_id,
                "player_name": p.name,
                "charges": p.dash_charges,
                "positions": orbs_collected,
            })

        if destroyed:
            self.events.append({
                "type": "break",
                "player_id": player_id,
                "walls": destroyed,
            })

    def set_game_mode(self, mode: str):
        self.game_mode = mode
        self.round_number = 1
        self.round_reset_countdown = -1
        for p in self.players.values():
            p.is_it = False
            p.it_ticks = 0
        if self.players:
            p0 = random.choice(list(self.players.values()))
            p0.is_it = True
            p0.tag_cooldown = TAG_COOLDOWN

    def _reset_infestation_round(self):
        self.round_number += 1
        self.round_reset_countdown = -1
        for p in self.players.values():
            p.is_it = False
            p.tag_cooldown = 0
            p.it_ticks = 0
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(30, 80)
            p.x = math.cos(angle) * dist
            p.y = math.sin(angle) * dist
            p.vx = 0.0
            p.vy = 0.0
        if self.players:
            p0 = random.choice(list(self.players.values()))
            p0.is_it = True
            p0.tag_cooldown = TAG_COOLDOWN * 2
        self.events.append({"type": "round_start", "round": self.round_number})

    # ── World Generation ─────────────────────────────────────────────

    def generate_chunk(self, cx: int, cy: int) -> Chunk:
        key = (cx, cy)
        if key in self.chunks:
            return self.chunks[key]

        chunk = Chunk(cx, cy)

        for ty in range(TILES_PER_CHUNK):
            for tx in range(TILES_PER_CHUNK):
                tile_world_x = cx * CHUNK_SIZE + tx * TILE_SIZE + TILE_SIZE * 0.5
                tile_world_y = cy * CHUNK_SIZE + ty * TILE_SIZE + TILE_SIZE * 0.5

                dist_origin = math.sqrt(tile_world_x ** 2 + tile_world_y ** 2)
                if dist_origin < SPAWN_CLEAR_RADIUS:
                    continue

                world_tx = cx * TILES_PER_CHUNK + tx
                world_ty = cy * TILES_PER_CHUNK + ty

                n1 = noise2d(
                    world_tx * NOISE_SCALE_PRIMARY,
                    world_ty * NOISE_SCALE_PRIMARY,
                    seed=self.seed, octaves=4, persistence=0.45,
                )
                n2 = noise2d(
                    world_tx * NOISE_SCALE_DETAIL + 500,
                    world_ty * NOISE_SCALE_DETAIL + 500,
                    seed=self.seed + 42, octaves=2, persistence=0.4,
                )

                is_wall = n1 > WALL_THRESHOLD or (n1 > -0.05 and n2 > DETAIL_THRESHOLD)

                if is_wall:
                    chunk.wall_grid[ty][tx] = True

        rng = random.Random(self.seed ^ (cx * 73856093) ^ (cy * 19349663))
        for ty in range(1, TILES_PER_CHUNK - 1):
            for tx in range(1, TILES_PER_CHUNK - 1):
                if not chunk.wall_grid[ty][tx]:
                    continue
                neighbors = (
                    chunk.wall_grid[ty - 1][tx]
                    and chunk.wall_grid[ty + 1][tx]
                    and chunk.wall_grid[ty][tx - 1]
                    and chunk.wall_grid[ty][tx + 1]
                )
                if neighbors and rng.random() < DASH_ORB_CHANCE:
                    chunk.dash_tiles.add((tx, ty))

        self.chunks[key] = chunk
        return chunk

    def _is_wall_at(self, world_tx: int, world_ty: int) -> bool:
        cx = world_tx // TILES_PER_CHUNK
        cy = world_ty // TILES_PER_CHUNK
        chunk = self.generate_chunk(cx, cy)
        local_tx = world_tx % TILES_PER_CHUNK
        local_ty = world_ty % TILES_PER_CHUNK
        return chunk.wall_grid[local_ty][local_tx]

    # ── Raycasting ───────────────────────────────────────────────────

    def raycast_distances(self, x: float, y: float) -> List[float]:
        distances = []
        for i in range(NUM_RAYCASTS):
            angle = i * 2.0 * math.pi / NUM_RAYCASTS
            dx = math.cos(angle)
            dy = math.sin(angle)
            hit = float(RAYCAST_MAX)
            for s in range(1, RAYCAST_MAX // RAYCAST_STEP + 1):
                px = x + dx * s * RAYCAST_STEP
                py = y + dy * s * RAYCAST_STEP
                tx = int(math.floor(px / TILE_SIZE))
                ty = int(math.floor(py / TILE_SIZE))
                if self._is_wall_at(tx, ty):
                    hit = float(s * RAYCAST_STEP)
                    break
            distances.append(hit)
        return distances

    # ── Collision ────────────────────────────────────────────────────

    @staticmethod
    def _circle_rect_push(cx, cy, radius, rx, ry, rw, rh):
        closest_x = max(rx, min(cx, rx + rw))
        closest_y = max(ry, min(cy, ry + rh))
        dx = cx - closest_x
        dy = cy - closest_y
        dist_sq = dx * dx + dy * dy
        if dist_sq < radius * radius:
            dist = math.sqrt(dist_sq) if dist_sq > 0.0001 else 0.01
            overlap = radius - dist
            return (dx / dist * overlap, dy / dist * overlap)
        return None

    def _resolve_wall_collisions(self, p: Player):
        tile_x = int(math.floor(p.x / TILE_SIZE))
        tile_y = int(math.floor(p.y / TILE_SIZE))
        for dtx in range(-1, 2):
            for dty in range(-1, 2):
                wtx = tile_x + dtx
                wty = tile_y + dty
                if self._is_wall_at(wtx, wty):
                    wx = wtx * TILE_SIZE
                    wy = wty * TILE_SIZE
                    push = self._circle_rect_push(
                        p.x, p.y, PLAYER_RADIUS, wx, wy, TILE_SIZE, TILE_SIZE
                    )
                    if push:
                        p.x += push[0]
                        p.y += push[1]
                        if abs(push[0]) > abs(push[1]):
                            p.vx = 0
                        else:
                            p.vy = 0

    # ── Tick ─────────────────────────────────────────────────────────

    def tick(self):
        self.events = []
        self.tick_count += 1

        for p in self.players.values():
            if p.dash_ticks_left > 0:
                p.dash_ticks_left -= 1
                p.x += p.vx
                p.y += p.vy
            else:
                speed_mult = IT_SPEED_BONUS if p.is_it else 1.0
                accel = ACCELERATION * speed_mult
                max_spd = MAX_SPEED * speed_mult

                p.vx += p.input_dx * accel
                p.vy += p.input_dy * accel
                p.vx *= FRICTION
                p.vy *= FRICTION

                speed = math.sqrt(p.vx * p.vx + p.vy * p.vy)
                if speed > max_spd:
                    p.vx = p.vx / speed * max_spd
                    p.vy = p.vy / speed * max_spd

                p.x += p.vx
                p.y += p.vy

            dist_from_origin = math.sqrt(p.x * p.x + p.y * p.y)
            if dist_from_origin > WORLD_BOUNDARY:
                pull = (dist_from_origin - WORLD_BOUNDARY) * 0.003
                p.vx -= (p.x / dist_from_origin) * pull
                p.vy -= (p.y / dist_from_origin) * pull

            self._resolve_wall_collisions(p)

            if p.tag_cooldown > 0:
                p.tag_cooldown -= TICK_DT
            if p.dash_cooldown > 0:
                p.dash_cooldown -= TICK_DT
            if p.tagged_by_timer > 0:
                p.tagged_by_timer -= TICK_DT
                if p.tagged_by_timer <= 0:
                    p.last_tagged_by = ""
            if p.break_cooldown > 0:
                p.break_cooldown -= TICK_DT

        # ── Round reset countdown (infestation) ──
        if self.round_reset_countdown > 0:
            self.round_reset_countdown -= 1
            if self.round_reset_countdown == 0:
                self._reset_infestation_round()
            return

        # ── Tag resolution ──
        if self.game_mode == "infestation":
            for a in list(self.players.values()):
                if not a.is_it or a.tag_cooldown > 0:
                    continue
                for b in list(self.players.values()):
                    if b.id == a.id or b.is_it:
                        continue
                    dx = a.x - b.x
                    dy = a.y - b.y
                    if dx * dx + dy * dy < TAG_RADIUS * TAG_RADIUS:
                        b.is_it = True
                        b.tag_cooldown = TAG_COOLDOWN
                        a.tag_cooldown = TAG_COOLDOWN * 0.4
                        b.last_tagged_by = a.id
                        b.tagged_by_timer = 6.0
                        self.events.append({
                            "type": "tag",
                            "tagger_id": a.id,
                            "tagged_id": b.id,
                            "tagger_name": a.name,
                            "tagged_name": b.name,
                        })

            survivors = [p for p in self.players.values() if not p.is_it]
            if len(survivors) <= 1 and len(self.players) >= 2:
                winner = survivors[0] if survivors else None
                self.events.append({
                    "type": "round_end",
                    "winner_name": winner.name if winner else "Nobody",
                    "winner_id": winner.id if winner else "",
                    "round": self.round_number,
                })
                self.round_reset_countdown = INFESTATION_RESET_TICKS
        else:
            tag_happened = False
            for a in list(self.players.values()):
                if tag_happened:
                    break
                if not a.is_it or a.tag_cooldown > 0:
                    continue
                for b in list(self.players.values()):
                    if b.id == a.id or b.is_it:
                        continue
                    dx = a.x - b.x
                    dy = a.y - b.y
                    if dx * dx + dy * dy < TAG_RADIUS * TAG_RADIUS:
                        a.is_it = False
                        b.is_it = True
                        b.tag_cooldown = TAG_COOLDOWN
                        b.last_tagged_by = a.id
                        b.tagged_by_timer = 6.0
                        a.it_ticks = 0
                        self.events.append({
                            "type": "tag",
                            "tagger_id": a.id,
                            "tagged_id": b.id,
                            "tagger_name": a.name,
                            "tagged_name": b.name,
                        })
                        tag_happened = True
                        break

        # ── Ping for IT players ──
        for p in self.players.values():
            if p.is_it:
                p.it_ticks += 1
                if (p.it_ticks >= PING_AFTER_TICKS
                        and (p.it_ticks - PING_AFTER_TICKS) % PING_REPEAT_TICKS == 0):
                    others = [o for o in self.players.values()
                              if o.id != p.id and not o.is_it]
                    if others:
                        others.sort(key=lambda o: (o.x - p.x) ** 2 + (o.y - p.y) ** 2)
                        nearest = others[0]
                        dx = nearest.x - p.x
                        dy = nearest.y - p.y
                        dist = math.sqrt(dx * dx + dy * dy) or 1.0
                        self.events.append({
                            "type": "ping",
                            "player_id": p.id,
                            "dx": round(dx / dist, 3),
                            "dy": round(dy / dist, 3),
                        })
            else:
                p.it_ticks = 0

    # ── Serialization ────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "type": "state",
            "tick": self.tick_count,
            "mode": self.game_mode,
            "survivors": sum(1 for p in self.players.values() if not p.is_it),
            "round": self.round_number,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "x": round(p.x, 1),
                    "y": round(p.y, 1),
                    "vx": round(p.vx, 2),
                    "vy": round(p.vy, 2),
                    "is_it": p.is_it,
                    "is_bot": p.is_bot,
                    "dash_cd": round(max(0, p.dash_cooldown), 2),
                    "dash_charges": p.dash_charges,
                    "dashing": p.dash_ticks_left > 0,
                    "break_cd": round(max(0, p.break_cooldown), 2),
                }
                for p in self.players.values()
            ],
            "events": self.events,
        }

    def get_chunks_data(self, x: float, y: float, radius: int = 2) -> dict:
        cx = int(math.floor(x / CHUNK_SIZE))
        cy = int(math.floor(y / CHUNK_SIZE))
        chunks_out = []
        for dcx in range(-radius, radius + 1):
            for dcy in range(-radius, radius + 1):
                chunk = self.generate_chunk(cx + dcx, cy + dcy)
                chunks_out.append({
                    "cx": chunk.cx,
                    "cy": chunk.cy,
                    "walls": chunk.get_walls(),
                    "dash_tiles": chunk.get_dash_tile_positions(),
                })
        return {"type": "chunks", "chunks": chunks_out}

    def get_observation(self, player_id: str) -> Optional[dict]:
        if player_id not in self.players:
            return None

        p = self.players[player_id]
        others = [o for o in self.players.values() if o.id != player_id]
        others.sort(key=lambda o: (o.x - p.x) ** 2 + (o.y - p.y) ** 2)
        nearest = others[:5]

        tile_x = int(math.floor(p.x / TILE_SIZE))
        tile_y = int(math.floor(p.y / TILE_SIZE))
        nearby_walls = []
        for dtx in range(-3, 4):
            for dty in range(-3, 4):
                wtx = tile_x + dtx
                wty = tile_y + dty
                if self._is_wall_at(wtx, wty):
                    nearby_walls.append((
                        wtx * TILE_SIZE,
                        wty * TILE_SIZE,
                        TILE_SIZE,
                        TILE_SIZE,
                    ))

        nearby_orbs = []
        for dtx in range(-5, 6):
            for dty in range(-5, 6):
                wtx = tile_x + dtx
                wty = tile_y + dty
                cx_o = wtx // TILES_PER_CHUNK
                cy_o = wty // TILES_PER_CHUNK
                chunk_o = self.chunks.get((cx_o, cy_o))
                if chunk_o:
                    ltx = wtx % TILES_PER_CHUNK
                    lty = wty % TILES_PER_CHUNK
                    if (ltx, lty) in chunk_o.dash_tiles:
                        nearby_orbs.append((
                            wtx * TILE_SIZE + TILE_SIZE * 0.5 - p.x,
                            wty * TILE_SIZE + TILE_SIZE * 0.5 - p.y,
                        ))

        return {
            "self_x": p.x,
            "self_y": p.y,
            "self_vx": p.vx,
            "self_vy": p.vy,
            "is_it": p.is_it,
            "tag_cooldown": p.tag_cooldown,
            "dash_cooldown": p.dash_cooldown,
            "dash_charges": p.dash_charges,
            "break_cooldown": p.break_cooldown,
            "is_dashing": p.dash_ticks_left > 0,
            "it_ticks": p.it_ticks,
            "last_tagged_by": p.last_tagged_by,
            "nearby_orbs": nearby_orbs,
            "nearest": [
                {
                    "id": o.id,
                    "dx": o.x - p.x,
                    "dy": o.y - p.y,
                    "vx": o.vx,
                    "vy": o.vy,
                    "is_it": o.is_it,
                    "dash_cd": o.dash_cooldown,
                    "is_dashing": o.dash_ticks_left > 0,
                }
                for o in nearest
            ],
            "walls_nearby": nearby_walls,
            "raycasts": self.raycast_distances(p.x, p.y),
        }
