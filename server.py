import asyncio
import json
import os
import random
import time
import uuid

import websockets

from engine import GameEngine, TICK_RATE, TICK_DT
from bots import ScriptedBot, RLBot

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8765))
MIN_PLAYERS = 4
MAX_PLAYERS = 8
ROOM_IDLE_TIMEOUT = 30.0

POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_policy.pt")

PSYCHOLOGIST_NAMES = [
    "Freud", "Jung", "Pavlov", "Skinner", "Maslow",
    "Rogers", "Piaget", "Erikson", "Bandura", "Milgram",
    "Zimbardo", "Adler", "Wundt", "James", "Watson",
    "Thorndike", "Seligman", "Kahneman", "Beck", "Frankl",
    "Fromm", "Horney", "Klein", "Lacan", "Bowlby",
    "Ainsworth", "Vygotsky", "Bruner", "Chomsky", "Festinger",
    "Asch", "Lewin", "Allport", "Eysenck", "Dweck",
    "Goleman", "Gardner", "Sternberg", "Binet", "Wechsler",
    "Rorschach", "Harlow", "Lorenz", "Ekman", "Damasio",
    "Pinker", "Sacks", "Luria", "Broca", "Wernicke",
    "Sperry", "Ramachandran", "Neisser", "Tulving", "Loftus",
    "Ebbinghaus", "Miller", "Broadbent", "Treisman", "Posner",
    "Zajonc", "Lazarus", "Selye", "Cannon", "Cialdini",
    "Tajfel", "Kohlberg", "Gilligan", "Montessori", "Bloom",
    "Kolb", "Deci", "Plutchik", "Yerkes", "Wolpe",
    "Perls", "May", "Yalom", "Winnicott", "Berne",
    "Grof", "Rank", "Reich", "Sullivan", "Murray",
    "Cattell", "Spearman", "Guilford", "Wertheimer", "Koffka",
    "Kohler", "Mischel", "Rotter", "Dollard", "Hull",
    "Tolman", "Guthrie", "Rescorla", "Garcia", "Bolles",
]


def _make_bot_brain():
    if os.path.exists(POLICY_PATH):
        bot = RLBot(POLICY_PATH)
        if bot.policy is not None:
            return bot
    return ScriptedBot()


# ─── Game Room ────────────────────────────────────────────────────────

class GameRoom:
    def __init__(self, room_id: str, mode: str):
        self.id = room_id
        self.engine = GameEngine(game_mode=mode)
        self.clients: dict = {}
        self.bot_brains: dict = {}
        self._name_pool = list(PSYCHOLOGIST_NAMES)
        random.shuffle(self._name_pool)
        self._name_idx = 0
        self.last_human_time = time.monotonic()

    @property
    def mode(self) -> str:
        return self.engine.game_mode

    @property
    def human_count(self) -> int:
        return len(self.clients)

    @property
    def total_count(self) -> int:
        return len(self.engine.players)

    def has_space(self) -> bool:
        return self.human_count < MAX_PLAYERS - 1

    def _next_bot_name(self) -> str:
        name = self._name_pool[self._name_idx % len(self._name_pool)]
        self._name_idx += 1
        return name

    def add_human(self, ws, player_id: str, name: str):
        player = self.engine.add_player(player_id, name)
        self.clients[ws] = player_id
        self.last_human_time = time.monotonic()
        self._adjust_bots()
        return player

    def remove_human(self, ws):
        pid = self.clients.get(ws)
        if pid:
            self.engine.remove_player(pid)
            del self.clients[ws]
        self._adjust_bots()
        if not self.clients:
            self.last_human_time = time.monotonic()

    def _adjust_bots(self):
        if self.human_count == 0:
            bot_target = 0
        else:
            bot_target = max(1, MAX_PLAYERS - self.human_count)

        bot_ids = list(self.bot_brains.keys())
        current = len(bot_ids)

        while current > bot_target and bot_ids:
            bid = bot_ids.pop()
            self.engine.remove_player(bid)
            del self.bot_brains[bid]
            current -= 1

        while current < bot_target:
            bid = f"bot_{uuid.uuid4().hex[:6]}"
            name = self._next_bot_name()
            self.engine.add_player(bid, name, is_bot=True)
            self.bot_brains[bid] = _make_bot_brain()
            current += 1

    async def tick_and_broadcast(self):
        if not self.clients:
            return

        for bot_id, brain in list(self.bot_brains.items()):
            obs = self.engine.get_observation(bot_id)
            if obs:
                dx, dy, dash, do_break = brain.get_action(obs)
                self.engine.set_input(bot_id, dx, dy)
                if dash:
                    self.engine.trigger_dash(bot_id)
                if do_break:
                    self.engine.trigger_break(bot_id)

        self.engine.tick()

        state_json = json.dumps(self.engine.get_state())
        send_chunks = (self.engine.tick_count % TICK_RATE == 0)

        tasks = []
        for ws, pid in list(self.clients.items()):
            async def _send(ws=ws, pid=pid):
                try:
                    await ws.send(state_json)
                    if send_chunks:
                        player = self.engine.players.get(pid)
                        if player:
                            cdata = json.dumps(
                                self.engine.get_chunks_data(player.x, player.y, radius=2)
                            )
                            await ws.send(cdata)
                except websockets.exceptions.ConnectionClosed:
                    pass
            tasks.append(_send())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# ─── Room Manager ─────────────────────────────────────────────────────

class RoomManager:
    def __init__(self):
        self.rooms: dict = {}

    def find_or_create(self, mode: str) -> GameRoom:
        best = None
        for room in self.rooms.values():
            if room.mode == mode and room.has_space():
                if best is None or room.human_count > best.human_count:
                    best = room
        if best:
            return best

        room_id = uuid.uuid4().hex[:6]
        room = GameRoom(room_id, mode)
        self.rooms[room_id] = room
        return room

    def cleanup(self):
        now = time.monotonic()
        to_remove = [
            rid for rid, room in self.rooms.items()
            if room.human_count == 0
            and (now - room.last_human_time) > ROOM_IDLE_TIMEOUT
        ]
        for rid in to_remove:
            del self.rooms[rid]

    @property
    def stats(self):
        return {
            "rooms": len(self.rooms),
            "humans": sum(r.human_count for r in self.rooms.values()),
            "total": sum(r.total_count for r in self.rooms.values()),
        }


# ─── Game Server ──────────────────────────────────────────────────────

class GameServer:
    def __init__(self):
        self.manager = RoomManager()
        self.client_rooms: dict = {}

    async def handle_client(self, ws):
        player_id = None
        room = None
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "join" and room is None:
                    name = str(msg.get("name", "Player"))[:16]
                    req_mode = str(msg.get("mode", "classic"))
                    if req_mode not in ("classic", "infestation"):
                        req_mode = "classic"

                    room = self.manager.find_or_create(req_mode)
                    player_id = uuid.uuid4().hex[:8]
                    player = room.add_human(ws, player_id, name)
                    self.client_rooms[ws] = room

                    await ws.send(json.dumps({
                        "type": "welcome",
                        "id": player_id,
                        "seed": room.engine.seed,
                        "mode": room.mode,
                        "room": room.id,
                    }))
                    chunks = room.engine.get_chunks_data(player.x, player.y, radius=2)
                    await ws.send(json.dumps(chunks))

                elif msg.get("type") == "input" and player_id and room:
                    dx = float(msg.get("dx", 0))
                    dy = float(msg.get("dy", 0))
                    room.engine.set_input(player_id, dx, dy)
                    if msg.get("dash"):
                        room.engine.trigger_dash(player_id)
                    if msg.get("break"):
                        room.engine.trigger_break(player_id)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if room:
                room.remove_human(ws)
            if ws in self.client_rooms:
                del self.client_rooms[ws]

    async def game_loop(self):
        cleanup_counter = 0
        while True:
            start = time.monotonic()

            for room in list(self.manager.rooms.values()):
                await room.tick_and_broadcast()

            cleanup_counter += 1
            if cleanup_counter >= TICK_RATE * 5:
                self.manager.cleanup()
                cleanup_counter = 0

            elapsed = time.monotonic() - start
            sleep_time = TICK_DT - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


async def main():
    server = GameServer()
    async with websockets.serve(server.handle_client, HOST, PORT):
        stats = server.manager.stats
        print(f"Freerun server running on ws://{HOST}:{PORT}")
        await server.game_loop()


if __name__ == "__main__":
    asyncio.run(main())
