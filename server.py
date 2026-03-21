import asyncio
import json
import os
import random
import time
import urllib.parse
import uuid

from aiohttp import web
import aiohttp

from engine import GameEngine, TICK_RATE, TICK_DT
from bots import HybridBot
from brain import BotBrain

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8765))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_PLAYERS = 4
MAX_PLAYERS = 8
ROOM_IDLE_TIMEOUT = 30.0

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


_shared_brain = BotBrain.shared()


def _make_bot_brain():
    return HybridBot(brain=_shared_brain)


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
                    await ws.send_str(state_json)
                    if send_chunks:
                        player = self.engine.players.get(pid)
                        if player:
                            cdata = json.dumps(
                                self.engine.get_chunks_data(player.x, player.y, radius=2)
                            )
                            await ws.send_str(cdata)
                except Exception:
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


# ─── Built-in HTTP + WebSocket serving (aiohttp) ─────────────────────

_LOBBY_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Freerun</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e17;color:#fff;font-family:'Segoe UI',sans-serif;
display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh}
h1{letter-spacing:6px;font-size:48px;margin-bottom:4px}
p.sub{color:#556;font-size:14px;margin-bottom:40px}
form{display:flex;flex-direction:column;gap:14px;width:280px}
input{padding:12px;border:1px solid #334;border-radius:8px;background:#141a26;
color:#fff;font-size:16px;outline:none}
input:focus{border-color:#00ff96}
.modes{display:flex;gap:10px}
.modes label{flex:1;text-align:center;padding:10px;border:1px solid #334;
border-radius:8px;cursor:pointer;font-size:14px;transition:0.2s}
.modes input[type=radio]{display:none}
.modes input[type=radio]:checked+span{color:#00ff96}
.modes label:has(input:checked){border-color:#00ff96;background:#0d1a12}
button{padding:14px;border:none;border-radius:8px;background:#00ff96;color:#0a0e17;
font-size:18px;font-weight:bold;cursor:pointer;letter-spacing:2px}
button:hover{background:#00cc78}
.hint{color:#334;font-size:11px;margin-top:30px;text-align:center}
</style></head><body>
<h1>FREERUN</h1><p class="sub">infinite tag</p>
<form action="/play" method="GET">
<input name="name" placeholder="Enter a name..." maxlength="16" required autocomplete="off">
<div class="modes">
<label><input type="radio" name="mode" value="classic" checked><span>Classic</span></label>
<label><input type="radio" name="mode" value="infestation"><span>Infestation</span></label>
<label><input type="radio" name="mode" value="hvb"><span>Human vs Bot</span></label>
</div>
<button type="submit">PLAY</button>
</form>
<p class="hint">WASD move &middot; SPACE dash &middot; E break</p>
</body></html>"""


def _build_game_html(name, mode, ws_url):
    path = os.path.join(SCRIPT_DIR, "client.html")
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__PLAYER_NAME__", name.replace('"', '\\"'))
    html = html.replace("__WS_URL__", ws_url)
    html = html.replace("__GAME_MODE__", mode)
    return html


def _ws_url_from_request(request):
    host = request.headers.get("Host", f"localhost:{PORT}")
    if "onrender.com" in host or request.scheme == "https":
        return f"wss://{host}/ws"
    return f"ws://{host}/ws"


_game_server = GameServer()


async def handle_lobby(request):
    return web.Response(text=_LOBBY_HTML, content_type="text/html")


async def handle_play(request):
    name = request.query.get("name", "Player")[:16]
    mode = request.query.get("mode", "classic")
    if mode not in ("classic", "infestation", "hvb"):
        mode = "classic"
    ws_url = _ws_url_from_request(request)
    html = _build_game_html(name, mode, ws_url)
    return web.Response(text=html, content_type="text/html")


async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    player_id = None
    room = None
    try:
        async for raw in ws:
            if raw.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                msg = json.loads(raw.data)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "join" and room is None:
                name = str(msg.get("name", "Player"))[:16]
                req_mode = str(msg.get("mode", "classic"))
                if req_mode not in ("classic", "infestation", "hvb"):
                    req_mode = "classic"

                room = _game_server.manager.find_or_create(req_mode)
                player_id = uuid.uuid4().hex[:8]
                player = room.add_human(ws, player_id, name)
                _game_server.client_rooms[ws] = room

                await ws.send_str(json.dumps({
                    "type": "welcome",
                    "id": player_id,
                    "seed": room.engine.seed,
                    "mode": room.mode,
                    "room": room.id,
                }))
                chunks = room.engine.get_chunks_data(player.x, player.y, radius=2)
                await ws.send_str(json.dumps(chunks))

            elif msg.get("type") == "input" and player_id and room:
                dx = float(msg.get("dx", 0))
                dy = float(msg.get("dy", 0))
                room.engine.set_input(player_id, dx, dy)
                if msg.get("dash"):
                    room.engine.trigger_dash(player_id)
                if msg.get("break"):
                    room.engine.trigger_break(player_id)
    finally:
        if room:
            room.remove_human(ws)
        if ws in _game_server.client_rooms:
            del _game_server.client_rooms[ws]

    return ws


async def game_loop_task(app):
    async def _loop():
        cleanup_counter = 0
        while True:
            start = time.monotonic()
            for room in list(_game_server.manager.rooms.values()):
                await room.tick_and_broadcast()
            cleanup_counter += 1
            if cleanup_counter >= TICK_RATE * 5:
                _game_server.manager.cleanup()
                cleanup_counter = 0
            elapsed = time.monotonic() - start
            sleep_time = TICK_DT - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    app["game_loop"] = asyncio.create_task(_loop())


async def cleanup_game_loop(app):
    app["game_loop"].cancel()
    try:
        await app["game_loop"]
    except asyncio.CancelledError:
        pass
    _shared_brain.save()


def main():
    app = web.Application()
    app.router.add_get("/", handle_lobby)
    app.router.add_get("/play", handle_play)
    app.router.add_get("/ws", handle_ws)
    app.on_startup.append(game_loop_task)
    app.on_cleanup.append(cleanup_game_loop)
    print(f"Freerun server starting on http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
