import pygame
import numpy as np
import math
import random
import sys
import time

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080     # display resolution
WORLD_W, WORLD_H = 5000, 3500       # simulation world size
WIDTH, HEIGHT = WORLD_W, WORLD_H    # alias for world bounds (used everywhere)
FPS = 60
BG_COLOR = (195, 195, 195)
GRID_LINE_COLOR = (182, 182, 182)
GRID_SPACING = 100

# Starting counts
NUM_PREDATORS = 200
NUM_PREY = 500
NUM_FOLIAGE = 1200
MAX_FOLIAGE = 2000

# Population limits
MAX_PREDATORS = 500
MAX_PREY = 1000

# LOD: below this screen-pixel size, draw a dot instead of blob sprite
LOD_DOT_THRESHOLD = 4

# Vision
NUM_RAYS = 20
PREDATOR_FOV = math.radians(60)
PREDATOR_VISION_RANGE = 350.0      # scaled up for larger world
PREY_FOV = math.radians(270)
PREY_VISION_RANGE = 200.0          # scaled up for larger world

# HP
PREY_MAX_HP = 2
DAMAGE_COOLDOWN = 45
CONTACT_DIST = 14

# Energy
PREDATOR_MAX_ENERGY = 300.0
PREY_MAX_ENERGY = 200.0
PREDATOR_ENERGY_DRAIN = 0.35
PREY_ENERGY_DRAIN_MOVE = 0.03
PREDATOR_EAT_GAIN = 120.0
PREY_EAT_GAIN = 60.0

# Movement
PREDATOR_BASE_SPEED = 2.5
PREY_BASE_SPEED = 3.2
CREATURE_RADIUS = 8
MAX_TURN_RATE = 0.25

# Foliage
FOLIAGE_RADIUS = 3
FOLIAGE_REGROW_RATE = 0.8          # higher for bigger world
PREY_EAT_DIST = 10

# Reproduction
PRED_REPRODUCE_THRESHOLD = 0.70
PREY_REPRODUCE_THRESHOLD = 0.80

# Neural network
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.3

NN_INPUTS = NUM_RAYS * 2 + 2    # per ray: distance + type_value (-1/0/+1), plus energy + hp
NN_HIDDEN = 16
NN_OUTPUTS = 2

T_PREDATOR = 1
T_PREY = 2
T_FOLIAGE = 3

GRID_CELL = 360

# ─── COLORS (Pezzza-inspired muted palette) ─────────────────────────────────
COL_PREDATOR = (140, 35, 35)
COL_PREDATOR_BODY = (120, 28, 28)
COL_PREY = (75, 185, 140)
COL_PREY_BODY = (60, 165, 120)
COL_FOLIAGE_VARY = [(170, 210, 175), (160, 200, 165), (175, 215, 180), (165, 205, 170)]
COL_RAY = (150, 150, 155)
COL_HUD_BG = (30, 30, 35, 200)
COL_HUD_TEXT = (220, 220, 225)
COL_HUD_DIM = (140, 140, 150)

# Sprite cache for stretched+rotated blobs
_sprite_cache = {}
ANGLE_BUCKETS = 72       # 5-degree resolution
STRETCH_LEVELS = 6       # quantized stretch levels


def create_blob_sprite(base_color, radius):
    """Pre-render a glossy 3D sphere blob with highlight and soft edges."""
    pad = 4
    d = (radius + pad) * 2
    surf = pygame.Surface((d, d), pygame.SRCALPHA)

    cx, cy = d / 2.0, d / 2.0
    xs = np.arange(d, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    norm = dist / (radius + 0.5)

    # Alpha: fully opaque center ~70%, then soft falloff at rim
    alpha = np.where(
        norm < 0.7, 255.0,
        np.clip((1.0 - ((norm - 0.7) / 0.35)) * 255, 0, 255)
    )
    alpha = np.where(norm > 1.05, 0, alpha).astype(np.uint8)

    # Glossy highlight spot (offset upper-left, like a shiny sphere)
    hl_x, hl_y = cx - radius * 0.3, cy - radius * 0.3
    hl_dist = np.sqrt((xx - hl_x) ** 2 + (yy - hl_y) ** 2)
    hl_norm = np.clip(hl_dist / (radius * 0.8), 0, 1)

    bright = np.array([min(255, c + 110) for c in base_color], dtype=np.float32)
    base = np.array(base_color, dtype=np.float32)
    dark = np.array([max(0, c - 45) for c in base_color], dtype=np.float32)

    # Highlight influence + rim darkening
    hl_t = hl_norm ** 1.6
    edge_t = np.clip(norm, 0, 1) ** 1.3

    pix3d = pygame.surfarray.pixels3d(surf)
    for c in range(3):
        col = bright[c] * (1 - hl_t) + base[c] * hl_t
        col = col * (1 - edge_t * 0.5) + dark[c] * (edge_t * 0.5)
        pix3d[:, :, c] = np.clip(col, 0, 255).astype(np.uint8)
    del pix3d

    a_arr = pygame.surfarray.pixels_alpha(surf)
    a_arr[:] = alpha
    del a_arr

    return surf


def get_cached_blob(base_sprite, stretch_idx, angle_bucket):
    """Get a stretched+rotated blob from cache, or create and cache it."""
    key = (id(base_sprite), stretch_idx, angle_bucket)
    cached = _sprite_cache.get(key)
    if cached is not None:
        return cached

    bw, bh = base_sprite.get_size()
    sx = 1.0 + stretch_idx * 0.14
    sy = 1.0 - stretch_idx * 0.02
    new_w = max(1, int(bw * sx))
    new_h = max(1, int(bh * sy))

    stretched = pygame.transform.smoothscale(base_sprite, (new_w, new_h))
    angle_deg = angle_bucket * (360.0 / ANGLE_BUCKETS)
    rotated = pygame.transform.rotate(stretched, -angle_deg)

    _sprite_cache[key] = rotated
    return rotated


# ─── SPATIAL GRID ────────────────────────────────────────────────────────────
class SpatialGrid:
    __slots__ = ('cell_size', 'cells')

    def __init__(self, cell_size=GRID_CELL):
        self.cell_size = cell_size
        self.cells = {}

    def clear(self):
        self.cells.clear()

    def insert(self, entity):
        key = (int(entity.x // self.cell_size), int(entity.y // self.cell_size))
        try:
            self.cells[key].append(entity)
        except KeyError:
            self.cells[key] = [entity]

    def query_nearby(self, x, y, radius):
        cs = self.cell_size
        min_cx = int((x - radius) // cs)
        max_cx = int((x + radius) // cs)
        min_cy = int((y - radius) // cs)
        max_cy = int((y + radius) // cs)
        result = []
        cells = self.cells
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                bucket = cells.get((cx, cy))
                if bucket:
                    result.extend(bucket)
        return result


# ─── NEURAL NETWORK ─────────────────────────────────────────────────────────
class NeuralNetwork:
    __slots__ = ('weights', 'biases')

    def __init__(self, sizes=None):
        if sizes is None:
            sizes = [NN_INPUTS, NN_HIDDEN, NN_OUTPUTS]
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]).astype(np.float32) * 0.5)
            self.biases.append(np.zeros(sizes[i + 1], dtype=np.float32))

    def forward(self, x):
        a = np.asarray(x, dtype=np.float32)
        for w, b in zip(self.weights, self.biases):
            a = np.tanh(a @ w + b)
        return a

    def mutate(self):
        child = NeuralNetwork.__new__(NeuralNetwork)
        child.weights = []
        child.biases = []
        for w, b in zip(self.weights, self.biases):
            nw = w.copy()
            nb = b.copy()
            wm = np.random.random(w.shape) < MUTATION_RATE
            nw[wm] += np.random.randn(wm.sum()) * MUTATION_STRENGTH
            bm = np.random.random(b.shape) < MUTATION_RATE
            nb[bm] += np.random.randn(bm.sum()) * MUTATION_STRENGTH
            child.weights.append(nw)
            child.biases.append(nb)
        return child


# ─── CREATURES ───────────────────────────────────────────────────────────────
class Creature:
    etype = 0

    def __init__(self, x, y, energy, max_energy, speed, color, body_color,
                 fov, vision_range, max_hp=0):
        self.x = x
        self.y = y
        self.energy = energy
        self.max_energy = max_energy
        self.speed = speed
        self.color = color
        self.body_color = body_color
        self.angle = random.uniform(0, 2 * math.pi)
        self.alive = True
        self.age = 0
        self.fov = fov
        self.vision_range = vision_range
        self.max_hp = max_hp
        self.hp = max_hp
        self.damage_timer = 0
        self.brain = NeuralNetwork()
        self.ray_endpoints = []
        self.ray_hits = []
        self.cur_speed = 0.0

    def take_damage(self):
        if self.damage_timer > 0:
            return False
        self.hp -= 1
        self.damage_timer = DAMAGE_COOLDOWN
        if self.hp <= 0:
            self.alive = False
        return True

    def see(self, nearby):
        half_fov = self.fov / 2
        sector_width = self.fov / NUM_RAYS
        vr = self.vision_range
        best_dist = [vr] * NUM_RAYS
        best_type = [0] * NUM_RAYS
        my_x, my_y, my_angle = self.x, self.y, self.angle

        visible = getattr(self, 'can_see', None)
        for e in nearby:
            if e is self or not e.alive:
                continue
            if visible and e.etype not in visible:
                continue
            dx = e.x - my_x
            dy = e.y - my_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1 or dist_sq > vr * vr:
                continue
            ang = math.atan2(dy, dx)
            rel = (ang - my_angle + math.pi) % (2 * math.pi) - math.pi
            if abs(rel) > half_fov:
                continue
            sector = int((rel + half_fov) / sector_width)
            if sector < 0:
                sector = 0
            elif sector >= NUM_RAYS:
                sector = NUM_RAYS - 1
            dist = math.sqrt(dist_sq)
            if dist < best_dist[sector]:
                best_dist[sector] = dist
                best_type[sector] = e.etype

        # Type encoding: +1 = food, -1 = danger, 0 = nothing
        # Predators: prey = food(+1)
        # Prey: foliage = food(+1), predator = danger(-1)
        type_map = getattr(self, 'type_signal', {})

        inputs = []
        self.ray_endpoints = []
        self.ray_hits = []
        inv_vr = 1.0 / vr
        for i in range(NUM_RAYS):
            ray_angle = my_angle - half_fov + (i + 0.5) * sector_width
            bt = best_type[i]
            bd = best_dist[i]
            inputs.append(bd * inv_vr)
            inputs.append(type_map.get(bt, 0.0))
            ca_r = math.cos(ray_angle)
            sa_r = math.sin(ray_angle)
            ex = my_x + ca_r * vr
            ey = my_y + sa_r * vr
            self.ray_endpoints.append((ex, ey))
            if bt > 0:
                hx = my_x + ca_r * bd
                hy = my_y + sa_r * bd
                self.ray_hits.append((hx, hy, bt))
            else:
                self.ray_hits.append(None)

        inputs.append(self.energy / self.max_energy)
        inputs.append(self.hp / self.max_hp if self.max_hp > 0 else 1.0)
        return inputs

    def think_and_move(self, inputs):
        out = self.brain.forward(inputs)
        self._last_turn = float(out[0])
        self._last_spd = (float(out[1]) + 1.0) / 2.0
        self._apply_movement()

    def coast(self):
        """Reuse last brain output — no vision/NN this frame."""
        self._apply_movement()

    def _apply_movement(self):
        turn = getattr(self, '_last_turn', 0.0)
        spd = getattr(self, '_last_spd', 0.5)
        self.angle += turn * MAX_TURN_RATE
        actual_speed = self.speed * max(0.1, spd)
        self.cur_speed = actual_speed
        self.x += math.cos(self.angle) * actual_speed
        self.y += math.sin(self.angle) * actual_speed
        if self.x < 0:
            self.x = -self.x
            self.angle = math.pi - self.angle
        elif self.x > WIDTH:
            self.x = 2 * WIDTH - self.x
            self.angle = math.pi - self.angle
        if self.y < 0:
            self.y = -self.y
            self.angle = -self.angle
        elif self.y > HEIGHT:
            self.y = 2 * HEIGHT - self.y
            self.angle = -self.angle
        self.x = max(0.0, min(float(WIDTH), self.x))
        self.y = max(0.0, min(float(HEIGHT), self.y))

    _blob_sprite = None      # set per subclass at startup
    _blob_sprite_hit = None  # white flash version

    def draw(self, surface, show_vision):
        if not self.alive:
            return
        ix, iy = int(self.x), int(self.y)
        r = CREATURE_RADIUS
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        perp_x, perp_y = -sa, ca

        # Vision cone
        if show_vision and self.ray_endpoints:
            for ex, ey in self.ray_endpoints:
                pygame.draw.line(surface, COL_RAY,
                                 (ix, iy), (int(ex), int(ey)), 1)

        # Quantize stretch and angle for cache lookup
        stretch = min(1.0, self.cur_speed / self.speed) if self.speed > 0 else 0
        stretch_idx = min(STRETCH_LEVELS - 1, int(stretch * STRETCH_LEVELS))

        angle_deg = math.degrees(self.angle) % 360
        angle_bucket = int(angle_deg / (360.0 / ANGLE_BUCKETS)) % ANGLE_BUCKETS

        # Pick sprite (normal or hit flash)
        use_hit = self.damage_timer > DAMAGE_COOLDOWN - 6
        base = self._blob_sprite_hit if use_hit else self._blob_sprite
        blob = get_cached_blob(base, stretch_idx, angle_bucket)

        # Blit centered
        rect = blob.get_rect(center=(ix, iy))
        surface.blit(blob, rect)

        # Two googly eyes — drawn on top of the blob
        eye_fwd = r * 0.3
        eye_spread = r * 0.35
        eye_r = max(3, int(r * 0.30))
        pupil_r = max(2, int(r * 0.14))

        for side in (-1, 1):
            ex = self.x + ca * eye_fwd + perp_x * eye_spread * side
            ey = self.y + sa * eye_fwd + perp_y * eye_spread * side

            # White sclera
            pygame.draw.circle(surface, (245, 245, 240),
                               (int(ex), int(ey)), eye_r)
            # Black pupil shifted in facing direction
            ppx = ex + ca * pupil_r * 0.8
            ppy = ey + sa * pupil_r * 0.8
            pygame.draw.circle(surface, (15, 15, 20),
                               (int(ppx), int(ppy)), pupil_r)

    def draw_zoomed(self, surface, show_vision, sim, z, is_selected=False):
        """Draw creature with camera zoom/pan applied."""
        sx, sy = sim.world_to_screen(self.x, self.y)
        ix, iy = int(sx), int(sy)
        r = CREATURE_RADIUS
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        perp_x, perp_y = -sa, ca

        # Vision cone — show if global toggle OR this creature is selected
        if (show_vision or is_selected) and self.ray_endpoints:
            for i, (ex, ey) in enumerate(self.ray_endpoints):
                esx, esy = sim.world_to_screen(ex, ey)
                pygame.draw.line(surface, COL_RAY,
                                 (ix, iy), (int(esx), int(esy)), 1)

            # Hit markers — colored dots where rays detected something
            if is_selected and hasattr(self, 'ray_hits'):
                for hit in self.ray_hits:
                    if hit is None:
                        continue
                    hx, hy, ht = hit
                    hsx, hsy = sim.world_to_screen(hx, hy)
                    if ht == T_PREDATOR:
                        hcol = (220, 50, 50)
                    elif ht == T_PREY:
                        hcol = (50, 200, 120)
                    else:
                        hcol = (120, 190, 120)
                    dot_r = max(3, int(4 * z))
                    pygame.draw.circle(surface, hcol,
                                       (int(hsx), int(hsy)), dot_r)
                    pygame.draw.circle(surface, (255, 255, 255),
                                       (int(hsx), int(hsy)), dot_r, 1)

        # Blob sprite
        stretch = min(1.0, self.cur_speed / self.speed) if self.speed > 0 else 0
        stretch_idx = min(STRETCH_LEVELS - 1, int(stretch * STRETCH_LEVELS))

        angle_deg = math.degrees(self.angle) % 360
        angle_bucket = int(angle_deg / (360.0 / ANGLE_BUCKETS)) % ANGLE_BUCKETS

        use_hit = self.damage_timer > DAMAGE_COOLDOWN - 6
        base = self._blob_sprite_hit if use_hit else self._blob_sprite
        blob = get_cached_blob(base, stretch_idx, angle_bucket)

        # Scale the blob by zoom
        if z != 1.0:
            bw, bh = blob.get_size()
            new_w = max(1, int(bw * z))
            new_h = max(1, int(bh * z))
            blob = pygame.transform.smoothscale(blob, (new_w, new_h))

        rect = blob.get_rect(center=(ix, iy))
        surface.blit(blob, rect)

        # Eyes (scaled by zoom)
        eye_fwd = r * 0.3 * z
        eye_spread = r * 0.35 * z
        eye_r = max(2, int(r * 0.30 * z))
        pupil_r = max(1, int(r * 0.14 * z))

        for side in (-1, 1):
            ex = sx + ca * eye_fwd + perp_x * eye_spread * side
            ey = sy + sa * eye_fwd + perp_y * eye_spread * side
            pygame.draw.circle(surface, (245, 245, 240),
                               (int(ex), int(ey)), eye_r)
            ppx = ex + ca * pupil_r * 0.8
            ppy = ey + sa * pupil_r * 0.8
            pygame.draw.circle(surface, (15, 15, 20),
                               (int(ppx), int(ppy)), pupil_r)

    def draw_mini(self, surface, scale_x, scale_y, ox, oy):
        """Draw on minimap."""
        if not self.alive:
            return
        mx = ox + int(self.x * scale_x)
        my = oy + int(self.y * scale_y)
        pygame.draw.circle(surface, self.color, (mx, my), 2)


class Predator(Creature):
    etype = T_PREDATOR
    can_see = {T_PREY}
    type_signal = {T_PREY: 1.0}     # prey = food (+1)

    def __init__(self, x, y, brain=None):
        super().__init__(x, y, PREDATOR_MAX_ENERGY, PREDATOR_MAX_ENERGY,
                         PREDATOR_BASE_SPEED, COL_PREDATOR, COL_PREDATOR_BODY,
                         PREDATOR_FOV, PREDATOR_VISION_RANGE)
        if brain:
            self.brain = brain

    def update(self, nearby, think):
        self.age += 1
        self.energy -= PREDATOR_ENERGY_DRAIN
        if self.energy <= 0:
            self.alive = False
            return
        if think:
            inputs = self.see(nearby)
            self.think_and_move(inputs)
        else:
            self.coast()
        for e in nearby:
            if e.etype == T_PREY and e.alive:
                if (self.x - e.x) ** 2 + (self.y - e.y) ** 2 < CONTACT_DIST ** 2:
                    if e.take_damage():
                        self.energy = min(self.energy + PREDATOR_EAT_GAIN,
                                          self.max_energy)
                    break


class Prey(Creature):
    etype = T_PREY
    can_see = {T_PREDATOR, T_FOLIAGE}
    type_signal = {T_PREDATOR: -1.0, T_FOLIAGE: 1.0}  # danger (-1), food (+1)

    def __init__(self, x, y, brain=None):
        super().__init__(x, y, PREY_MAX_ENERGY, PREY_MAX_ENERGY,
                         PREY_BASE_SPEED, COL_PREY, COL_PREY_BODY,
                         PREY_FOV, PREY_VISION_RANGE, PREY_MAX_HP)
        if brain:
            self.brain = brain

    def update(self, nearby, think):
        self.age += 1
        if self.damage_timer > 0:
            self.damage_timer -= 1
        self.energy -= PREY_ENERGY_DRAIN_MOVE
        if self.energy <= 0:
            self.alive = False
            return
        if think:
            inputs = self.see(nearby)
            self.think_and_move(inputs)
        else:
            self.coast()
        for e in nearby:
            if e.etype == T_FOLIAGE and e.alive:
                if (self.x - e.x) ** 2 + (self.y - e.y) ** 2 < PREY_EAT_DIST ** 2:
                    e.alive = False
                    self.energy = min(self.energy + PREY_EAT_GAIN, self.max_energy)
                    break


class Foliage:
    etype = T_FOLIAGE

    def __init__(self, x=None, y=None):
        self.x = x if x is not None else random.uniform(20, WIDTH - 20)
        self.y = y if y is not None else random.uniform(20, HEIGHT - 20)
        self.alive = True
        self.color = random.choice(COL_FOLIAGE_VARY)

    def draw(self, surface):
        if self.alive:
            pygame.draw.circle(surface, self.color,
                               (int(self.x), int(self.y)), FOLIAGE_RADIUS)


# ─── SIMULATION ──────────────────────────────────────────────────────────────
class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Predator vs Prey — Evolving AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Helvetica", 14)
        self.font_big = pygame.font.SysFont("Helvetica", 22)
        self.font_timer = pygame.font.SysFont("Helvetica", 28, bold=True)
        self.running = True
        self.paused = False
        self.show_vision = False
        self.sim_speed = 1
        self.frame = 0
        self.generation = 0
        self.grid = SpatialGrid()
        self.selected = None
        self.start_time = time.time()

        # Camera: zoom + pan
        self.cam_zoom = 1.0
        self.cam_x = 0.0  # world X at screen center
        self.cam_y = 0.0  # world Y at screen center
        self.dragging = False
        self.drag_last = (0, 0)
        self._reset_camera()

        # Pre-render glossy blob sprites
        _sprite_cache.clear()
        Predator._blob_sprite = create_blob_sprite(COL_PREDATOR_BODY, CREATURE_RADIUS)
        Predator._blob_sprite_hit = create_blob_sprite((220, 220, 210), CREATURE_RADIUS)
        Prey._blob_sprite = create_blob_sprite(COL_PREY_BODY, CREATURE_RADIUS)
        Prey._blob_sprite_hit = create_blob_sprite((220, 220, 210), CREATURE_RADIUS)

        self._init_entities()

        self.pred_history = []
        self.prey_history = []
        self.history_interval = 10

        # Minimap dimensions
        self.mini_w = 180
        self.mini_h = int(self.mini_w * WORLD_H / WORLD_W)
        self.mini_surface = pygame.Surface((self.mini_w, self.mini_h), pygame.SRCALPHA)

    def _reset_camera(self):
        self.cam_zoom = min(SCREEN_W / WORLD_W, SCREEN_H / WORLD_H)
        self.cam_x = WORLD_W / 2.0
        self.cam_y = WORLD_H / 2.0

    def world_to_screen(self, wx, wy):
        sx = (wx - self.cam_x) * self.cam_zoom + SCREEN_W / 2
        sy = (wy - self.cam_y) * self.cam_zoom + SCREEN_H / 2
        return sx, sy

    def screen_to_world(self, sx, sy):
        wx = (sx - SCREEN_W / 2) / self.cam_zoom + self.cam_x
        wy = (sy - SCREEN_H / 2) / self.cam_zoom + self.cam_y
        return wx, wy

    def _init_entities(self):
        self.predators = [Predator(random.uniform(50, WIDTH - 50),
                                   random.uniform(50, HEIGHT - 50))
                          for _ in range(NUM_PREDATORS)]
        self.prey = [Prey(random.uniform(50, WIDTH - 50),
                          random.uniform(50, HEIGHT - 50))
                     for _ in range(NUM_PREY)]
        self.foliage = [Foliage() for _ in range(NUM_FOLIAGE)]

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_v:
                    self.show_vision = not self.show_vision
                elif event.key == pygame.K_UP:
                    self.sim_speed = min(self.sim_speed + 1, 10)
                elif event.key == pygame.K_DOWN:
                    self.sim_speed = max(self.sim_speed - 1, 1)
                elif event.key == pygame.K_r:
                    self._init_entities()
                    self.frame = 0
                    self.generation = 0
                    self.pred_history.clear()
                    self.prey_history.clear()
                    self.selected = None
                    self.start_time = time.time()
                elif event.key == pygame.K_HOME or event.key == pygame.K_h:
                    self._reset_camera()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:  # left click — select / place food
                    wx, wy = self.screen_to_world(mx, my)
                    self.selected = None
                    for c in self.predators + self.prey:
                        if c.alive and math.hypot(c.x - wx, c.y - wy) < 15 / self.cam_zoom:
                            self.selected = c
                            break
                    if not self.selected:
                        for _ in range(5):
                            self.foliage.append(
                                Foliage(wx + random.uniform(-30, 30),
                                        wy + random.uniform(-30, 30)))
                elif event.button == 3:  # right click — start pan
                    self.dragging = True
                    self.drag_last = (mx, my)
                elif event.button == 4:  # scroll up — zoom in
                    # Zoom toward mouse position
                    wx, wy = self.screen_to_world(mx, my)
                    self.cam_zoom = min(self.cam_zoom * 1.15, 8.0)
                    self.cam_x = wx - (mx - SCREEN_W / 2) / self.cam_zoom
                    self.cam_y = wy - (my - SCREEN_H / 2) / self.cam_zoom
                elif event.button == 5:  # scroll down — zoom out
                    wx, wy = self.screen_to_world(mx, my)
                    self.cam_zoom = max(self.cam_zoom / 1.15,
                                        min(SCREEN_W / WORLD_W, SCREEN_H / WORLD_H))
                    self.cam_x = wx - (mx - SCREEN_W / 2) / self.cam_zoom
                    self.cam_y = wy - (my - SCREEN_H / 2) / self.cam_zoom
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    mx, my = event.pos
                    dx = mx - self.drag_last[0]
                    dy = my - self.drag_last[1]
                    self.cam_x -= dx / self.cam_zoom
                    self.cam_y -= dy / self.cam_zoom
                    self.drag_last = (mx, my)
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                wx, wy = self.screen_to_world(mx, my)
                if event.y > 0:
                    self.cam_zoom = min(self.cam_zoom * 1.15, 8.0)
                elif event.y < 0:
                    self.cam_zoom = max(self.cam_zoom / 1.15,
                                        min(SCREEN_W / WORLD_W, SCREEN_H / WORLD_H))
                self.cam_x = wx - (mx - SCREEN_W / 2) / self.cam_zoom
                self.cam_y = wy - (my - SCREEN_H / 2) / self.cam_zoom

    def rebuild_grid(self):
        self.grid.clear()
        for p in self.predators:
            if p.alive:
                self.grid.insert(p)
        for p in self.prey:
            if p.alive:
                self.grid.insert(p)
        for f in self.foliage:
            if f.alive:
                self.grid.insert(f)

    def update(self):
        if self.paused:
            return
        self.frame += 1
        self.rebuild_grid()

        max_range = max(PREDATOR_VISION_RANGE, PREY_VISION_RANGE, CONTACT_DIST)
        # Alternate-frame thinking: half the creatures think each frame
        think_even = (self.frame % 2 == 0)
        for i, pred in enumerate(self.predators):
            if pred.alive:
                think = (i % 2 == 0) == think_even
                pred.update(self.grid.query_nearby(pred.x, pred.y, max_range),
                            think)
        for i, p in enumerate(self.prey):
            if p.alive:
                think = (i % 2 == 0) == think_even
                p.update(self.grid.query_nearby(p.x, p.y, max_range),
                         think)

        self.predators = [p for p in self.predators if p.alive]
        self.prey = [p for p in self.prey if p.alive]
        self.foliage = [f for f in self.foliage if f.alive]

        if self.selected and not self.selected.alive:
            self.selected = None

        if len(self.predators) < MAX_PREDATORS:
            new_preds = []
            for pred in self.predators:
                if pred.energy > pred.max_energy * PRED_REPRODUCE_THRESHOLD:
                    pred.energy *= 0.5
                    child = Predator(pred.x + random.uniform(-15, 15),
                                     pred.y + random.uniform(-15, 15),
                                     brain=pred.brain.mutate())
                    child.energy = pred.energy
                    new_preds.append(child)
                    self.generation += 1
            self.predators.extend(new_preds)

        if len(self.prey) < MAX_PREY:
            new_prey = []
            for p in self.prey:
                if p.energy > p.max_energy * PREY_REPRODUCE_THRESHOLD:
                    p.energy *= 0.5
                    child = Prey(p.x + random.uniform(-15, 15),
                                 p.y + random.uniform(-15, 15),
                                 brain=p.brain.mutate())
                    child.energy = p.energy
                    new_prey.append(child)
                    self.generation += 1
            self.prey.extend(new_prey)

        if len(self.foliage) < MAX_FOLIAGE and random.random() < FOLIAGE_REGROW_RATE:
            self.foliage.append(Foliage())

        if self.frame % self.history_interval == 0:
            self.pred_history.append(len(self.predators))
            self.prey_history.append(len(self.prey))
            if len(self.pred_history) > 500:
                self.pred_history.pop(0)
                self.prey_history.pop(0)

    # ─── DRAWING ─────────────────────────────────────────────────────────────
    def draw_minimap(self):
        mx, my = SCREEN_W - self.mini_w - 20, 20
        sx = self.mini_w / WORLD_W
        sy = self.mini_h / WORLD_H

        self.mini_surface.fill((160, 160, 160, 220))

        for f in self.foliage:
            if f.alive:
                px = int(f.x * sx)
                py = int(f.y * sy)
                self.mini_surface.set_at((min(px, self.mini_w - 1),
                                          min(py, self.mini_h - 1)),
                                         (140, 190, 145))

        for p in self.prey:
            p.draw_mini(self.mini_surface, sx, sy, 0, 0)
        for pred in self.predators:
            pred.draw_mini(self.mini_surface, sx, sy, 0, 0)

        self.screen.blit(self.mini_surface, (mx, my))
        pygame.draw.rect(self.screen, (100, 100, 105),
                         (mx, my, self.mini_w, self.mini_h), 1)

        # Viewport rectangle on minimap
        vl, vt = self.screen_to_world(0, 0)
        vr, vb = self.screen_to_world(SCREEN_W, SCREEN_H)
        vx = mx + max(0, int(vl * sx))
        vy = my + max(0, int(vt * sy))
        vw = min(self.mini_w - (vx - mx), int((vr - vl) * sx))
        vh = min(self.mini_h - (vy - my), int((vb - vt) * sy))
        if vw > 0 and vh > 0:
            pygame.draw.rect(self.screen, (255, 255, 255), (vx, vy, vw, vh), 1)

    def draw_pop_graph(self):
        gw, gh = 320, 110
        gx = SCREEN_W - gw - 20
        gy = SCREEN_H - gh - 20

        panel = pygame.Surface((gw, gh), pygame.SRCALPHA)
        panel.fill(COL_HUD_BG)
        self.screen.blit(panel, (gx, gy))
        pygame.draw.rect(self.screen, (60, 60, 65), (gx, gy, gw, gh), 1)

        if len(self.pred_history) < 2:
            return

        max_pop = max(max(self.pred_history), max(self.prey_history), 1)
        pad_top = 22
        pad_bot = 4

        def plot(history, color):
            n = len(history)
            pts = []
            for i, val in enumerate(history):
                px = gx + int(i / max(n - 1, 1) * (gw - 1))
                py = gy + pad_top + int((1 - val / max_pop) * (gh - pad_top - pad_bot))
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, color, False, pts, 2)

        plot(self.pred_history, COL_PREDATOR)
        plot(self.prey_history, COL_PREY)

        # Labels
        pred_n = self.pred_history[-1]
        prey_n = self.prey_history[-1]
        self.screen.blit(
            self.font.render(f"Predators: {pred_n}", True, COL_PREDATOR),
            (gx + 8, gy + 4))
        self.screen.blit(
            self.font.render(f"Prey: {prey_n}", True, COL_PREY),
            (gx + 150, gy + 4))

    def draw_stats(self):
        # Right side stats (like Pezzza: timer, ticks, frame time)
        elapsed = time.time() - self.start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        timer_str = f"{minutes:02d}:{seconds:02d}"

        rx = SCREEN_W - 200
        ry = 20 + self.mini_h + 15

        self.screen.blit(self.font_timer.render(timer_str, True, (50, 50, 55)),
                         (rx, ry))
        ry += 36
        self.screen.blit(self.font.render(f"Ticks: {self.frame}", True, (80, 80, 85)),
                         (rx, ry))
        ry += 20
        ft = self.clock.get_time()
        self.screen.blit(self.font.render(f"Frame time: {ft:.1f}ms", True, (80, 80, 85)),
                         (rx, ry))
        ry += 20
        self.screen.blit(self.font.render(f"Speed: {self.sim_speed}x", True, (80, 80, 85)),
                         (rx, ry))
        ry += 20
        self.screen.blit(self.font.render(f"Zoom: {self.cam_zoom:.1f}x", True, (80, 80, 85)),
                         (rx, ry))

    def draw_hud(self):
        pw, ph = 300, 55
        px, py = 15, 15

        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(COL_HUD_BG)
        self.screen.blit(panel, (px, py))
        pygame.draw.rect(self.screen, (60, 60, 65), (px, py, pw, ph), 1)

        self.screen.blit(
            self.font.render(
                f"Predators: {len(self.predators)}   Prey: {len(self.prey)}   "
                f"Foliage: {len(self.foliage)}", True, COL_HUD_TEXT),
            (px + 8, py + 8))
        self.screen.blit(
            self.font.render(
                f"Gen: {self.generation}   FPS: {self.clock.get_fps():.0f}"
                f"{'   [PAUSED]' if self.paused else ''}", True, COL_HUD_DIM),
            (px + 8, py + 28))

    def draw_controls(self):
        text = "SPACE=pause  V=rays  R=restart  H=reset zoom  UP/DOWN=speed  Scroll=zoom  RightDrag=pan"
        tw = self.font.size(text)[0] + 16
        cx = (SCREEN_W - tw) // 2
        cy = SCREEN_H - 30
        panel = pygame.Surface((tw, 22), pygame.SRCALPHA)
        panel.fill((30, 30, 35, 150))
        self.screen.blit(panel, (cx, cy))
        self.screen.blit(self.font.render(text, True, (120, 120, 125)),
                         (cx + 8, cy + 4))

    def draw_selected_zoomed(self):
        if not self.selected or not self.selected.alive:
            self.selected = None
            return
        c = self.selected
        sx, sy = self.world_to_screen(c.x, c.y)
        ix, iy = int(sx), int(sy)
        ring_r = int((CREATURE_RADIUS + 5) * self.cam_zoom)
        pygame.draw.circle(self.screen, (255, 255, 255), (ix, iy), ring_r, 2)

        pw, ph = 210, 95
        px = min(ix + 25, SCREEN_W - pw - 10)
        py = max(iy - ph - 10, 10)
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(COL_HUD_BG)
        self.screen.blit(panel, (px, py))
        pygame.draw.rect(self.screen, (80, 80, 85), (px, py, pw, ph), 1)

        kind = "Predator" if c.etype == T_PREDATOR else "Prey"
        lines = [
            f"{kind}  Age: {c.age}",
            f"Energy: {c.energy:.0f}/{c.max_energy:.0f}",
            f"HP: {c.hp}/{c.max_hp}" if c.max_hp > 0 else "No HP (starve only)",
            f"Speed: {c.speed:.1f}  FOV: {math.degrees(c.fov):.0f}deg",
        ]
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, COL_HUD_TEXT),
                             (px + 8, py + 8 + i * 21))

    def _in_view(self, wx, wy, margin=30):
        """Check if a world point is visible on screen (with margin)."""
        sx, sy = self.world_to_screen(wx, wy)
        m = margin * self.cam_zoom
        return -m < sx < SCREEN_W + m and -m < sy < SCREEN_H + m

    def draw(self):
        self.screen.fill(BG_COLOR)
        z = self.cam_zoom
        spacing = GRID_SPACING

        # Grid lines — skip if spacing on screen < 8px
        screen_spacing = spacing * z
        if screen_spacing >= 8:
            w_left, w_top = self.screen_to_world(0, 0)
            w_right, w_bot = self.screen_to_world(SCREEN_W, SCREEN_H)
            start_gx = int(w_left // spacing) * spacing
            start_gy = int(w_top // spacing) * spacing
            for gx in range(int(start_gx), int(w_right) + spacing, spacing):
                sx, _ = self.world_to_screen(gx, 0)
                sx = int(sx)
                if 0 <= sx <= SCREEN_W:
                    pygame.draw.line(self.screen, GRID_LINE_COLOR,
                                     (sx, 0), (sx, SCREEN_H))
            for gy in range(int(start_gy), int(w_bot) + spacing, spacing):
                _, sy = self.world_to_screen(0, gy)
                sy = int(sy)
                if 0 <= sy <= SCREEN_H:
                    pygame.draw.line(self.screen, GRID_LINE_COLOR,
                                     (0, sy), (SCREEN_W, sy))

        # Foliage — simple dots
        for f in self.foliage:
            if f.alive and self._in_view(f.x, f.y):
                sx, sy = self.world_to_screen(f.x, f.y)
                r = max(1, int(FOLIAGE_RADIUS * z))
                pygame.draw.circle(self.screen, f.color, (int(sx), int(sy)), r)

        # Creatures — LOD: blob sprites when close, colored dots when far
        screen_r = CREATURE_RADIUS * z
        use_blobs = screen_r >= LOD_DOT_THRESHOLD

        for creature_list in (self.prey, self.predators):
            for c in creature_list:
                if not c.alive or not self._in_view(c.x, c.y, 50):
                    continue
                if use_blobs:
                    c.draw_zoomed(self.screen, self.show_vision, self, z,
                                   is_selected=(c is self.selected))
                else:
                    # Fast dot rendering
                    sx, sy = self.world_to_screen(c.x, c.y)
                    dr = max(1, int(screen_r))
                    pygame.draw.circle(self.screen, c.color,
                                       (int(sx), int(sy)), dr)

        # UI overlays (not affected by camera)
        self.draw_hud()
        self.draw_minimap()
        self.draw_stats()
        self.draw_pop_graph()
        self.draw_controls()
        self.draw_selected_zoomed()

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            for _ in range(self.sim_speed):
                self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Simulation().run()
