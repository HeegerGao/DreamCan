import cv2
import numpy as np
import pygame
import pymunk
import shapely.geometry as sg
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d

WINDOW = 512
SIM_HZ = 100
CONTROL_HZ = 10
K_P, K_V = 100, 20
GOAL_POSE = np.array([256, 256, np.pi / 4])


def to_pygame(p):
    return round(p[0]), round(p[1])


def light_color(color):
    c = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    return SpaceDebugColor(r=c[0], g=c[1], b=c[2], a=c[3])


class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface):
        self.surface = surface
        super().__init__()

    def draw_circle(self, pos, angle, radius, outline_color, fill_color):
        p = to_pygame(pos)
        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius - 4), 0)

    def draw_segment(self, a, b, color):
        pygame.draw.aalines(self.surface, color.as_int(), False, [to_pygame(a), to_pygame(b)])

    def draw_fat_segment(self, a, b, radius, outline_color, fill_color):
        p1, p2 = to_pygame(a), to_pygame(b)
        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r <= 2:
            return
        orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
        if orthog[0] == 0 and orthog[1] == 0:
            return
        scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
        orthog = [round(orthog[0] * scale), round(orthog[1] * scale)]
        points = [
            (p1[0] - orthog[0], p1[1] - orthog[1]),
            (p1[0] + orthog[0], p1[1] + orthog[1]),
            (p2[0] + orthog[0], p2[1] + orthog[1]),
            (p2[0] - orthog[0], p2[1] - orthog[1]),
        ]
        pygame.draw.polygon(self.surface, fill_color.as_int(), points)
        pygame.draw.circle(self.surface, fill_color.as_int(), (round(p1[0]), round(p1[1])), round(radius))
        pygame.draw.circle(self.surface, fill_color.as_int(), (round(p2[0]), round(p2[1])), round(radius))

    def draw_polygon(self, verts, radius, outline_color, fill_color):
        ps = [to_pygame(v) for v in verts]
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps + [ps[0]])
        for i in range(len(verts)):
            self.draw_fat_segment(verts[i], verts[(i + 1) % len(verts)], 2, fill_color, fill_color)

    def draw_dot(self, size, pos, color):
        pygame.draw.circle(self.surface, color.as_int(), to_pygame(pos), round(size), 0)


def to_shapely(body, shapes):
    polys = []
    for shape in shapes:
        verts = [body.local_to_world(v) for v in shape.get_vertices()]
        polys.append(sg.Polygon(verts + [verts[0]]))
    return sg.MultiPolygon(polys)


class PushTEnv:
    def __init__(self, render_size=96, success_threshold=0.95):
        self.render_size = render_size
        self.success_threshold = success_threshold
        self._seed = None

    def seed(self, seed):
        self._seed = seed

    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        for a, b in [((5, 506), (5, 5)), ((5, 5), (506, 5)), ((506, 5), (506, 506)), ((5, 506), (506, 506))]:
            wall = pymunk.Segment(self.space.static_body, a, b, 2)
            wall.color = pygame.Color("LightGray")
            self.space.add(wall)
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        rs = np.random.RandomState(seed=self._seed)
        state = [rs.randint(50, 450), rs.randint(50, 450), rs.randint(100, 400), rs.randint(100, 400), rs.randn() * 2 * np.pi - np.pi]
        self.agent.position = state[:2]
        self.block.angle = state[4]
        self.block.position = state[2:4]
        self.space.step(1.0 / SIM_HZ)
        return self.render()

    def step(self, action):
        dt = 1.0 / SIM_HZ
        for _ in range(SIM_HZ // CONTROL_HZ):
            self.agent.velocity += (K_P * (action - self.agent.position) + K_V * (Vec2d(0, 0) - self.agent.velocity)) * dt
            self.space.step(dt)
        goal = self.goal_body()
        goal_geom = to_shapely(goal, self.block.shapes)
        coverage = goal_geom.intersection(to_shapely(self.block, self.block.shapes)).area / goal_geom.area
        return self.render(), coverage, coverage > self.success_threshold

    def state(self):
        return np.array(tuple(self.agent.position) + tuple(self.block.position) + (self.block.angle % (2 * np.pi),), dtype=np.float32)

    def goal_body(self):
        body = pymunk.Body(1, pymunk.moment_for_box(1, (50, 100)))
        body.position = GOAL_POSE[:2].tolist()
        body.angle = GOAL_POSE[2]
        return body

    def render(self):
        canvas = pygame.Surface((WINDOW, WINDOW))
        canvas.fill((255, 255, 255))
        goal = self.goal_body()
        for shape in self.block.shapes:
            pts = [to_pygame(goal.local_to_world(v)) for v in shape.get_vertices()]
            pygame.draw.polygon(canvas, pygame.Color("LightGreen"), pts + [pts[0]])
        self.space.debug_draw(DrawOptions(canvas))
        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        return cv2.resize(img, (self.render_size, self.render_size))

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, length=4):
        v1 = [(-length * scale / 2, scale), (length * scale / 2, scale), (length * scale / 2, 0), (-length * scale / 2, 0)]
        v2 = [(-scale / 2, scale), (-scale / 2, length * scale), (scale / 2, length * scale), (scale / 2, scale)]
        inertia = pymunk.moment_for_poly(1, vertices=v1)
        body = pymunk.Body(1, inertia + inertia)
        s1, s2 = pymunk.Poly(body, v1), pymunk.Poly(body, v2)
        for s in (s1, s2):
            s.color = pygame.Color("LightSlateGray")
            s.filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS())
        body.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, s1, s2)
        return body
