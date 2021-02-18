import functools
import numpy as np
import torch

from . import dn, d2, ease, mesh, torch_util as tu

# Constants

ORIGIN = np.array((0, 0, 0))

X = np.array((1, 0, 0))
Y = np.array((0, 1, 0))
Z = np.array((0, 0, 1))

UP = Z

# SDF Class

_ops = {}


class SDF3:
    def __init__(self, f):
        self.f = f

    def __call__(self, p):
        return self.f(p).reshape((-1, 1))

    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        raise AttributeError

    def __or__(self, other):
        return union(self, other)

    def __and__(self, other):
        return intersection(self, other)

    def __sub__(self, other):
        return difference(self, other)

    def k(self, k=None):
        self._k = k
        return self

    def generate(self, *args, **kwargs):
        return mesh.generate(self, *args, **kwargs)

    def save(self, path, *args, **kwargs):
        return mesh.save(path, self, *args, **kwargs)

    def show_slice(self, *args, **kwargs):
        return mesh.show_slice(self, *args, **kwargs)


def sdf3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))

    return wrapper


def op3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))

    _ops[f.__name__] = wrapper
    return wrapper


def op32(f):
    def wrapper(*args, **kwargs):
        return d2.SDF2(f(*args, **kwargs))

    _ops[f.__name__] = wrapper
    return wrapper


# Helpers


def _length(a):
    return torch.linalg.norm(a, dim=1)


def _normalize_np(a):
    return a / np.linalg.norm(a)


def _dot(a, b):
    return (a * b).sum(1)


def _perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError("zero vector")
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


_vec = tu.vec
_min = tu.torch_min
_max = tu.torch_max

# Primitives


@sdf3
def sphere(radius=1, center=ORIGIN):
    def f(p):
        return _length(p - tu.to_torch(p, center)) - radius

    return f


@sdf3
def plane(normal=UP, point=ORIGIN):
    normal = _normalize_np(normal)

    def f(p):
        return torch.mv(tu.to_torch(p, point) - p, tu.to_torch(p, normal))

    return f


@sdf3
def slab(x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(plane(X, (x0, 0, 0)))
    if x1 is not None:
        fs.append(plane(-X, (x1, 0, 0)))
    if y0 is not None:
        fs.append(plane(Y, (0, y0, 0)))
    if y1 is not None:
        fs.append(plane(-Y, (0, y1, 0)))
    if z0 is not None:
        fs.append(plane(Z, (0, 0, z0)))
    if z1 is not None:
        fs.append(plane(-Z, (0, 0, z1)))
    return intersection(*fs, k=k)


@sdf3
def box(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a = np.array(a)
        b = np.array(b)
        size = b - a
        center = a + size / 2
        return box(size, center)
    size = np.array(size)

    def f(p):
        q = (p - tu.to_torch(p, center)) - tu.to_torch(p, size / 2)
        return _length(_max(q, 0)) + _min(q.amax(1), 0)

    return f


@sdf3
def rounded_box(size, radius):
    size = np.array(size)

    def f(p):
        q = p.abs() - tu.to_torch(p, size / 2 + radius)
        return _length(_max(q, 0)) + _min(q.amax(1), 0) - radius

    return f


@sdf3
def wireframe_box(size_np, thickness_np):
    size_np = np.array(size_np)

    def g(a, b, c):
        return _length(_max(_vec(a, b, c), 0)) + _min(_max(a, _max(b, c)), 0)

    def f(p):
        size, thickness = tu.to_torch(p, size_np, thickness_np)
        p = p.abs() - size / 2 - thickness / 2
        q = (p + thickness / 2).abs() - thickness / 2
        px, py, pz = p[:, 0], p[:, 1], p[:, 2]
        qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
        return _min(_min(g(px, qy, qz), g(qx, py, qz)), g(qx, qy, pz))

    return f


@sdf3
def torus(r1, r2):
    def f(p):
        xy = p[:, [0, 1]]
        z = p[:, 2]
        a = _length(xy) - r1
        b = _length(_vec(a, z)) - r2
        return b

    return f


@sdf3
def capsule(a_np, b_np, radius):
    a_np = np.array(a_np)
    b_np = np.array(b_np)

    def f(p):
        a, b = tu.to_torch(p, a_np, b_np)
        pa = p - a
        ba = b - a
        h = (torch.mv(pa, ba) / torch.dot(ba, ba)).clamp(0, 1).reshape((-1, 1))
        return _length(pa - (ba * h)) - radius

    return f


@sdf3
def cylinder(radius):
    def f(p):
        return _length(p[:, [0, 1]]) - radius

    return f


@sdf3
def capped_cylinder(a_np, b_np, radius):
    a_np = np.array(a_np)
    b_np = np.array(b_np)

    def f(p):
        a, b = tu.to_torch(p, a_np, b_np)
        ba = b - a
        pa = p - a
        baba = torch.dot(ba, ba)
        paba = torch.mv(pa, ba).reshape((-1, 1))
        x = _length(pa * baba - ba * paba) - radius * baba
        y = (paba - baba * 0.5).abs() - baba * 0.5
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        x2 = x * x
        y2 = y * y * baba
        d = torch.where(
            _max(x, y) < 0,
            -_min(x2, y2),
            torch.where(x > 0, x2, 0) + torch.where(y > 0, y2, 0),
        )
        return torch.sign(d) * d.abs().sqrt() / baba

    return f


@sdf3
def rounded_cylinder(ra, rb, h):
    def f(p):
        d = _vec(_length(p[:, [0, 1]]) - ra + rb, p[:, 2].abs() - h / 2 + rb)
        return _min(_max(d[:, 0], d[:, 1]), 0) + _length(_max(d, 0)) - rb

    return f


@sdf3
def capped_cone(a_np, b_np, ra, rb):
    a_np = np.array(a_np)
    b_np = np.array(b_np)

    def f(p):
        a, b = tu.to_torch(p, a_np, b_np)
        rba = rb - ra
        baba = torch.dot(b - a, b - a)
        papa = _dot(p - a, p - a)
        paba = torch.mv(p - a, b - a) / baba
        x = (papa - paba * paba * baba).sqrt()
        cax = _max(0, x - torch.where(paba < 0.5, ra, rb))
        cay = (paba - 0.5).abs() - 0.5
        k = rba * rba + baba
        f = ((rba * (x - ra) + paba * baba) / k).clamp(0, 1)
        cbx = x - ra - f * rba
        cby = paba - f
        s = torch.where(torch.logical_and(cbx < 0, cay < 0), -1, 1)
        return (
            s * _min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba).sqrt()
        )

    return f


@sdf3
def rounded_cone(r1, r2, h):
    def f(p):
        q = _vec(_length(p[:, [0, 1]]), p[:, 2])
        b = (r1 - r2) / h
        a = np.sqrt(1 - b * b)
        k = q @ _vec(-b, a)
        c1 = _length(q) - r1
        c2 = _length(q - _vec(0, h)) - r2
        c3 = (q @ _vec(a, b)) - r1
        return torch.where(k < 0, c1, torch.where(k > a * h, c2, c3))

    return f


@sdf3
def ellipsoid(size_np):
    size_np = np.array(size_np)

    def f(p):
        size = tu.to_torch(p, size_np)
        k0 = _length(p / size)
        k1 = _length(p / (size * size))
        return k0 * (k0 - 1) / k1

    return f


@sdf3
def pyramid(h):
    def f(p):
        a = p[:, [0, 1]].abs() - 0.5
        w = a[:, 1] > a[:, 0]
        a[w] = a[:, [1, 0]][w]
        px = a[:, 0]
        py = p[:, 2]
        pz = a[:, 1]
        m2 = h * h + 0.25
        qx = pz
        qy = h * py - 0.5 * px
        qz = h * px + 0.5 * py
        s = _max(-qx, 0)
        t = ((qy - 0.5 * pz) / (m2 + 0.25)).clamp(0, 1)
        a = m2 * (qx + s) ** 2 + qy * qy
        b = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2
        d2 = torch.where(_min(qy, -qx * m2 - qy * 0.5) > 0, 0, _min(a, b))
        return ((d2 + qz * qz) / m2).sqrt() * torch.sign(_max(qz, -py))

    return f


# Platonic Solids


@sdf3
def tetrahedron(r):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        return (_max((x + y).abs() - z, (x - y).abs() + z) - 1) / np.sqrt(3)

    return f


@sdf3
def octahedron(r):
    def f(p):
        return (p.abs().sum(1) - r) * np.tan(np.radians(30))

    return f


@sdf3
def dodecahedron(r):
    x, y, z = _normalize_np(((1 + np.sqrt(5)) / 2, 1, 0))

    def f(p):
        p = (p / r).abs()
        a = torch.mv(p, tu.vec(x, y, z).to(p))
        b = torch.mv(p, tu.vec(z, x, y).to(p))
        c = torch.mv(p, tu.vec(y, z, x).to(p))
        q = (_max(_max(a, b), c) - x) * r
        return q

    return f


@sdf3
def icosahedron(r):
    r *= 0.8506507174597755
    x, y, z = _normalize_np(((np.sqrt(5) + 3) / 2, 1, 0))
    w = np.sqrt(3) / 3

    def f(p):
        p = (p / r).abs()
        a = torch.mv(p, tu.vec(x, y, z).to(p))
        b = torch.mv(p, tu.vec(z, x, y).to(p))
        c = torch.mv(p, tu.vec(y, z, x).to(p))
        d = torch.mv(p, tu.vec(w, w, w).to(p)) - x
        return _max(_max(_max(a, b), c) - x, d) * r

    return f


# Positioning


@op3
def translate(other, offset_np):
    def f(p):
        offset = tu.to_torch(p, offset_np)
        return other(p - offset)

    return f


@op3
def scale(other, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    m = min(x, min(y, z))

    def f(p):
        return other(p / tu.to_torch(p, s)) * tu.to_torch(p, m)

    return f


@op3
def rotate(other, angle, vector=Z):
    x, y, z = _normalize_np(vector)
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    matrix = np.array(
        [
            [m * x * x + c, m * x * y + z * s, m * z * x - y * s],
            [m * x * y - z * s, m * y * y + c, m * y * z + x * s],
            [m * z * x + y * s, m * y * z - x * s, m * z * z + c],
        ]
    ).T

    def f(p):
        return other(p @ tu.to_torch(p, matrix))

    return f


@op3
def rotate_to(other, a, b):
    a = _normalize_np(np.array(a))
    b = _normalize_np(np.array(b))
    dot = np.dot(b, a)
    if dot == 1:
        return other
    if dot == -1:
        return rotate(other, np.pi, _perpendicular(a))
    angle = np.arccos(dot)
    v = _normalize_np(np.cross(b, a))
    return rotate(other, angle, v)


@op3
def orient(other, axis):
    return rotate_to(other, UP, axis)


@op3
def circular_array(other, count, offset):
    other = other.translate(X * offset)
    da = 2 * np.pi / count

    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        d = torch.hypot(x, y)
        a = torch.atan2(y, x) % da
        d1 = other(_vec(torch.cos(a - da) * d, torch.sin(a - da) * d, z))
        d2 = other(_vec(torch.cos(a) * d, torch.sin(a) * d, z))
        return _min(d1, d2)

    return f


# Alterations


@op3
def elongate(other, size):
    def f(p):
        q = p.abs() - tu.to_torch(p, size)
        x = q[:, 0].reshape((-1, 1))
        y = q[:, 1].reshape((-1, 1))
        z = q[:, 2].reshape((-1, 1))
        w = _min(_max(x, _max(y, z)), 0)
        return other(_max(q, 0)) + w

    return f


@op3
def twist(other, k):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        c = torch.cos(k * z)
        s = torch.sin(k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))

    return f


@op3
def bend(other, k):
    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        c = torch.cos(k * x)
        s = torch.sin(k * x)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))

    return f


@op3
def bend_linear(other, p0_np, p1_np, v_np, e=ease.linear):
    p0_np = np.array(p0_np)
    p1_np = np.array(p1_np)
    v_np = -np.array(v_np)
    ab_np = p1_np - p0_np

    def f(p):
        p0, v, ab = tu.to_torch(p, p0_np, v_np, ab_np)
        t = (((p - p0) @ ab) / (ab @ ab)).clamp(0, 1)
        t = e(t).reshape((-1, 1))
        return other(p + t * v)

    return f


@op3
def bend_radial(other, r0_np, r1_np, dz_np, e=ease.linear):
    def f(p):
        r0, r1, dz = tu.to_torch(p, r0_np, r1_np, dz_np)
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        r = torch.hypot(x, y)
        t = ((r - r0) / (r1 - r0)).clamp(0, 1)
        z = z - dz * e(t)
        return other(_vec(x, y, z))

    return f


@op3
def transition_linear(f0, f1, p0_np=-Z, p1_np=Z, e=ease.linear):
    p0_np = np.array(p0_np)
    p1_np = np.array(p1_np)
    ab_np = p1_np - p0_np

    def f(p):
        p0, ab = tu.to_torch(p, p0_np, ab_np)
        d1 = f0(p)
        d2 = f1(p)
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1

    return f


@op3
def transition_radial(f0, f1, r0=0, r1=1, e=ease.linear):
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        r = torch.hypot(p[:, 0], p[:, 1])
        t = ((r - r0) / (r1 - r0)).clamp(0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1

    return f


@op3
def wrap_around(other, x0, x1, r=None, e=ease.linear):
    p0_np = X * x0
    p1_np = X * x1
    v_np = Y
    if r is None:
        r = np.linalg.norm(p1_np - p0_np) / (2 * np.pi)

    def f(p):
        p0, p1, v = tu.to_torch(p, p0_np, p1_np, v_np)
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        d = np.hypot(x, y) - r
        d = d.reshape((-1, 1))
        a = np.arctan2(y, x)
        t = (a + np.pi) / (2 * np.pi)
        t = e(t).reshape((-1, 1))
        q = p0 + (p1 - p0) * t + v * d
        q[:, 2] = z
        return other(q)

    return f


# 3D => 2D Operations


@op32
def slice(other):
    # TODO: support specifying a slice plane
    # TODO: probably a better way to do this
    s = slab(z0=-1e-9, z1=1e-9)
    a = other & s
    b = other.negate() & s

    def f(p):
        p = _vec(p[:, 0], p[:, 1], torch.zeros_like(p[:, 0]))
        A = a(p).reshape(-1)
        B = -b(p).reshape(-1)
        w = A <= 0
        A[w] = B[w]
        return A

    return f


# Common

union = op3(dn.union)
difference = op3(dn.difference)
intersection = op3(dn.intersection)
blend = op3(dn.blend)
negate = op3(dn.negate)
dilate = op3(dn.dilate)
erode = op3(dn.erode)
shell = op3(dn.shell)
repeat = op3(dn.repeat)
