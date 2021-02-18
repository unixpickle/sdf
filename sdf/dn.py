import itertools
import torch

_min = torch.minimum
_max = torch.maximum


def union(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, "_k", None)
            if K is None:
                d1 = _min(d1, d2)
            else:
                h = (0.5 + 0.5 * (d2 - d1) / K).clamp(0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m - K * h * (1 - h)
        return d1

    return f


def difference(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, "_k", None)
            if K is None:
                d1 = _max(d1, -d2)
            else:
                h = (0.5 - 0.5 * (d2 + d1) / K).clamp(0, 1)
                m = d1 + (-d2 - d1) * h
                d1 = m + K * h * (1 - h)
        return d1

    return f


def intersection(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, "_k", None)
            if K is None:
                d1 = _max(d1, d2)
            else:
                h = (0.5 - 0.5 * (d2 - d1) / K).clamp(0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m + K * h * (1 - h)
        return d1

    return f


def blend(a, *bs, k=0.5):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, "_k", None)
            d1 = K * d2 + (1 - K) * d1
        return d1

    return f


def negate(other):
    def f(p):
        return -other(p)

    return f


def dilate(other, r):
    def f(p):
        return other(p) - r

    return f


def erode(other, r):
    def f(p):
        return other(p) + r

    return f


def shell(other, thickness):
    def f(p):
        return other(p).abs() - thickness / 2

    return f


def repeat(other, spacing, count=None, padding=0):
    count_th = torch.tensor(count) if count is not None else None
    spacing_th = torch.tensor(spacing)

    def neighbors(dim, padding, spacing):
        try:
            padding = [padding[i] for i in range(dim)]
        except Exception:
            padding = [padding] * dim
        try:
            spacing = [spacing[i] for i in range(dim)]
        except Exception:
            spacing = [spacing] * dim
        for i, s in enumerate(spacing):
            if s == 0:
                padding[i] = 0
        axes = [list(range(-p, p + 1)) for p in padding]
        return list(itertools.product(*axes))

    def f(p):
        q = p / torch.where(
            spacing_th == 0, torch.ones_like(spacing_th), spacing_th
        ).to(p)
        if count_th is None:
            index = q.round()
        else:
            count_dev = count_th.to(index.device)
            index = q.round().clamp(-count_dev, count_dev)

        offsets = neighbors(p.shape[-1], padding, spacing)
        indices = torch.cat([index + n for n in offsets])
        A = other(p - spacing_th.to(p) * indices).view(len(offsets), -1)
        return A.min(0)

    return f
