from math import dist
import numpy as np
from numpy import pi, zeros_like
import matplotlib.pyplot as plt
import itertools


class Line:
    def __init__(self, x, y, vx, vy):
        self.P = np.array((x, y), ndmin=2).T
        v = np.array((vx, vy), ndmin=2).T
        self.v = v / np.linalg.norm(v)
        self.n = np.array([self.v[1], -self.v[0]])

        self.R = np.r_[self.n.T, self.v.T]

    def project_to_line_coord(self, points):
        return self.R @ (points - self.P)

    def project_from_line_coord(self, points):
        return (self.R.T @ points) + self.P

    def distance_to_point(self, point):
        distance = np.cross((point - self.P).T, self.v.T)[0]
        return np.abs(distance)


class Polygon:
    def __init__(self, n=5, radius=10, center=(0, 0), orientation=0):
        phi = np.linspace(0, 2 * pi, n + 1) + orientation
        self.points = np.array(center, ndmin=2).T + radius * np.array(
            [np.cos(phi), np.sin(phi)]
        )
        self.line_indices = np.array([[i, i + 1] for i in range(n - 1)] + [[n - 1, 0]])

        self.center = np.array(center, ndmin=2).T
        self.bounding_radius = radius

    def is_hit_by_line(self, line):
        distance = line.distance_to_point(self.center)
        is_hit = distance <= self.bounding_radius
        return is_hit

    def line_segments(self):
        ind_flat = self.line_indices.flatten()
        n = self.line_indices.shape[0]
        x_of_lines = self.points[0][ind_flat].reshape(n, 2).T
        y_of_lines = self.points[1][ind_flat].reshape(n, 2).T
        return x_of_lines, y_of_lines

    def transformed(self, view):
        new_poly = Polygon()
        new_poly.points = view.project_to_line_coord(self.points)
        new_poly.line_indices = self.line_indices.copy()
        return new_poly


def plot_poly(ax, poly, *args, **kwargs):
    x_of_lines, y_of_lines = poly.line_segments()
    ax.plot(x_of_lines, y_of_lines, *args, **kwargs)


def plot_line(ax, line, length):
    p = line.P
    q = line.P + length * line.v
    (l,) = ax.plot(p[0], p[1], "*", markersize=10)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=l.get_color())


if __name__ == "__main__":

    def rand(dist=1.0):
        ret = 2 * dist * (np.random.rand() - 0.5)
        # print(f"r = {ret}")
        return ret

    FIELDSIZE = 50
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    fig = plt.figure(num=1)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # line = Line(-5, -35, 0.1, 1)
    line = Line(rand(FIELDSIZE / 3), rand(FIELDSIZE / 3), rand(), rand())
    linerot = Line(0, 0, 0, 1)
    plot_line(ax1, line, FIELDSIZE)
    plot_line(ax2, linerot, FIELDSIZE)

    polys = [
        Polygon(
            np.random.randint(3, 10),
            radius=np.random.rand() * 10,
            center=(rand(FIELDSIZE), rand(FIELDSIZE)),
            orientation=rand(pi),
        )
        for i in range(100)
        # Polygon(5, radius=5, center=(14, -10), orientation=np.pi / 12),
        # Polygon(6, radius=4, center=(-5, -5), orientation=-np.pi / 12),
        # Polygon(6, radius=5, center=(25, -25), orientation=-np.pi / 12),
        # Polygon(6, radius=7, center=(-5, -15), orientation=-np.pi / 12),
        # Polygon(6, radius=3, center=(5, -30), orientation=-np.pi / 12),
    ]
    distance = np.inf
    for i, poly in enumerate(polys):
        if poly.is_hit_by_line(line):
            color = next(colors)
            poly_rot = poly.transformed(line)  # line.transform_to_view(poly1.points)
            plot_poly(ax1, poly, ":", color=color)
            plot_poly(ax2, poly_rot, ":", color=color)

            x_of_lines, y_of_lines = poly_rot.line_segments()
            hit_indices = np.logical_xor(x_of_lines[0] < 0, x_of_lines[1] < 0)
            x_of_hit_lines = x_of_lines[:, hit_indices]
            y_of_hit_lines = y_of_lines[:, hit_indices]

            y_hit = (
                y_of_hit_lines[0] * x_of_hit_lines[1]
                - y_of_hit_lines[1] * x_of_hit_lines[0]
            ) / (x_of_hit_lines[1] - x_of_hit_lines[0])
            x_hit = zeros_like(y_hit)
            ax2.plot(x_hit, y_hit, "ko", markersize=3)
            pos_dist = y_hit[y_hit > 0]
            if len(pos_dist):
                _distance = np.min(pos_dist)
                print(_distance)
                if _distance < distance:
                    distance = _distance
        else:
            poly_rot = poly.transformed(line)  # line.transform_to_view(poly1.points)
            plot_poly(ax1, poly, ":", color=(0.8, 0.8, 0.8))
            plot_poly(ax2, poly_rot, ":", color=(0.8, 0.8, 0.8))
    if distance < np.inf:
        print(f"distance = {distance}")
        ax2.plot(0, distance, "ko", markersize=12)
        p = line.project_from_line_coord(np.array([[0], [distance]]))
        ax1.plot(p[0], p[1], "ko", markersize=12)
    else:
        print("No hit")

    for ax in (ax1, ax2):
        ax.set_xlim((-FIELDSIZE - 10, FIELDSIZE + 10))
        ax.set_ylim((-FIELDSIZE - 10, FIELDSIZE + 10))
        ax.set_aspect("equal")
        ax.grid(True)
    fig.tight_layout()
    plt.show()
