from os import name
import numpy as np
from numpy import ndim, pi, zeros_like
import matplotlib.pyplot as plt


class Line:
    def __init__(self, x, y, vx, vy):
        self.P = np.array((x, y), ndmin=2).T
        v = np.array((vx, vy), ndmin=2).T
        self.v = v / np.linalg.norm(v)
        self.n = np.array([self.v[1], -self.v[0]])

        self.R = np.r_[self.n.T, self.v.T]

    def project_to_line_coord(self, points):
        return self.R @ (points - self.P)


class Polygon:
    def __init__(self, n=5, radius=10, center=(0, 0), orientation=0):
        phi = np.linspace(0, 2 * pi, n + 1) + orientation
        self.points = np.array(center, ndmin=2).T + radius * np.array(
            [np.cos(phi), np.sin(phi)]
        )
        self.line_indices = np.array([[i, i + 1] for i in range(n - 1)] + [[n - 1, 0]])

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
    ax.plot(p[0], p[1], "*")
    ax.plot([p[0], q[0]], [p[1], q[1]])


if __name__ == "__main__":
    poly1 = Polygon(5, radius=15, center=(10, -10), orientation=np.pi / 12)
    poly2 = Polygon(6, radius=10, center=(20, -5), orientation=-np.pi / 12)

    line = Line(-5, -35, 0.5, 1)
    linerot = Line(0, 0, 0, 1)
    fig = plt.figure(num=1)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    p1r = poly1.transformed(line)  # line.transform_to_view(poly1.points)
    p2r = poly2.transformed(line)  # line.transform_to_view(poly2.points)
    plot_poly(ax1, poly1, "ro:")
    plot_poly(ax1, poly2, "bo-.")
    plot_line(ax1, line, 50)
    plot_poly(ax2, p1r, "ro:")
    plot_poly(ax2, p2r, "bo-.")
    plot_line(ax2, linerot, 50)

    x_of_lines, y_of_lines = p1r.line_segments()
    hit_indices = np.logical_xor(x_of_lines[0] < 0, x_of_lines[1] < 0)
    x_of_hit_lines = x_of_lines[:, hit_indices]
    y_of_hit_lines = y_of_lines[:, hit_indices]

    y_hit = (
        y_of_hit_lines[0] * x_of_hit_lines[1] - y_of_hit_lines[1] * x_of_hit_lines[0]
    ) / (x_of_hit_lines[1] - x_of_hit_lines[0])
    x_hit = zeros_like(y_hit)

    ax2.plot(x_hit, y_hit, "o")

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.grid(True)
    fig.tight_layout()
    plt.show()
