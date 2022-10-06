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

class View(Line):
    def __init__(self, x, y, vx, vy):
        super().__init__(x, y, vx, vy)
        self.R = np.r_[self.n.T, self.v.T]

    def transform_to_view(self, points):
        return self.R @ (points - self.P)


def polygon(n=5, radius=10, center=(0, 0), orientation=0):
    phi = np.linspace(0, 2 * pi, n + 1) + orientation
    p = np.array(center, ndmin=2).T + radius * np.array([np.cos(phi), np.sin(phi)])
    return p


def plot_poly(ax, poly, *args, **kwargs):
    ax.plot(poly[0], poly[1], *args, **kwargs)


def plot_line(ax, line, length):
    p = line.P
    q = line.P + length * line.v
    ax.plot(p[0], p[1], "*")
    ax.plot([p[0], q[0]], [p[1], q[1]])


if __name__ == "__main__":
    poly1 = polygon(5, radius=15, center=(10, -10), orientation=np.pi / 12)
    poly2 = polygon(6, radius=10, center=(20, -5), orientation=-np.pi / 12)
    allpolys = np.c_[poly1, poly2]
    line = View(-5, -35, 0.5, 1)
    linerot = View(0, 0, 0, 1)
    fig = plt.figure(num=1)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    p1r = line.transform_to_view(poly1)
    p2r = line.transform_to_view(poly2)
    plot_poly(ax1, poly1, "o:")
    plot_poly(ax1, poly2, "o-.")
    plot_line(ax1, line, 50)
    plot_poly(ax2, p1r, "o:")
    plot_poly(ax2, p2r, "o-.")
    plot_line(ax2, linerot, 50)

    vertices = np.array([[0,1],[1,2],[2,3],[3,4], [4,5], [3,0]])
    x_of_lines = p1r[0][vertices.flatten()].reshape(vertices.shape[0],2).T
    hit_indices = np.logical_xor(x_of_lines[0]<0, x_of_lines[1] <0)
    x_of_hit_lines = x_of_lines[:, hit_indices]
    y_of_lines = p1r[1][vertices.flatten()].reshape(vertices.shape[0],2).T
    y_of_hit_lines = y_of_lines[:, hit_indices]

    y_hit = (y_of_hit_lines[0]*x_of_hit_lines[1]-y_of_hit_lines[1]*x_of_hit_lines[0])/(x_of_hit_lines[1]-x_of_hit_lines[0])
    x_hit = zeros_like(y_hit)

    ax2.plot(x_hit, y_hit, 'o')

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.grid(True)
    fig.tight_layout()
    plt.show()

#

if 0:
    x_of_lines = p1r[0][np.array([0,1,1,2,2,3,3,4, 4,5, 4,3])].reshape(6,2).T
    hit_indices = np.logical_xor(x_of_lines[0]<0, x_of_lines[1] <0)
    x_of_hit_lines = x_of_lines[:, hit_indices]
    y_of_lines = p1r[1][np.array([0,1,1,2,2,3,3,4, 4,5, 4, 3])].reshape(6,2).T
    y_of_hit_lines = y_of_lines[:, hit_indices]

    # y1 = np.cross(y_of_hit_lines[:,0], x_of_hit_lines[:,0])/(x_of_hit_lines[1,0]-x_of_hit_lines[0,0])
    # y2 = np.cross(y_of_hit_lines[:,1], x_of_hit_lines[:,1])/(x_of_hit_lines[1,1]-x_of_hit_lines[0,1])
    #np.cross(y_of_hit_lines[:,0], x_of_hit_lines[:,0]) / (x_of_hit_lines[1]-x_of_hit_lines[0])
    y_hit = np.cross(y_of_hit_lines.T, x_of_hit_lines.T)/(x_of_hit_lines[1]-x_of_hit_lines[0])
    x_hit = zeros_like(y_hit)

    # poly1[1][np.array([0,3,3,4,4,5])].reshape(3,2)]
    # poly1[np.r_[np.ones(6,dtype=np.integer), np.zeros(6,dtype=np.integer)].reshape(2,6), np.array([0, 3, 3, 4, 4, 5])]