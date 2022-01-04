import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def verlet(nx, nt, delta, delta_t, start_u, alpha, beta, xf=-1):
    u = np.empty((nx + 1, nt + 1))
    v = np.empty((nx + 1, nt + 1))
    a = np.zeros(nx + 1)

    u[:, 0] = start_u
    v[0, :] = v[nx, :] = 0
    v[:, 0] = 0

    af = np.zeros(nx + 1)
    if 0 < xf < nx:
        af[xf] = math.cos(50 * 0 / nt)
    a[1:nx] = (u[2:, 0] - 2 * u[1:nx, 0] + u[:nx - 1, 0]) / delta ** 2 - beta * (u[1:nx, 0] - u[1:nx, 0]) / delta_t + alpha * af[1:nx]

    for n in range(1, nt + 1):
        vp = v[:, n - 1] + delta_t / 2 * a
        u0 = u
        u[:, n] = u[:, n - 1] + vp * delta_t
        af = np.zeros(nx + 1)
        if 0 < xf < nx:
            af[xf] = math.cos(50 * n / nt)
        a[1:nx] = (u[2:, n] - 2 * u[1:nx, n] + u[:nx - 1, n]) / delta ** 2 - beta * (u[1:nx, n] - u0[1:nx, n - 1]) / delta_t + alpha * af[1:nx]
        v[:, n] = vp + delta_t / 2 * a

    return u, v


def draw_nodes(nodes, path, x_range=None, y_range=None):
    x_range = x_range if x_range is not None else (0, nodes.shape[1])
    y_range = y_range if y_range is not None else (0, nodes.shape[0])
    x = np.linspace(x_range[0], x_range[1], nodes.shape[1])
    y = np.linspace(y_range[0], y_range[1], nodes.shape[0])

    plt.clf()
    plt.pcolor(x, y, nodes)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(path, dpi=200)


@njit
def energy(u, v, delta):
    nx = u.shape[0] - 1
    return delta / 4 * (((u[1, :] - u[0, :]) / delta) ** 2 + ((u[nx, :] - u[nx - 1, :]) / delta) ** 2) + delta / 2 * np.sum(v[1:nx, :] ** 2 + ((u[2:, :] - u[:nx - 1, :]) / (2 * delta)) ** 2, 0)


def main():
    nx = 150
    nt = 1000
    delta = 0.1
    delta_t = 0.05
    xa = 7.5
    sigma = 0.5

    # start conditions
    start_u = math.e ** (-(np.linspace(0, nx * delta, num=(nx + 1)) - xa) ** 2 / (2 * sigma))
    start_u[0] = start_u[nx] = 0

    u_b0, v_b0 = verlet(nx, nt, delta, delta_t, start_u, 0, 0)
    e_b0 = energy(u_b0, v_b0, delta)
    draw_nodes(u_b0, "u_b0.png", x_range=(0, nt * delta_t), y_range=(0, nx * delta))

    u_b01, v_b01 = verlet(nx, nt, delta, delta_t, start_u, 0, 0.1)
    e_b01 = energy(u_b01, v_b01, delta)
    draw_nodes(u_b01, "u_b01.png", x_range=(0, nt * delta_t), y_range=(0, nx * delta))

    u_b1, v_b1 = verlet(nx, nt, delta, delta_t, start_u, 0, 1)
    e_b1 = energy(u_b1, v_b1, delta)
    draw_nodes(u_b1, "u_b1.png", x_range=(0, nt * delta_t), y_range=(0, nx * delta))

    plt.clf()
    plt.plot([i * delta_t for i in range(nt + 1)], e_b0, "k-", label="beta = 0")
    plt.plot([i * delta_t for i in range(nt + 1)], e_b01, "r-", label="beta = 0.1")
    plt.plot([i * delta_t for i in range(nt + 1)], e_b1, "b-", label="beta = 1")
    plt.legend(loc="upper right")
    plt.xlabel("t")
    plt.ylabel("E(t)")
    plt.savefig("e.png", dpi=200)

    start_u = np.zeros(nx + 1)
    u_a1_b1, v_a1_b1 = verlet(nx, nt, delta, delta_t, start_u, 1, 1, xf=25)
    e_a1_b1 = energy(u_a1_b1, v_a1_b1, delta)
    draw_nodes(u_a1_b1, "u_a1_b1.png", x_range=(0, nt * delta_t), y_range=(0, nx * delta))

    plt.clf()
    plt.plot([i * delta_t for i in range(nt + 1)], e_a1_b1, "k-")
    # plt.legend(loc="upper right")
    plt.xlabel("t")
    plt.ylabel("E(t)")
    plt.savefig("e_a1_b1.png", dpi=200)


if __name__ == '__main__':
    main()
