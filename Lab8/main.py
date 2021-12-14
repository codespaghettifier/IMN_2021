import math
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def velocity_map(nx, ny, x1, x2, y1, delta, stream):
    v_x = np.zeros((nx + 1, ny + 1))
    v_y = np.zeros((nx + 1, ny + 1))

    for x in range(1, x1):
        for y in range(1, ny):
            v_x[x, y] = (stream[x, y + 1] - stream[x, y - 1]) / (2 * delta)
            v_y[x, y] = -(stream[x + 1, y] - stream[x - 1, y]) / (2 * delta)

    for x in range(x1, x2 + 1):
        for y in range(y1 + 1, ny):
            v_x[x, y] = (stream[x, y + 1] - stream[x, y - 1]) / (2 * delta)
            v_y[x, y] = -(stream[x + 1, y] - stream[x - 1, y]) / (2 * delta)

    for x in range(x2 + 1, nx):
        for y in range(1, ny):
            v_x[x, y] = (stream[x, y + 1] - stream[x, y - 1]) / (2 * delta)
            v_y[x, y] = -(stream[x + 1, y] - stream[x - 1, y]) / (2 * delta)

    for y in range(ny + 1):
        v_x[0, y] = v_x[1, y]
        v_y[0, y] = v_y[1, y]
        v_x[nx, y] = v_x[nx - 1, y]
        v_y[nx, y] = v_y[nx - 1, y]

    return v_x, v_y

@njit
def velocity_module_map(nx, ny, v_x, v_y):
    mod = np.zeros((nx + 1, ny + 1))
    for x in range(nx + 1):
        for y in range(ny + 1):
            mod[x, y] = (v_x[x, y]**2 + v_y[x, y]**2)**0.5
    return mod

def read_stream_func(nx, ny, path):
    stream = np.zeros((nx + 1, ny + 1))
    with open(path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            words = line.split()
            words = [int(words[0]), int(words[1]), float(words[2])]
            stream[words[0], words[1]] = words[2]
    return stream

@njit
def a_d(nx, ny, x1, x2, y1, delta, sigma, xa, ya, d, psi, iterations):
    v_x, v_y = velocity_map(nx, ny, x1, x2, y1, delta, psi)
    v_mod = velocity_module_map(nx, ny, v_x, v_y)
    delta_t = delta / (4 * np.amax(v_mod))
    u0 = np.zeros((nx + 1, ny + 1))

    u0_t = np.zeros((5, nx + 1, ny + 1))

    for x in range(nx + 1):
        for y in range(ny + 1):
            u0[x, y] = 1 / (2 * math.pi * sigma**2) * math.e**(-((x * delta - xa)**2 + (y * delta - ya)**2) / (2 * sigma**2))

    c = np.zeros(iterations)
    x_sr = np.zeros(iterations)

    for iteration in range(iterations):
        u1 = u0.copy()
        for _ in range(20):
            u1_r = np.zeros((nx + 1, ny + 1))
            for x in range(nx + 1):
                for y in range(1, ny):
                    if x1 <= x <= x2 and y <= y1:
                        continue

                    x_prev = x - 1 if x - 1 >= 0 else nx
                    x_next = x + 1 if x + 1 <= nx else 0

                    u1_r[x, y] = 1 / (1 + (2 * d * delta_t) / delta**2) * (u0[x, y]
                        - 0.5 * delta_t * v_x[x, y] * ((u0[x_next, y] - u0[x_prev, y]) / (2 * delta) + (u1[x_next, y] - u1[x_prev, y]) / (2 * delta))
                        - 0.5 * delta_t * v_y[x, y] * ((u0[x, y + 1] - u0[x, y - 1]) / (2 * delta) + (u1[x, y + 1] - u1[x, y - 1]) / (2 * delta))
                        + 0.5 * delta_t * d * ((u0[x_next, y] + u0[x_prev, y] + u0[x, y + 1] + u0[x, y - 1] - 4 * u0[x, y]) / delta**2 + (u1[x_next, y] + u1[x_prev, y] + u1[x, y + 1] + u1[x, y - 1]) / delta**2)
                        )
            u1 = u1_r
        u0 = u1

        if iteration % (iterations // 5) > 0 and iteration > 0:
            u0_t[iteration // (iterations // 5), :, :] = u0

        c[iteration] = np.sum(u0) * delta**2
        x_sr[iteration] = np.sum(np.multiply(np.sum(u0, 1), np.arange(nx + 1) * delta)) * delta**2

    return c, x_sr, u0_t

def draw_nodes(nodes, path):
    plt.clf()
    plt.axes().set_aspect('equal')
    plt.pcolor(nodes.transpose())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(path, dpi = 200)

def main():
    nx = 400
    ny = 90
    x1 = 200
    x2 = 210
    y1 = 50
    delta = 0.01
    sigma = 10 * delta
    xa = 0.45
    ya = 0.45

    stream = read_stream_func(nx, ny, "psi.dat")

    v_x, v_y = velocity_map(nx, ny, x1, x2, y1, delta, stream)
    v_mod = velocity_module_map(nx, ny, v_x, v_y)
    delta_t = delta / (4 * np.amax(v_mod))

    # D = 0
    c, x_sr, u_t = a_d(nx, ny, x1, x2, y1, delta, sigma, xa, ya, 0, stream, 10000)

    plt.clf()
    plt.plot([i * delta_t for i in range(len(c))], x_sr, "k-", label = "x_sr(t)")
    plt.plot([i * delta_t for i in range(len(c))], c, "r-", label = "c(t)")
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("c(t), x_sr(t)")
    plt.savefig("density_D0.png", dpi = 200)

    for i in range(5):
        draw_nodes(u_t[i], f"density_map_D0_0_{(i + 1) * 2}tmax.png")

    # D = 0.1
    c, x_sr, u_t = a_d(nx, ny, x1, x2, y1, delta, sigma, xa, ya, 0.1, stream, 10000)

    plt.clf()
    plt.plot([i * delta_t for i in range(len(c))], x_sr, "k-", label = "x_sr(t)")
    plt.plot([i * delta_t for i in range(len(c))], c, "r-", label = "c(t)")
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("c(t), x_sr(t)")
    plt.savefig("density_D01.png", dpi = 200)

    for i in range(5):
        draw_nodes(u_t[i], f"density_map_D01_0_{(i + 1) * 2}tmax.png")

if __name__ == '__main__':
    main()
