import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from numba import njit


@njit
def init_matrices(nx, ny, n, delta, delta_t, kb, kd, ta, tb, tc, td):
    a = np.zeros((n, n))
    b = np.zeros((n, n))
    c = np.zeros(n)

    for x in range(1, nx):
        for y in range(1, ny):
            l = x + y * (nx + 1)
            a[l, l - nx - 1] = a[l, l - 1] = a[l, l + 1] = a[l, l + nx + 1] = delta_t / (2 * delta ** 2)
            a[l, l] = -2 * delta_t / delta ** 2 - 1
            b[l, l - nx - 1] = b[l, l - 1] = b[l, l + 1] = b[l, l + nx + 1] = -delta_t / (2 * delta ** 2)
            b[l, l] = 2 * delta_t / delta ** 2 - 1

    for y in range(ny + 1):
        l = y * (nx + 1)
        a[l, l] = 1
        b[l, l] = 1
        c[l] = 0

    for y in range(ny + 1):
        l = nx + y * (nx + 1)
        a[l, l] = 1
        b[l, l] = 1
        c[l] = 0

    for x in range(1, nx):
        l = x + ny * (nx + 1)
        a[l, l - nx - 1] = -1 / (kb * delta)
        a[l, l] = 1 + 1 / (kb * delta)
        c[l] = tb
        b[l, :] = 0

    for x in range(1, nx):
        l = x
        a[l, l + nx + 1] = -1 / (kd * delta)
        a[l, l] = 1 + 1 / (kd * delta)
        c[l] = td
        b[l, :] = 0

    t = np.zeros(n)
    for y in range(ny + 1):
        t[y * (nx + 1)] = ta
        t[nx + y * (nx + 1)] = tc

    return a, b, c, t


def draw_nodes(nodes, path):
    plt.clf()
    plt.pcolor(nodes)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(path, dpi = 200)


@njit
def diffusion_map(t0, t1, delta_t):
    return (t1 - t0) / delta_t


def main():
    nx = 40
    ny = 40
    n = (nx + 1) * (ny + 1)
    delta = 1
    delta_t = 1
    ta = 40
    tb = 0
    tc = 30
    td = 0
    kb = 0.1
    kd = 0.6
    iteration_limit = 2000

    a, b, c, t = init_matrices(nx, ny, n, delta, delta_t, kb, kd, ta, tb, tc, td)
    lu, piv = scipy.linalg.lu_factor(a)
    map_iterations = [100, 200, 500, 1000, 2000]
    for iteration in range(iteration_limit + 1):
        d = (b @ t) + c
        t1 = scipy.linalg.lu_solve((lu, piv), d)

        if iteration in map_iterations:
            draw_nodes(t.reshape(nx + 1, ny + 1), f"temperature_map_it{iteration}.png")
            draw_nodes(diffusion_map(t, t1, delta_t).reshape(nx + 1, ny + 1), f"diffusion_map_it{iteration}.png")

        t = t1


if __name__ == '__main__':
    main()
