import math

import \
    numpy.ma
from numba import jit
import numpy as np
import matplotlib.pyplot as plt

@jit(nopython=True)
def fill_edge_psi(psi, nx, ny, delta, x1, y1, q_in, mi):
    y_ny = ny * delta
    y_y1 = y1 * delta
    q_out = q_in * (y_ny**3 - y_y1**3 - 3 * y_y1 * y_ny**2 + 3 * y_y1**2 * y_ny) / y_ny**3

    # Edge A
    for y in range(y1, ny + 1):
        y_y = delta * y
        psi[0, y] = q_in / (2 * mi) * (y_y**3 / 3 - 0.5 * y_y**2 * (y_y1 + y_ny) + y_y * y_y1 * y_ny)

    # Edge C
    for y in range(ny + 1):
        y_y = delta * y
        psi[nx, y] = q_out / (2 * mi) * (y_y**3 / 3 - 0.5 * y_y**2 * y_ny) + q_in * y_y1**2 * (3 * y_ny - y_y1) / (12 * mi)

    # Edge B
    for x in range(1, nx):
        psi[x, ny] = psi[0, ny]

    # Edge D
    for x in range(x1, nx):
        psi[x, 0] = psi[0, y1]

    # Edge E
    for y in range(1, y1 + 1):
        psi[x1, y] = psi[0, y1]

    # Edge F
    for x in range(1, x1 + 1):
        psi[x, y1] = psi[0, y1]

@jit(nopython=True)
def fill_edge_zeta(zeta, psi, nx, ny, delta, x1, y1, q_in, mi):
    y_ny = ny * delta
    y_y1 = y1 * delta
    q_out = q_in *  (y_ny**3 - y_y1**3 - 3 * y_y1 * y_ny**2 + 3 * y_y1**2 * y_ny) / y_ny**3

    # Edge A
    for y in range(y1, ny + 1):
        y_y = delta * y
        zeta[0, y] = q_in / (2 * mi) * (2 * y_y - y_y1 - y_ny)

    # Edge C
    for y in range(ny + 1):
        y_y = delta * y
        zeta[nx, y] = q_out / (2 * mi) * (2 * y_y - y_ny)

    # Edge B
    for x in range(1, nx):
        zeta[x, ny] = 2 / delta**2 * (psi[x, ny - 1] - psi[x, ny])

    # Edge D
    for x in range(x1, nx):
        zeta[x, 0] = 2 / delta**2 * (psi[x, 1] - psi[x, 0])

    # Edge E
    for y in range(1, y1):
        zeta[x1, y] = 2 / delta**2 * (psi[x1 + 1, y] - psi[x1, y])

    # Edge F
    for x in range(1, x1):
        zeta[x, y1] = 2 / delta**2 * (psi[x, y1 + 1] - psi[x, y1])

    # Node at the intersection of E and F
    zeta[x1, y1] = 0.5 * (zeta[x1 - 1, y1] + zeta[x1, y1 - 1])

def relaxate(nx, ny, delta, x1, y1, q_in, mi, rho, iteration_limit, omega):
    # next_iteration and gamma isolated to functions because printing floats breaks @jit

    @jit(nopython = True)
    def next_iteration(psi, zeta):
        omega_i = omega(iteration)
        for x in range(1, nx):
            for y in range(1, ny):
                if x == nx or x == 0 or y == ny or y == 0 or (x <= x1 and y <= y1):
                    continue

                psi[x, y] = 0.25 * (psi[x + 1, y] + psi[x - 1, y] + psi[x, y + 1] + psi[x, y - 1] - delta**2 * zeta[x, y])
                zeta[x, y] = 0.25 * (zeta[x + 1, y] + zeta[x - 1, y] + zeta[x, y + 1] + zeta[x, y - 1]) -  omega_i * rho / (16 * mi) * ((psi[x, y + 1] - psi[x, y - 1]) * (zeta[x + 1, y] - zeta[x - 1, y]) - (psi[x + 1, y] - psi[x - 1, y]) * (zeta[x, y + 1] - zeta[x, y - 1]))

        fill_edge_zeta(zeta, psi, nx, ny, delta, x1, y1, q_in, mi)

    # @jit(nopython = True)
    def gamma():
        the_sum = 0
        for x in range(1, nx):
            the_sum += psi[x + 1, y1 + 2] + psi[x - 1, y1 + 2] + psi[x, y1 + 3] + psi[x, y1 + 1] - 4 * psi[x, y1 + 2] - delta**2 * zeta[x, y1 + 2]
        return the_sum

    psi = np.zeros((nx + 1, ny + 1), dtype = np.float_)
    zeta = np.zeros((nx + 1, ny + 1), dtype = np.float_)
    fill_edge_psi(psi, nx, ny, delta, x1, y1, q_in, mi)
    fill_edge_zeta(zeta, psi, nx, ny, delta, x1, y1, q_in, mi)

    for iteration in range(iteration_limit):
        next_iteration(psi, zeta)
        print(f"iteration:  {iteration}\tgamma: {gamma()}")

    return psi, zeta

@jit(nopython=True)
def velocity_map_x(nx, ny, delta, x1, y1, psi):
    velocity_x = np.zeros((nx + 1, ny + 1), dtype = np.float_)
    
    for x in range(1, nx):
        for y in range(1, ny):
            if x == nx or x == 0 or y == ny or y == 0 or (x <= x1 and y <= y1):
                continue

            velocity_x[x, y] = (psi[x, y + 1] - psi[x, y]) / delta

    return velocity_x

@jit(nopython=True)
def velocity_map_y(nx, ny, delta, x1, y1, psi):
    velocity_y = np.zeros((nx + 1, ny + 1), dtype = np.float_)
    
    for x in range(1, nx):
        for y in range(1, ny):
            if x == nx or x == 0 or y == ny or y == 0 or (x <= x1 and y <= y1):
                continue

            velocity_y[x, y] = - (psi[x + 1, y] - psi[x, y]) / delta

    return velocity_y

@jit(nopython=True)
def get_values_range(nx, ny, x1, y1, nodes):
    min_value = nodes[x1, y1]
    max_value = nodes[x1, y1]

    for x in range(1, nx):
        for y in range(1, ny):
            if x == nx or x == 0 or y == ny or y == 0 or (x <= x1 and y <= y1):
                continue

            min_value = min(min_value, nodes[x, y])
            max_value = max(max_value, nodes[x, y])

    return min_value, max_value

def draw_contour(nodes, values_range, path, number_of_levels = 40):
    levels = np.linspace(values_range[0], values_range[1], number_of_levels).tolist()

    plt.clf()
    plt.pcolor(nodes.transpose(), vmin = values_range[0], vmax = values_range[1])
    plt.contour(nodes.transpose(), levels = levels, colors = ['#000000'], linestyles = 'solid')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(path, dpi = 200)

def draw_nodes(nodes, values_range, path):
    plt.clf()
    plt.pcolor(nodes.transpose(), vmin = values_range[0], vmax = values_range[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(path, dpi = 200)

def main():
    delta = 0.01
    rho = 1
    mi = 1
    nx = 200
    ny = 90
    x1 = 50
    y1 = 55
    iteration_limit = 20000

    @jit(nopython=True)
    def omega(iteration):
        return 0 if iteration < 2000 else 0

    psi, zeta = relaxate(nx, ny, delta, x1, y1, -1000, mi, rho, iteration_limit, omega)
    velocity_x = velocity_map_x(nx, ny, delta, x1, y1, psi)
    velocity_y = velocity_map_y(nx, ny, delta, x1, y1, psi)
    draw_contour(psi, get_values_range(nx, ny, x1, y1, psi), "psi_q_minus1000.png")
    draw_contour(zeta, get_values_range(nx, ny, x1, y1, zeta), "zeta_q_minus1000.png")
    draw_nodes(velocity_x, get_values_range(nx, ny, x1, y1, velocity_x), "velocity_x_q_minus1000")
    draw_nodes(velocity_y, get_values_range(nx, ny, x1, y1, velocity_y), "velocity_y_q_minus1000")

    psi, zeta = relaxate(nx, ny, delta, x1, y1, -4000, mi, rho, iteration_limit, omega)
    velocity_x = velocity_map_x(nx, ny, delta, x1, y1, psi)
    velocity_y = velocity_map_y(nx, ny, delta, x1, y1, psi)
    draw_contour(psi, get_values_range(nx, ny, x1, y1, psi), "psi_q_minus4000.png")
    draw_contour(zeta, get_values_range(nx, ny, x1, y1, zeta), "zeta_q_minus4000.png")
    draw_nodes(velocity_x, get_values_range(nx, ny, x1, y1, velocity_x), "velocity_x_q_minus4000")
    draw_nodes(velocity_y, get_values_range(nx, ny, x1, y1, velocity_y), "velocity_y_q_minus4000")

    psi, zeta = relaxate(nx, ny, delta, x1, y1, 4000, mi, rho, iteration_limit, omega)
    velocity_x = velocity_map_x(nx, ny, delta, x1, y1, psi)
    velocity_y = velocity_map_y(nx, ny, delta, x1, y1, psi)
    draw_contour(psi, get_values_range(nx, ny, x1, y1, psi), "psi_q_4000.png")
    draw_contour(zeta, get_values_range(nx, ny, x1, y1, zeta), "zeta_q_4000.png")
    draw_nodes(velocity_x, get_values_range(nx, ny, x1, y1, velocity_x), "velocity_x_q_4000")
    draw_nodes(velocity_y, get_values_range(nx, ny, x1, y1, velocity_y), "velocity_y_q_4000")

if __name__ == '__main__':
    main()
