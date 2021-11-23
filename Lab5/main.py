import math
from numba import jit
import numpy as np
import matplotlib.pyplot as plt

@jit
def get_initialized_nodes(nx, ny, boundary_condition):
    nodes = np.zeros(((nx + 1), (ny + 1)), dtype = np.float_)
    for x in range(nx):
        nodes[x][0] = boundary_condition(x, 0)
        nodes[x][ny] = boundary_condition(x, ny)
    for y in range(ny):
        nodes[0][y] = boundary_condition(0, y)
        nodes[nx][y] = boundary_condition(nx, y)
    return nodes

@jit
def functional_integral(nodes, nx, ny, delta, step_size):
    the_sum = 0
    for x in range(0, nx - step_size + 1, step_size):
        for y in range(0, ny - step_size + 1, step_size):
            the_sum +=  ((nodes[x + step_size][y] - nodes[x][y] + nodes[x + step_size][y + step_size] - nodes[x][y + step_size]) / (2 * step_size * delta))**2 + ((nodes[x][y + step_size] - nodes[x][y] + nodes[x + step_size][y + step_size] - nodes[x + step_size][y]) / (2 * step_size * delta))**2
    return the_sum * (step_size * delta)**2 * 0.5

@jit
def relaxate_iteration(nodes, nx, ny, step_size):
    for x in range(step_size, nx, step_size):
        for y in range(step_size, ny, step_size):
            nodes[x][y] = 0.25 * (nodes[x + step_size][y] + nodes[x - step_size][y] + nodes[x][y + step_size] + nodes[x][y - step_size])

@jit
def relaxate(nodes, nx, ny, delta, step_size, tolerance):
    integrals = np.arange(1, dtype = np.float_)
    integral = functional_integral(nodes, nx, ny, delta, step_size)
    integrals[0] = integral
    new_nodes = nodes.copy()

    # it = 0
    while True:
        relaxate_iteration(new_nodes, nx, ny, step_size)
        new_integral = functional_integral(new_nodes, nx, ny, delta, step_size)
        integrals = np.append(integrals, new_integral)

        # print(f"iteration: {it}\tintegral change: {(new_integral - integral) / integral}")
        if abs((new_integral - integral) / integral) < tolerance:
            break

        integral = new_integral
        # it += 1

    return new_nodes, integrals

@jit
def interpolate(nodes, nx, ny, step_size):
    if step_size == 1:
        return

    for x in range(0, nx - step_size + 1, step_size):
        for y in range(0, ny - step_size + 1, step_size):
            nodes[x + int(step_size * 0.5)][y + int(step_size * 0.5)] = 0.25 * (nodes[x][y] + nodes[x + step_size][y] + nodes[x][y + step_size] + nodes[x + step_size][y + step_size])
            nodes[x + int(step_size * 0.5)][y] = 0.5 * (nodes[x][y] + nodes[x + step_size][y])
            nodes[x][y + int(step_size * 0.5)] = 0.5 * (nodes[x][y] + nodes[x][y + step_size])
            nodes[x + int(step_size * 0.5)][y + step_size] = 0.5 * (nodes[x][y + step_size] + nodes[x + step_size][y + step_size])
            nodes[x + step_size][y + int(step_size * 0.5)] = 0.5 * (nodes[x + step_size][y] + nodes[x + step_size][y + step_size])

    step_size = int(step_size * 0.5)
    interpolate(nodes, nx, ny, step_size)

def draw_nodes(nodes, path):
    plt.clf()
    plt.pcolor(nodes.transpose())
    plt.xlabel("x / (delta * k)")
    plt.ylabel("y / (delta * k)")
    plt.savefig(path, dpi = 200)

def get_nodes_subsample(nodes, nx, ny, step_size):
    new_nodes = np.zeros((int(nx / step_size) + 1, int(ny / step_size) + 1))
    for x in range(0, nx + 1, step_size):
        new_nodes[int(x / step_size)] = nodes[x][::step_size]

    return new_nodes

def main():
    delta = 0.2
    nx = 128
    ny = 128
    x_max = delta * nx
    y_max = delta * ny
    tolerance = 10**(-8)

    @jit
    def boundary_condition(x, y):
        if x == 0:
            return math.sin(math.pi * y * delta / y_max)
        elif y == ny:
            return -math.sin(2 * math.pi * x * delta/ x_max)
        elif x == nx:
            return math.sin(math.pi * y * delta/ y_max)
        elif y == 0:
            return math.sin(2 * math.pi * x * delta / x_max)
        else:
            raise ValueError("Point is not boundary")

    nodes = get_initialized_nodes(nx, ny, boundary_condition)
    integrals = []

    for step_size in [16, 8, 4, 2, 1]:
        nodes, integrals_k = relaxate(nodes, nx, ny, delta, step_size, tolerance)
        integrals.append(integrals_k)
        interpolate(nodes, nx, ny, step_size)
        draw_nodes(get_nodes_subsample(nodes, nx, ny, step_size), f"nodes_k{step_size}.png")

    plt.clf()
    markers = ['ko-', 'ro-', 'go-', 'bo-', 'yo-']
    step_sizes = [16, 8, 4, 2, 1]
    iterations = 1
    for i in range(len(integrals)):
        plt.plot([j for j in range(iterations, iterations + len(integrals[i]))], integrals[i], markers[i], label = f'S(it), k = {step_sizes[i]}, i_max = {len(integrals[i])}', linewidth = 1, markersize = 4)
        iterations += len(integrals[i])
    plt.xscale('log')
    plt.legend(loc = "upper right")
    plt.xlabel("it")
    plt.ylabel("S(it)")
    plt.savefig("integrals.png", dpi = 200)

if __name__ == '__main__':
    main()
