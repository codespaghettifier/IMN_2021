import math
import matplotlib.pyplot as plt

def global_relaxation(nx, ny, vs, delta, epsilon, rho, omega):
    vn = [[]]
    for x in range(1, nx):
        vn.append([])
        vn[x].append(vs[x][0])
        for y in range(1, ny):
           vn[x].append(0.25 * (vs[x + 1][y] + vs[x - 1][y] + vs[x][y + 1] + vs[x][y - 1] + delta**2 / epsilon * rho(x, y)))
        vn[x].append(vs[x][ny])
    vn.append([])

    for y in range(ny + 1):
        vn[0].append(vn[1][y])
        vn[nx].append(vn[nx - 1][y])

    for x in range(nx + 1):
        for y in range(ny + 1):
            vn[x][y] = (1 - omega) * vs[x][y] + omega * vn[x][y]

    return vn

def local_relaxation(nx, ny, vs, delta, epsilon, rho, omega):
    vn = [[y for y in x] for x in vs]
    for x in range(1, nx):
        for y in range(1, ny):
            vn[x][y] = (1 - omega) * vn[x][y] + omega * 0.25 * (vn[x + 1][y] + vn[x - 1][y] + vs[x][y + 1] + vn[x][y - 1] + delta**2 / epsilon * rho(x, y))

    for y in range(ny + 1):
        vn[0][y] = vn[1][y]
        vn[nx][y] = vn[nx - 1][y]

    return vn

def functional_integral(nx, ny, v, delta, rho):
    the_sum = 0
    for x in range(nx):
        for y in range(ny):
            the_sum += 0.5 * ((v[x + 1][y] - v[x][y]) / delta)**2 + 0.5 * ((v[x][y + 1] - v[x][y]) / delta)**2 - rho(x, y) * v[x][y]
    the_sum *= delta**2
    return the_sum

def get_start_nodes_map(nx, ny, v_start, v1, v2):
    vs = [[v_start for y in range(ny + 1)] for x in range(nx + 1)]
    for x in range(nx + 1):
        vs[x][0] = v1
        vs[x][ny] = v2
    return vs

def relaxation(nx, ny, v_start, v1, v2, delta, epsilon, rho, omega, relaxation_method, tolerance):
    vs = get_start_nodes_map(nx, ny, v_start, v1, v2)
    vs_integral = functional_integral(nx, ny, vs, delta, rho)
    integrals = [vs_integral]

    it = 0
    while True:
        vn = relaxation_method(nx, ny, vs, delta, epsilon, rho, omega)
        vn_integral = functional_integral(nx, ny, vn, delta, rho)
        integrals.append(vn_integral)

        if abs((vn_integral - vs_integral) / vs_integral) < tolerance:
            break

        else:
            it += 1
            print(it, "\t", abs((vn_integral - vs_integral) / vs_integral))

        vs = vn
        vs_integral = vn_integral

    return vn, integrals


def get_err_map(nx, ny, v, delta, epsilon, rho):
    return [[(v[x + 1][y] + v[x - 1][y] + v[x][y + 1] + v[x][y - 1] - 4 * v[x][y]) / delta**2 + rho(x, y) / epsilon for y in range(1, ny)] for x in range(1, nx)]

def swap_xy(nodes):
    swapped = []
    for y in range(len(nodes[0])):
        swapped.append([])
        for x in range(len(nodes)):
            swapped[y].append(nodes[x][y])
    return swapped

def draw_nodes(nodes, path):
    plt.clf()
    plt.pcolor(swap_xy(nodes))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(path, dpi = 200)

def main():
    epsilon = 1
    delta = 0.1
    nx = 150
    ny = 100
    v_start = 0
    v1 = 10
    v2 = 0
    x_max = delta * nx
    y_max = delta * ny
    sigmax = 0.1 * x_max
    sigmay = 0.1 * y_max
    tolerance = 10**(-8)

    def rho1(x, y):
        return math.e**(-(x * delta - 0.35 * x_max)**2 / sigmax**2 - (y * delta - 0.5 * y_max)**2 / sigmay**2)

    def rho2(x, y):
        return -math.e**(-(x * delta - 0.65 * x_max)**2 / sigmax**2 - (y * delta - 0.5 * y_max)**2 / sigmay**2)

    def rho(x, y):
        return  rho1(x, y) + rho2(x, y)

    global_v06, global_integrals06 = relaxation(nx, ny, v_start, v1, v2, delta, epsilon, rho, 0.6, global_relaxation, tolerance)
    err_v06 = get_err_map(nx, ny, global_v06, delta, epsilon, rho)
    draw_nodes(global_v06, "global_relaxation_omega06.png")
    draw_nodes(err_v06, "global_relaxation_omega06_error.png")

    global_v1, global_integrals1 = relaxation(nx, ny, v_start, v1, v2, delta, epsilon, rho, 1, global_relaxation, tolerance)
    err_v1 = get_err_map(nx, ny, global_v1, delta, epsilon, rho)
    draw_nodes(global_v1, "global_relaxation_omega1.png")
    draw_nodes(err_v1, "global_relaxation_omega1_error.png")

    plt.clf()
    plt.plot(global_integrals06, 'ro-', label='omega = 0.6', linewidth=1, markersize=4)
    plt.plot(global_integrals1, 'bo-', label='omega = 1', linewidth=1, markersize=4)
    plt.legend(loc = "upper right")
    plt.xlabel("S(it)")
    plt.ylabel("it")
    plt.savefig("global_relaxation_integrals.png", dpi = 200)

    local_v1, local_integrals1 = relaxation(nx, ny, v_start, v1, v2, delta, epsilon, rho, 1, local_relaxation, tolerance)
    local_v14, local_integrals14 = relaxation(nx, ny, v_start, v1, v2, delta, epsilon, rho, 1.4, local_relaxation, tolerance)
    local_v18, local_integrals18 = relaxation(nx, ny, v_start, v1, v2, delta, epsilon, rho, 1.8, local_relaxation, tolerance)
    local_v19, local_integrals19 = relaxation(nx, ny, v_start, v1, v2, delta, epsilon, rho, 1.9, local_relaxation, tolerance)

    plt.clf()
    plt.plot(local_integrals1, 'ro-', label='omega = 1', linewidth=1, markersize=4)
    plt.plot(local_integrals14, 'bo-', label='omega = 1.4', linewidth=1, markersize=4)
    plt.plot(local_integrals18, 'go-', label='omega = 1.8', linewidth=1, markersize=4)
    plt.plot(local_integrals19, 'yo-', label='omega = 1.9', linewidth=1, markersize=4)
    plt.legend(loc = "upper right")
    plt.xlabel("S(it)")
    plt.ylabel("it")
    plt.savefig("local_relaxation_integrals.png", dpi = 200)

if __name__ == '__main__':
    main()
