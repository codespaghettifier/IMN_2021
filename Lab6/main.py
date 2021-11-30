import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt

def get_equations_matrices(nx, ny, delta, v1, v2, v3, v4, epsilon, rho):
    n = (nx + 1) * (ny + 1)

    a = np.zeros(5 * n, dtype=np.float_)
    ja = np.zeros(5 * n, dtype=np.int_)
    ia = np.full(n + 1, -1, dtype=np.int_)
    b = np.zeros(n, dtype=np.float_)

    k = -1
    for l in range(n):
        y = l // (nx + 1)
        x = l - y * (nx + 1)

        is_edge = True if x == 0 or x == nx  or y == 0 or y == ny else False
        vb = v1 if x == 0 else v2 if y == ny else v3 if x == nx else v4 if y == 0 else 0

        b[l] = vb if is_edge else -rho(x, y)

        ia[l] = -1

        if l - nx - 1 >= 0 and not is_edge:
            k += 1
            if ia[l] < 0:
                ia[l] = k
            a[k] = epsilon(l) / delta**2
            ja[k] = l - nx - 1

        if l - 1 >= 0 and not is_edge:
            k += 1
            if ia[l] < 0:
                ia[l] = k
            a[k] = epsilon(l) / delta**2
            ja[k] = l - 1

        k += 1
        if ia[l] < 0:
            ia[l] = k
        if not is_edge:
            a[k] = - (2 * epsilon(l) + epsilon(l + 1) + epsilon(l + nx + 1)) / delta**2
        else:
            a[k] = 1
        ja[k] = l

        if l < n and not is_edge:
            k += 1
            a[k] = epsilon(l + 1) / delta**2
            ja[k] = l + 1

        if l <  n - nx - 1 and not is_edge:
            k += 1
            a[k] = epsilon(l + nx + 1) / delta**2
            ja[k] = l + nx + 1
    ia[n] = k + 1

    csr = csr_matrix((a, ja, ia), shape=(n, n))
    return csr, b

def generate_equations_and_save_them_to_file(nx, ny, delta, v1, v2, v3, v4, epsilon, rho, path):
    n = (nx + 1) * (ny + 1)

    a = np.zeros(5 * n, dtype=np.float_)
    ja = np.zeros(5 * n, dtype=np.int_)
    ia = np.full(n + 1, -1, dtype=np.int_)
    b = np.zeros(n, dtype=np.float_)


    lija = np.empty(0, dtype=np.float_)
    lijb = np.empty(0, dtype=np.float_)

    k = -1
    for l in range(n):
        y = l // (nx + 1)
        x = l - y * (nx + 1)

        is_edge = True if x == 0 or x == nx  or y == 0 or y == ny else False
        vb = v1 if x == 0 else v2 if y == ny else v3 if x == nx else v4 if y == 0 else 0

        b[l] = vb if is_edge else -rho(x, y)

        ia[l] = -1

        if l - nx - 1 >= 0 and not is_edge:
            k += 1
            if ia[l] < 0:
                ia[l] = k
            a[k] = epsilon(l) / delta**2
            ja[k] = l - nx - 1

        if l - 1 >= 0 and not is_edge:
            k += 1
            if ia[l] < 0:
                ia[l] = k
            a[k] = epsilon(l) / delta**2
            ja[k] = l - 1

        k += 1
        if ia[l] < 0:
            ia[l] = k
        if not is_edge:
            a[k] = - (2 * epsilon(l) + epsilon(l + 1) + epsilon(l + nx + 1)) / delta**2
        else:
            a[k] = 1
        ja[k] = l

        if l < n and not is_edge:
            k += 1
            a[k] = epsilon(l + 1) / delta**2
            ja[k] = l + 1

        if l <  n - nx - 1 and not is_edge:
            k += 1
            a[k] = epsilon(l + nx + 1) / delta**2
            ja[k] = l + nx + 1

        # new_lija_row = np.array([l, x, y, a[l]])
        lija = np.append(lija, [l, x, y, a[l]])
        # new_lijb_row = np.array([l, x, y, b[l]])
        lijb = np.append(lijb, [l, x, y, b[l]])

    ia[n] = k + 1

    lija = lija.reshape((n, 4))
    lijb = lijb.reshape((n, 4))

    with open(path, "w") as file:
        file.write("Matrix A, CSR:\n")
        file.write("l\ti[l]\tj[l]\ta[l]\n")
        for l in range(n):
            file.write(str(lija[l][0]) + "\t" + str(lija[l][1]) + "\t" + str(lija[l][2]) + "\t" + str(lija[l][3]) + "\n")

        file.write("\n\nVector b:\n")
        file.write("l\ti[l]\tj[l]\tb[l]\n")
        for l in range(n):
            file.write(str(lijb[l][0]) + "\t" + str(lijb[l][1]) + "\t" + str(lijb[l][2]) + "\t" + str(lijb[l][3]) + "\n")

def draw_nodes(nodes, path):
    plt.clf()
    plt.pcolor(nodes)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(path, dpi = 200)

def main():
    delta = 0.1
    nx = 4
    ny = 4
    epsilon1 = 1
    epsilon2 = 1
    v1 = 10
    v2 = -10
    v3 = 10
    v4 = -10
    n = (nx + 1) * (ny + 1)

    def rho_task_2_3_and_5(x, y):
        return  0

    def epsilon(l):
        y = l // (nx + 1)
        x = l - y * (nx + 1)
        return epsilon1 if x <= nx / 2 else epsilon2

    # Task 2.3
    generate_equations_and_save_them_to_file(nx, ny, delta, v1, v2, v3, v4, epsilon, rho_task_2_3_and_5, "matrix_A_vector_b.dat")

    # Task 2.5a
    nx = ny = 50
    csr, b = get_equations_matrices(nx, ny, delta, v1, v2, v3, v4, epsilon, rho_task_2_3_and_5)
    v, exit_code = gmres(csr, b, tol=10**(-8), atol=10**(-8), restart=500, maxiter=500)
    draw_nodes(v.reshape((nx + 1, ny + 1)), "task_2_5_a.png")

    # Task 2.5b
    nx = ny = 100
    csr, b = get_equations_matrices(nx, ny, delta, v1, v2, v3, v4, epsilon, rho_task_2_3_and_5)
    v, exit_code = gmres(csr, b, tol=10**(-8), atol=10**(-8), restart=500, maxiter=500)
    draw_nodes(v.reshape((nx + 1, ny + 1)), "task_2_5_b.png")

    # Task 2.5c
    nx = ny = 200
    csr, b = get_equations_matrices(nx, ny, delta, v1, v2, v3, v4, epsilon, rho_task_2_3_and_5)
    v, exit_code = gmres(csr, b, tol=10**(-8), atol=10**(-8), restart=500, maxiter=500)
    draw_nodes(v.reshape((nx + 1, ny + 1)), "task_2_5_c.png")

    #Task 2.6
    nx = ny = 100
    v1 = v2 = v3 = v4 = 0
    x_max = delta * nx
    y_max = delta * ny
    sigma = x_max / 10

    def rho_task_2_6(x, y):
        def rho1(x, y):
            return math.e**(-(x * delta - 0.25 * x_max)**2 / sigma**2 - (y * delta - 0.5 * y_max)**2 / sigma**2)

        def rho2(x, y):
            return -math.e**(-(x * delta - 0.75 * x_max)**2 / sigma**2 - (y * delta - 0.5 * y_max)**2 / sigma**2)

        return rho1(x, y) + rho2(x, y)

    # Task 2.6a
    epsilon1 = 1
    epsilon2 = 1
    csr, b = get_equations_matrices(nx, ny, delta, v1, v2, v3, v4, epsilon, rho_task_2_6)
    v, exit_code = gmres(csr, b, tol=10**(-8), atol=10**(-8), restart=500, maxiter=500)
    draw_nodes(v.reshape((nx + 1, ny + 1)), "task_2_6_a.png")

    # Task 2.6a
    epsilon1 = 1
    epsilon2 = 2
    csr, b = get_equations_matrices(nx, ny, delta, v1, v2, v3, v4, epsilon, rho_task_2_6)
    v, exit_code = gmres(csr, b, tol=10**(-8), atol=10**(-8), restart=500, maxiter=500)
    draw_nodes(v.reshape((nx + 1, ny + 1)), "task_2_6_b.png")

    # Task 2.6a
    epsilon1 = 1
    epsilon2 = 10
    csr, b = get_equations_matrices(nx, ny, delta, v1, v2, v3, v4, epsilon, rho_task_2_6)
    v, exit_code = gmres(csr, b, tol=10**(-8), atol=10**(-8), restart=500, maxiter=500)
    draw_nodes(v.reshape((nx + 1, ny + 1)), "task_2_6_c.png")

if __name__ == '__main__':
    main()
