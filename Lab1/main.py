import math
import matplotlib.pyplot as plt

def write_results_to_file(file, results):
    for xy in results:
        file.write(str(xy[0]) + "\t" + str(xy[1]) + "\n")
    file.write("\n\n")

def euler(y0, func, t_min, t_max, delta_t):
    xy = [(t_min, y0)]
    t = t_min
    y = y0
    while t + delta_t <= t_max:
        y = y + delta_t * func(t, y)
        t += delta_t
        xy.append((t, y))

    return xy

def rk2(y0, func, t_min, t_max, delta_t):
    xy = [(t_min, y0)]
    t = t_min
    y = y0
    while t + delta_t <= t_max:
        k1 = func(t, y)
        k2 = func(t + delta_t, y + delta_t * k1)
        y = y + delta_t * 0.5 * (k1 + k2)
        t += delta_t
        xy.append((t, y))

    return xy

def rk4(y0, func, t_min, t_max, delta_t):
    xy = [(t_min, y0)]
    t = t_min
    y = y0
    while t + delta_t <= t_max:
        k1 = func(t, y)
        k2 = func(t + delta_t * 0.5, y + delta_t * 0.5 * k1)
        k3 = func(t + delta_t * 0.5, y + delta_t * 0.5 * k2)
        k4 = func(t + delta_t, y + delta_t * k3)
        y = y + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += delta_t
        xy.append((t, y))

    return xy

def rk4_rlc(q0, i0, func_v, t_min, t_max, delta_t, r, l, c, omega_v):
    xy = [(t_min, q0)]
    t = t_min
    q = q0
    i = i0
    while t + delta_t < t_max:
        kq1 = i
        ki1 = func_v(omega_v, t) / l - 1 / (l * c) * q - r / l * i
        kq2 = i + delta_t * 0.5 * ki1
        ki2 = func_v(omega_v, t + delta_t * 0.5) / l - 1 / (l * c) * (q + delta_t * 0.5 * kq1) - r / l * (i + delta_t * 0.5 * ki1)
        kq3 = i + delta_t * 0.5 * ki2
        ki3 = func_v(omega_v, t + delta_t * 0.5) / l - 1 / (l * c) * (q + delta_t * 0.5 * kq2) - r / l * (i + delta_t * 0.5 * ki2)
        kq4 = i + delta_t * 0.5 * ki3
        ki4 = func_v(omega_v, t + delta_t) / l - 1 / (l * c) * (q + delta_t * kq3) - r / l * (i + delta_t * ki3)
        q = q + delta_t / 6 * (kq1 + 2 * kq2 + 2 * kq3 + kq4)
        i = i + delta_t / 6 * (ki1 + 2 * ki2 + 2 * ki3 + ki4)
        t += delta_t
        xy.append((t, q))

    return xy


def main():
    def func_f(t, y):
        return -y

    def analytical(t):
        return  math.pow(math.e, -t)
    
    def plot3(xy1, xy2, xy3):
        plt.plot([xy[0] for xy in xy1], [xy[1] for xy in xy1], "ro-", label = "delta_t = 0.01", linewidth = 1, markersize = 4)
        plt.plot([xy[0] for xy in xy2], [xy[1] for xy in xy2], "go-", label = "delta_t = 0.1", linewidth = 1, markersize = 4)
        plt.plot([xy[0] for xy in xy3], [xy[1] for xy in xy3], "bo-", label = "delta_t = 1", linewidth = 1, markersize = 4)

    # Euler
    xy_euler001 = euler(1, func_f, 0, 5, 0.01)
    xy_euler01 = euler(1, func_f, 0, 5, 0.1)
    xy_euler1 = euler(1, func_f, 0, 5, 1)

    # with open("euler.dat", "w") as file:
    #     write_results_to_file(file, xy_euler001)
    #     write_results_to_file(file, xy_euler01)
    #     write_results_to_file(file, xy_euler1)

    plt.clf()
    plot3(xy_euler001, xy_euler01, xy_euler1)
    plt.plot([x / 100 for x in range(0, 500)], [analytical(x / 100) for x in range(0, 500)], "k-", label = "analitycznie")
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.savefig("euler.png", dpi = 200)

    xy_euler001_delta = [(xy[0], xy[1] - analytical(xy[0])) for xy in xy_euler001]
    xy_euler01_delta = [(xy[0], xy[1] - analytical(xy[0])) for xy in xy_euler01]
    xy_euler1_delta = [(xy[0], xy[1] - analytical(xy[0])) for xy in xy_euler1]

    # with open("euler_delta.dat", "w") as file:
    #     write_results_to_file(file, xy_euler001_delta)
    #     write_results_to_file(file, xy_euler01_delta)
    #     write_results_to_file(file, xy_euler1_delta)

    plt.clf()
    plot3(xy_euler001_delta, xy_euler01_delta, xy_euler1_delta)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("delta_y = y_numerycznie - y_analitycznie")
    plt.savefig("euler_delta.png", dpi = 200)

    # RK2
    xy_rk2_001 = rk2(1, func_f, 0, 5, 0.01)
    xy_rk2_01 = rk2(1, func_f, 0, 5, 0.1)
    xy_rk2_1 = rk2(1, func_f, 0, 5, 1)

    # with open("rk2.dat", "w") as file:
    #     write_results_to_file(file, xy_rk2_001)
    #     write_results_to_file(file, xy_rk2_01)
    #     write_results_to_file(file, xy_rk2_1)

    xy_rk2_001_delta = [(xy[0], abs(xy[1] - analytical(xy[0]))) for xy in xy_rk2_001]
    xy_rk2_01_delta = [(xy[0], abs(xy[1] - analytical(xy[0]))) for xy in xy_rk2_01]
    xy_rk2_1_delta = [(xy[0], abs(xy[1] - analytical(xy[0]))) for xy in xy_rk2_1]

    plt.clf()
    plot3(xy_rk2_001, xy_rk2_01, xy_rk2_1)
    plt.plot([x / 100 for x in range(0, 500)], [analytical(x / 100) for x in range(0, 500)], "k-", label = "analitycznie")
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.savefig("rk2.png", dpi = 200)

    # with open("rk2_delta.dat", "w") as file:
    #     write_results_to_file(file, xy_rk2_001_delta)
    #     write_results_to_file(file, xy_rk2_01_delta)
    #     write_results_to_file(file, xy_rk2_1_delta)

    plt.clf()
    plot3(xy_rk2_001_delta, xy_rk2_01_delta, xy_rk2_1_delta)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("delta_y = y_numerycznie - y_analitycznie")
    plt.savefig("rk2_delta.png", dpi = 200)

    # RK4
    xy_rk4_001 = rk4(1, func_f, 0, 5, 0.01)
    xy_rk4_01 = rk4(1, func_f, 0, 5, 0.1)
    xy_rk4_1 = rk4(1, func_f, 0, 5, 1)

    # with open("rk4.dat", "w") as file:
    #     write_results_to_file(file, xy_rk4_001)
    #     write_results_to_file(file, xy_rk4_01)
    #     write_results_to_file(file, xy_rk4_1)

    plt.clf()
    plot3(xy_rk4_001, xy_rk4_01, xy_rk4_1)
    plt.plot([x / 100 for x in range(0, 500)], [analytical(x / 100) for x in range(0, 500)], "k-", label = "analitycznie")
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.savefig("rk4.png", dpi = 200)

    xy_rk4_001_delta = [(xy[0], abs(xy[1] - analytical(xy[0]))) for xy in xy_rk4_001]
    xy_rk4_01_delta = [(xy[0], abs(xy[1] - analytical(xy[0]))) for xy in xy_rk4_01]
    xy_rk4_1_delta = [(xy[0], abs(xy[1] - analytical(xy[0]))) for xy in xy_rk4_1]

    # with open("rk4_delta.dat", "w") as file:
    #     write_results_to_file(file, xy_rk4_001_delta)
    #     write_results_to_file(file, xy_rk4_01_delta)
    #     write_results_to_file(file, xy_rk4_1_delta)

    plt.clf()
    plot3(xy_rk4_001_delta, xy_rk4_01_delta, xy_rk4_1_delta)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("delta_y = y_numerycznie - y_analitycznie")
    plt.savefig("rk4_delta.png", dpi = 200)

    #RLC
    def func_v(omega_v, t):
        return 10 * math.sin(omega_v * t)
    
    q0 = 0
    i0 = 0
    r = 100
    l = 0.1
    c = 0.001
    omega_0 = 1 / math.sqrt(l * c)
    t0 = 2 * math.pi / omega_0
    t_min = 0
    t_max = 4 * t0
    delta_t = 0.0001
    
    xy_rlc_05 = rk4_rlc(q0, i0, func_v, t_min, t_max, delta_t, r, l, c, 0.5 * omega_0)
    xy_rlc_08 = rk4_rlc(q0, i0, func_v, t_min, t_max, delta_t, r, l, c, 0.8 * omega_0)
    xy_rlc_1 = rk4_rlc(q0, i0, func_v, t_min, t_max, delta_t, r, l, c, omega_0)
    xy_rlc_12 = rk4_rlc(q0, i0, func_v, t_min, t_max, delta_t, r, l, c, 1.2 * omega_0)
    
    # with open("rk4_rlc.dat", "w") as file:
    #     write_results_to_file(file, xy_rlc_05)
    #     write_results_to_file(file, xy_rlc_08)
    #     write_results_to_file(file, xy_rlc_1)
    #     write_results_to_file(file, xy_rlc_12)
        
    plt.clf()
    plt.plot([xy[0] for xy in xy_rlc_05], [xy[1] for xy in xy_rlc_05], "k-", label = "omega_v = 0.5 omega_0", linewidth = 1)
    plt.plot([xy[0] for xy in xy_rlc_08], [xy[1] for xy in xy_rlc_08], "r-", label = "omega_v = 0.8 omega_0", linewidth = 1)
    plt.plot([xy[0] for xy in xy_rlc_1], [xy[1] for xy in xy_rlc_1], "b-", label = "omega_v = omega_0", linewidth = 1)
    plt.plot([xy[0] for xy in xy_rlc_12], [xy[1] for xy in xy_rlc_12], "g-", label = "omega_v = 1.2 omega_0", linewidth = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("Q(t)")
    plt.savefig("rk4_rlc.png", dpi = 200)

if __name__ == '__main__':
    main()
