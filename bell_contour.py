import math
import numpy as np
import matplotlib.pyplot as plt

# Funciones proporcionadas

def bell_nozzle(k, aratio, Rt, l_percent):
    entrant_angle = -135
    ea_radian = math.radians(entrant_angle)

    if l_percent == 60:
        Lnp = 0.6
    elif l_percent == 80:
        Lnp = 0.8
    elif l_percent == 90:
        Lnp = 0.9
    else:
        Lnp = 0.8

    angles = find_wall_angles(aratio, Rt, l_percent)
    nozzle_length = angles[0]
    theta_n = angles[1]
    theta_e = angles[2]

    data_intervel = 100
    ea_start = ea_radian
    ea_end = -math.pi / 2
    angle_list = np.linspace(ea_start, ea_end, data_intervel)
    xe = []
    ye = []
    for i in angle_list:
        xe.append(1.5 * Rt * math.cos(i))
        ye.append(1.5 * Rt * math.sin(i) + 2.5 * Rt)

    ea_start = -math.pi / 2
    ea_end = theta_n - math.pi / 2
    angle_list = np.linspace(ea_start, ea_end, data_intervel)
    xe2 = []
    ye2 = []
    for i in angle_list:
        xe2.append(0.382 * Rt * math.cos(i))
        ye2.append(0.382 * Rt * math.sin(i) + 1.382 * Rt)

    Nx = 0.382 * Rt * math.cos(theta_n - math.pi / 2)
    Ny = 0.382 * Rt * math.sin(theta_n - math.pi / 2) + 1.382 * Rt
    Ex = Lnp * ((math.sqrt(aratio) - 1) * Rt) / math.tan(math.radians(15))
    Ey = math.sqrt(aratio) * Rt
    m1 = math.tan(theta_n)
    m2 = math.tan(theta_e)
    C1 = Ny - m1 * Nx
    C2 = Ey - m2 * Ex
    Qx = (C2 - C1) / (m1 - m2)
    Qy = (m1 * C2 - m2 * C1) / (m1 - m2)

    int_list = np.linspace(0, 1, data_intervel)
    xbell = []
    ybell = []
    for t in int_list:
        xbell.append(((1 - t) ** 2) * Nx + 2 * (1 - t) * t * Qx + (t ** 2) * Ex)
        ybell.append(((1 - t) ** 2) * Ny + 2 * (1 - t) * t * Qy + (t ** 2) * Ey)

    nye = [-y for y in ye]
    nye2 = [-y for y in ye2]
    nybell = [-y for y in ybell]

    return angles, (xe, ye, nye, xe2, ye2, nye2, xbell, ybell, nybell)

def find_wall_angles(ar, Rt, l_percent=80):
    aratio = [4, 5, 10, 20, 30, 40, 50, 100]
    theta_n_60 = [20.5, 20.5, 16.0, 14.5, 14.0, 13.5, 13.0, 11.2]
    theta_n_80 = [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5]
    theta_n_90 = [20.0, 21.0, 24.0, 27.0, 28.5, 29.5, 30.2, 32.0]
    theta_e_60 = [26.5, 28.0, 32.0, 35.0, 36.2, 37.1, 35.0, 40.0]
    theta_e_80 = [14.0, 13.0, 11.0, 9.0, 8.5, 8.0, 7.5, 7.0]
    theta_e_90 = [11.5, 10.5, 8.0, 7.0, 6.5, 6.0, 6.0, 6.0]

    f1 = ((math.sqrt(ar) - 1) * Rt) / math.tan(math.radians(15))

    if l_percent == 60:
        theta_n = theta_n_60
        theta_e = theta_e_60
        Ln = 0.6 * f1
    elif l_percent == 80:
        theta_n = theta_n_80
        theta_e = theta_e_80
        Ln = 0.8 * f1
    elif l_percent == 90:
        theta_n = theta_n_90
        theta_e = theta_e_90
        Ln = 0.9 * f1
    else:
        theta_n = theta_n_80
        theta_e = theta_e_80
        Ln = 0.8 * f1

    x_index, x_val = find_nearest(aratio, ar)
    if round(aratio[x_index], 1) == round(ar, 1):
        return Ln, math.radians(theta_n[x_index]), math.radians(theta_e[x_index])

    if x_index > 2:
        ar_slice = aratio[x_index - 2 : x_index + 2]
        tn_slice = theta_n[x_index - 2 : x_index + 2]
        te_slice = theta_e[x_index - 2 : x_index + 2]
        tn_val = interpolate(ar_slice, tn_slice, ar)
        te_val = interpolate(ar_slice, te_slice, ar)
    elif len(aratio) - x_index <= 1:
        ar_slice = aratio[x_index - 2 : len(x_index)]
        tn_slice = theta_n[x_index - 2 : len(x_index)]
        te_slice = theta_e[x_index - 2 : len(x_index)]
        tn_val = interpolate(ar_slice, tn_slice, ar)
        te_val = interpolate(ar_slice, te_slice, ar)
    else:
        ar_slice = aratio[0 : x_index + 2]
        tn_slice = theta_n[0 : x_index + 2]
        te_slice = theta_e[0 : x_index + 2]
        tn_val = interpolate(ar_slice, tn_slice, ar)
        te_val = interpolate(ar_slice, te_slice, ar)

    return Ln, math.radians(tn_val), math.radians(te_val)

def interpolate(x_list, y_list, x):
    if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
        raise ValueError("x_list must be in strictly ascending order!")
    intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
    slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]

    if x <= x_list[0]:
        return y_list[0]
    elif x >= x_list[-1]:
        return y_list[-1]
    else:
        i = bisect_left(x_list, x) - 1
        return y_list[i] + slopes[i] * (x - x_list[i])

def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# Función principal para calcular y graficar el contorno de la tobera
def calculate_and_plot_contour(Rt, Re):
    k = 1.21  # Ratio de calores específicos, típico
    aratio = (Re / Rt) ** 2
    l_percent = 80  # Porcentaje de longitud de la tobera

    angles, contour = bell_nozzle(k, aratio, Rt, l_percent)

    xe, ye, nye, xe2, ye2, nye2, xbell, ybell, nybell = contour

    plt.figure(figsize=(10, 6))
    plt.plot(xe, ye, label='Entrante de la garganta')
    plt.plot(xe2, ye2, label='Salida de la garganta')
    plt.plot(xbell, ybell, label='Contorno de la campana')
    plt.xlabel('Longitud axial (m)')
    plt.ylabel('Radio (m)')
    plt.title('Contorno de una tobera de campana')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    Rt = 0.02  # Radio de la garganta
    Re = 0.2   # Radio de salida
    calculate_and_plot_contour(Rt, Re)
