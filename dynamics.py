import math as kk
import matplotlib.pyplot as plt
import numpy as np
import Tabl as tb


euler = lambda arg, der: arg + der * dt


def atmo(H):
    """
    :param H: высота над уровнем моря[м];

    :return: T - температура[K], a - местная скорость звука[м/с], P - давление[Па], ro - плотность[кг/м^3].
    """
    # isa model 0 < H < 11000

    if H >= 11000:
        h_t = 11000
    elif H <= 0:
        h_t = 0
    else:
        h_t = H

    T = -h_t * 0.0065 + 288.15
    a = kk.sqrt(T * 401.874)

    if (11000 - H) >= 0:
        h_troup = 0
    elif (11000 - H) <= -9000:
        h_troup = -9000
    else:
        h_troup = 11000 - H

    par_e = kk.e ** (h_troup * 0.034163191 / T)

    T_pro = T * 0.0034704147
    T_st = T_pro ** (9.80665 / (0.0065 * 287.0531))

    P = par_e * (T_st * 101325)
    ro = par_e * (T_st / T_pro * 1.225)

    return T, a, P, ro


def quatmultiply(in1, in2):
    mult = [in2[0] * in1[0] - in2[1] * in1[1] - in2[2] * in1[2] - in2[3] * in1[3],
            in2[0] * in1[1] + in2[1] * in1[0] - in2[2] * in1[3] + in2[3] * in1[2],
            in2[0] * in1[2] + in2[1] * in1[3] + in2[2] * in1[0] - in2[3] * in1[1],
            in2[0] * in1[3] - in2[1] * in1[2] + in2[2] * in1[1] + in2[3] * in1[0]]

    return mult


def norm_quat(quats):

    mod_q = kk.sqrt(quats[0] ** 2 + quats[1] ** 2 + quats[2] ** 2 + quats[3] ** 2)
    quats = [quats[0] / mod_q, quats[1] / mod_q, quats[2] / mod_q, quats[3] / mod_q]

    return quats[0], quats[1], quats[2], quats[3]


def form_c_ib(quats):

    ci_b = np.zeros(9)
    ci_b[0] = quats[0] ** 2 + quats[1] ** 2 - quats[2] ** 2 - quats[3] ** 2
    ci_b[1] = 2 * (quats[1] * quats[2] - quats[0] * quats[3])
    ci_b[2] = 2 * (quats[3] * quats[1] + quats[2] * quats[0])
    ci_b[3] = 2 * (quats[1] * quats[2] + quats[3] * quats[0])
    ci_b[4] = quats[0] ** 2 + quats[2] ** 2 - quats[1] ** 2 - quats[3] ** 2
    ci_b[5] = 2 * (quats[2] * quats[3] - quats[1] * quats[0])
    ci_b[6] = 2 * (quats[3] * quats[1] - quats[2] * quats[0])
    ci_b[7] = 2 * (quats[2] * quats[3] + quats[1] * quats[0])
    ci_b[8] = quats[0] ** 2 + quats[3] ** 2 - quats[1] ** 2 - quats[2]

    return ci_b


def wind_speed(w_xyz, C_IB):
    """
    Parameters
    ----------
    w_xyz : scope of 3 in real
        wind speed in earth normolized coord.
    C_IB : scope of 9 params in real
        cross matrix from normilized to body.

    Returns
    -------
    Uxyz : TYPE
        wind speed in body.

    """
    uxyz = [C_IB[0] * w_xyz[0] + C_IB[3] * w_xyz[0] + C_IB[6] * w_xyz[0],
            C_IB[1] * w_xyz[1] + C_IB[4] * w_xyz[1] + C_IB[7] * w_xyz[1],
            C_IB[2] * w_xyz[2] + C_IB[5] * w_xyz[2] + C_IB[8] * w_xyz[2]]
    return uxyz


class Bullet(object):

    def __init__(self, kalib, mas, jx, jy, jz, v_full, tang0, w_xyz0):
        # 0 - kalib, 1 - mass, 2 - Jx, 3 - Jy, 4 - Jz, 5 - speed_full, 6 - tang in deg, w_xyz[x, y, z] - wind_speed

        T, a, P, ro = atmo(0)

        self.d = kalib  # mid diam
        self.s_f = 0.25 * kk.pi * self.d ** 2
        self.mass = mas
        self.Jxyz = [jx, jy, jz]

        # in global
        self.x = 0
        self.y = 0
        self.z = 0

        self.v = v_full  # full speed m/s

        # angles
        self.tang = tang0 * kk.pi / 180  # in rad
        self.psi = 0 * kk.pi / 180  # in rad
        self.gamma = 0 * kk.pi / 180  # in rad
        self.alf = 0  # in deg
        self.betta = 0  # in deg
        self.mem_of_alf_p = 0  # in deg

        # in body
        self.vx1 = self.v * kk.cos(self.alf / 180 * kk.pi) * kk.cos(self.betta / 180 * kk.pi)
        self.vy1 = self.v * (-kk.sin(self.alf / 180 * kk.pi)) * kk.cos(self.betta / 180 * kk.pi)
        self.vz1 = self.v * kk.sin(self.betta / 180 * kk.pi)
        self.wx = 0
        self.wy = 0
        self.wz = 0

        self.mah = kk.sqrt((self.vx1 - w_xyz0[0]) ** 2 + (self.vy1 - w_xyz0[1]) ** 2 + (self.vz1 - w_xyz0[2]) ** 2) / a
        self.q = 0.5 * ro * ((self.vx1 - w_xyz0[0]) ** 2 + (self.vy1 - w_xyz0[1]) ** 2 + (self.vz1 - w_xyz0[2]) ** 2)

        # quats
        self.L0 = kk.cos(self.psi / 2) * kk.cos(self.tang / 2) * kk.cos(self.gamma / 2) - kk.sin(self.psi / 2) \
                  * kk.sin(self.tang / 2) * kk.sin(self.gamma / 2)
        self.L1 = kk.cos(self.psi / 2) * kk.cos(self.tang / 2) * kk.sin(self.gamma / 2) + kk.sin(self.psi / 2) \
                  * kk.sin(self.tang / 2) * kk.cos(self.gamma / 2)
        self.L2 = kk.sin(self.psi / 2) * kk.cos(self.tang / 2) * kk.cos(self.gamma / 2) + kk.cos(self.psi / 2) \
                  * kk.sin(self.tang / 2) * kk.sin(self.gamma / 2)
        self.L3 = kk.cos(self.psi / 2) * kk.sin(self.tang / 2) * kk.cos(self.gamma / 2) - kk.sin(self.psi / 2) \
                  * kk.cos(self.tang / 2) * kk.sin(self.gamma / 2)

    def windd(self, w_xyz):

        lxx = kk.cos(self.tang) * kk.cos(self.psi)
        lxy = kk.sin(self.tang)
        lxz = -kk.cos(self.tang) * kk.sin(self.psi)
        lyx = -kk.cos(self.gamma) * kk.sin(self.tang) * kk.cos(self.psi) + kk.sin(self.gamma) * kk.sin(self.psi)
        lyy = kk.cos(self.gamma) * kk.cos(self.tang)
        lyz = kk.cos(self.gamma) * kk.sin(self.tang) * kk.sin(self.psi) + kk.sin(self.gamma) * kk.cos(self.psi)
        lzx = kk.sin(self.gamma) * kk.sin(self.tang) * kk.cos(self.psi) + kk.cos(self.gamma) * kk.sin(self.psi)
        lzy = -kk.sin(self.gamma) * kk.cos(self.tang)
        lzz = -kk.sin(self.gamma) * kk.sin(self.tang) * kk.sin(self.psi) + kk.cos(self.gamma) * kk.cos(self.psi)

        dcm_l = [lxx, lxy, lxz, lyx, lyy, lyz, lzx, lzy, lzz]

        wxyz_b = np.zeros(3)

        wxyz_b[0] = dcm_l[0] * w_xyz[0] + dcm_l[1] * w_xyz[1] + dcm_l[2] * w_xyz[2]
        wxyz_b[1] = dcm_l[3] * w_xyz[0] + dcm_l[4] * w_xyz[1] + dcm_l[5] * w_xyz[2]
        wxyz_b[2] = dcm_l[7] * w_xyz[0] + dcm_l[7] * w_xyz[1] + dcm_l[8] * w_xyz[2]

        return wxyz_b

    def aero(self, u_xyz):

        T, a, P, ro = atmo(self.y)
        """___________________________________________________________________
                                    find flow params
        ___________________________________________________________________"""
        crd = 180 / kk.pi  # rad to deg

        v_u_x = self.vx1 - u_xyz[0]
        v_u_y = self.vy1 - u_xyz[1]
        v_u_z = self.vz1 - u_xyz[2]

        abs_v_flow = kk.sqrt(v_u_x ** 2 + v_u_y ** 2 + v_u_z ** 2)

        self.alf = kk.atan(-v_u_y / v_u_x) * crd
        self.betta = kk.asin(v_u_z / abs_v_flow) * crd

        self.mah = abs_v_flow / a
        self.q = 0.5 * ro * abs_v_flow ** 2

        if v_u_x == 0:  # self.vx1 == 0:
            alf_p = kk.pi / 2 * crd
        else:
            alf_p = kk.atan2(kk.sqrt(v_u_y ** 2 + v_u_z ** 2), v_u_x) * crd
            # kk.atan2(kk.sqrt(self.vy1 ** 2 + self.vz1 ** 2), self.vx1)
        # alf_p_ = (alf_p - self.mem_of_alf_p) / dt  # скоростной пространственный угол атаки
        self.mem_of_alf_p = alf_p

        if v_u_y == 0:  # self.vy1 == 0:
            fi_p = kk.pi / 2 * crd
        else:
            fi_p = kk.atan2(v_u_z, -v_u_y) * crd  # kk.atan2(self.vz1, -self.vy1)

        """___________________________________________________________________
                            find aero coef. with alf prostr
        ___________________________________________________________________"""
        # Подъемная сила
        Cy1_iz_korp_alf = 2 / 57.3 * kk.cos(tet)  # для конуса
        Cy1_iz_korp = Cy1_iz_korp_alf * alf_p / crd

        f_y1_p = Cy1_iz_korp * self.q * self.s_f

        # Сила сопротивления
        # Пусть поток ламинарный и дозвуковой
        rei_f = self.v * len_full / tb.tab_atm(self.y, 5)
        x_t_ = 0
        c_f = tb.tab_4_3(self.mah, x_t_) * tb.tab_4_2(rei_f, x_t_) / 2
        cx_tr = c_f * S_omiv / self.s_f

        cx_nos = tb.tab_4_11(self.mah, len_full / self.d)

        # p_dn = 0.0155 / kk.sqrt(len_full / self.d * c_f)
        # eta_dn = 1
        # cx_dn = -p_dn * eta_dn

        cx_0 = cx_tr + cx_nos  # + cx_dn

        cx_ind = (57.3 * Cy1_iz_korp_alf + 2 * tb.tab_4_40(self.mah, len_full / self.d, 1)) * (alf_p / crd / 57.3) ** 2
        Cx = cx_0 + cx_ind
        print(cx_0, cx_ind, "cx")
        print(cx_0, cx_ind, alf_p)

        """___________________________________________________________________
                                        momenti
        ___________________________________________________________________"""

        # tang

        mz_10 = 0  # форма идеального конуса -  в нейтральном положении при угле атаки 0 - обтекание равномерное

        x_f_alf = len_full - W_geom / self.s_f  # координата фокуса по углу атаки
        mz_alf = Cy1_iz_korp_alf * (x1_cm[krit_ust] - x_f_alf) / len_full
        # print(x_f_alf, x1_cm, mz_alf)

        # для конуса lambd_nos = lambd_f -> x_c_ob = x_cm of full metal form
        mz_wz = -2 * (1 - x1_cm[krit_ust] / len_full + (x1_cm[krit_ust] / len_full) ** 2 - x1_cm[0] / len_full)
        # print(mz_wz, mz_alf, alf_p, self.wz)
        # mz_alf_ = 0  # так как в модели отсутствуют крылья -> нет запаздывания скоса потока

        mz_p = mz_wz * self.wz * len_full / abs_v_flow # mz_10 + mz_alf * alf_p * crd #  + mz_wz * self.wz * len_full / abs_v_flow
        # mz_10 + mz_alf * alf_p + mz_wz * self.wz * len_full / abs_v_flow  # + mz_alf_ * alf_p_ * len_full / abs_v_flow

        Mz_p = mz_p * self.q * self.s_f * len_full

        # Mz_p = 0
        # kren

        """___________________________________________________________________
                            from prostr to std body coord
        ___________________________________________________________________"""

        fx = -Cx * self.q * self.s_f
        fy = f_y1_p * kk.cos(fi_p / crd)
        fz = f_y1_p * kk.sin(fi_p / crd)

        Mx = 0
        My = Mz_p * kk.sin(fi_p/crd)
        Mz = Mz_p * kk.cos(fi_p/crd)
        print(Mz_p, f_y1_p, fx, "force")
        # print(Mz, My, fy, fz)
        print()

        return [fx, fy, fz], [Mx, My, Mz]

    def dynamics(self, *args):  # 0 - Fxyz[x, y, z], 1 - Mxyz[x, y, z] all in body

        fxyz = args[0]
        mxyz = args[1]

        LL0, LL1, LL2, LL3 = norm_quat([self.L0, self.L1, self.L2, self.L3])

        # find current g in body

        quat_base = LL0 ** 2 + LL1 ** 2 + LL2 ** 2 + LL3 ** 2
        qinv = [LL0 / quat_base, -LL1 / quat_base, -LL2 / quat_base, -LL3 / quat_base]  # инверсия quat
        mult = quatmultiply(qinv, [0, 0, g, 0])
        amult = quatmultiply(mult, [LL0, LL1, LL2, LL3])

        """___________________________________________________________________
                                    find deriv..
        ___________________________________________________________________"""

        multxyz = quatmultiply([LL0, LL1, LL2, LL3], [0, self.vx1, self.vy1, self.vz1])
        der_xyz = quatmultiply(multxyz, qinv)

        der_vx1 = fxyz[0] / self.mass - amult[1] + self.vy1 * self.wz - self.vz1 * self.wy
        der_vy1 = fxyz[1] / self.mass - amult[2] + self.vz1 * self.wx - self.vx1 * self.wz
        der_vz1 = fxyz[2] / self.mass - amult[3] + self.vx1 * self.wy - self.vy1 * self.wx

        der_wx = mxyz[0] / self.Jxyz[0] - (self.Jxyz[2] - self.Jxyz[1]) / self.Jxyz[0] * self.wy * self.wz
        der_wy = mxyz[1] / self.Jxyz[1] - (self.Jxyz[0] - self.Jxyz[2]) / self.Jxyz[1] * self.wx * self.wz
        der_wz = mxyz[2] / self.Jxyz[2] - (self.Jxyz[1] - self.Jxyz[0]) / self.Jxyz[2] * self.wx * self.wy

        der_L0 = (-LL1 * self.wx - LL2 * self.wy - LL3 * self.wz) / 2
        der_L1 = (LL0 * self.wx - LL3 * self.wy + LL2 * self.wz) / 2
        der_L2 = (LL3 * self.wx + LL0 * self.wy - LL1 * self.wz) / 2
        der_L3 = (-LL2 * self.wx + LL1 * self.wy + LL0 * self.wz) / 2

        """___________________________________________________________________
                                find arguments
        ___________________________________________________________________"""
        # body speed
        self.vx1 = euler(self.vx1, der_vx1)
        self.vy1 = euler(self.vy1, der_vy1)
        self.vz1 = euler(self.vz1, der_vz1)
        self.v = kk.sqrt(self.vx1**2 + self.vy1**2 + self.vz1**2)

        # body ang. speed
        self.wx = euler(self.wx, der_wx)
        self.wy = euler(self.wy, der_wy)
        self.wz = euler(self.wz, der_wz)

        # quats
        self.L0 = euler(self.L0, der_L0)
        self.L1 = euler(self.L1, der_L1)
        self.L2 = euler(self.L2, der_L2)
        self.L3 = euler(self.L3, der_L3)

        LL0, LL1, LL2, LL3 = norm_quat([self.L0, self.L1, self.L2, self.L3])
        # print(kk.sqrt(LL0**2 + LL1**2 + LL2**2 + LL3**2))
        # Euler ang.
        self.tang = kk.asin(2 * (LL1 * LL2 + LL0 * LL3))
        self.gamma = kk.atan2((LL0 * LL1 - LL2 * LL3), (LL0 ** 2 + LL2 ** 2 - 0.5))
        self.psi = kk.atan2((LL0 * LL2 - LL1 * LL3), (LL0 ** 2 + LL1 ** 2 - 0.5))

        print(self.tang, self.psi, self.gamma)

        # print(self.v, self.tang * 180 / kk.pi, self.alf, self.wz * 180 / kk.pi, self.wy * 180 / kk.pi)
        # Earth normolized coords
        self.x = euler(self.x, der_xyz[1])
        self.y = euler(self.y, der_xyz[2])
        self.z = euler(self.z, der_xyz[3])


# model params
dt = 0.004
g = 9.81
wind_ = [0, 0, 10]  # wind in [x, y, z] projections on earth normilized coords

# mass params
Jx = [127.943 * 10 ** -9, 71.25 * 10 ** -9]  # кг * м^2
Jy = [774.765 * 10 ** -9, 439.11 * 10 ** -9]  # кг * м^2
Jz = [774.765 * 10 ** -9, 439.11 * 10 ** -9]  # кг * м^2
m = [11.85 * 10 ** -3, 5.61 * 10 ** -3]  # кг для стали 10
x1_cm = [30 * 10 ** -3, 25.1 * 10 ** -3]  # координата цм от носка

# geometry
d = 0.012  # м
W_geom = 1507 * 10 ** - 9  # м^3
S_omiv = 762.41735 * 10 ** -6  # м^2
krit_rasch = 0  # 0 - пуля, 1 - шар
krit_ust = 1  # 0 - статически неустойчивая, 1 - устойчивая

len_full = 0.04  # длина общая
tet = kk.atan(d/2 / len_full)  # deg - угол порураствора конуса

v_0 = 270  # м/с
tang_0 = 20  # deg

bull_12 = Bullet(d, m[krit_ust], Jx[krit_ust], Jy[krit_ust], Jz[krit_ust], v_0, tang_0, wind_)

time = 0

amult_0 = [0]
amult_1 = [1]
amult_2 = [2]
amult_3 = [3]

vxx = [bull_12.vx1]
vyy = [bull_12.vy1]
vzz = [bull_12.vz1]

der_vx = [0]
der_vy = [0]
der_vz = [0]

L00 = [bull_12.L0]
L11 = [bull_12.L1]
L22 = [bull_12.L2]
L33 = [bull_12.L3]

wxx = [bull_12.wx]
wyy = [bull_12.wy]
wzz = [bull_12.wz]

xx = [bull_12.x]
yy = [bull_12.y]
zz = [bull_12.z]

ti = [0]
"""___________________________________________________________________
                            Main body
___________________________________________________________________"""

while bull_12.y >= 0:  # time < 5: # bull_12.y >= 0:
    print(wind_speed(wind_, form_c_ib([bull_12.L0, bull_12.L1, bull_12.L2, bull_12.L3])))
    print(wind_)
    f_xyz, m_xyz = bull_12.aero(bull_12.windd(wind_))  # wind_speed(wind_, form_c_ib([bull_12.L0, bull_12.L1, bull_12.L2, bull_12.L3])))
    bull_12.dynamics(f_xyz, m_xyz)
    time += dt

    vxx.append(bull_12.vx1)
    vyy.append(bull_12.vy1)
    vzz.append(bull_12.vz1)

    L00.append(bull_12.L0)
    L11.append(bull_12.L1)
    L22.append(bull_12.L2)
    L33.append(bull_12.L3)

    wxx.append(bull_12.wx)
    wyy.append(bull_12.wy)
    wzz.append(bull_12.wz)

    xx.append(bull_12.x)
    yy.append(bull_12.y)
    zz.append(bull_12.z)

    ti.append(time)

plt.figure(figsize=(10 * 1.5, 6 * 1.5))
plt.plot(xx, yy)
plt.grid(True)
plt.show()

plt.figure(figsize=(10 * 1.5, 6 * 1.5))
plt.plot(ti, xx)
plt.plot(ti, yy)
plt.plot(ti, zz)
plt.grid(True)
plt.show()

plt.figure(figsize=(10 * 1.5, 6 * 1.5))
plt.plot(ti, vxx)
plt.plot(ti, vyy)
plt.plot(ti, vzz)
plt.grid(True)
plt.show()

plt.figure(figsize=(10 * 1.5, 6 * 1.5))
plt.plot(ti, wxx)
plt.plot(ti, wyy)
plt.plot(ti, wzz)
plt.grid(True)
plt.show()

plt.figure(figsize=(10 * 1.5, 6 * 1.5))
plt.plot(ti, L00)
plt.plot(ti, L11)
plt.plot(ti, L22)
plt.plot(ti, L33)
plt.grid(True)
plt.show()
