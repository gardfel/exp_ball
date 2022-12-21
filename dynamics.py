import math as kk
import matplotlib.pyplot as plt
import numpy as np
import Tabl as tb


euler = lambda arg, der: arg + der * dt
sign = lambda arg: arg / kk.sqrt(arg ** 2)


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
        self.s_f = kk.pi * (self.d * 2 / 4) ** 2
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

        if v_u_x == 0:  
            alf_p = kk.pi / 2 * crd
        else:
            alf_p = kk.atan2(kk.sqrt(v_u_y ** 2 + v_u_z ** 2), v_u_x) * crd
    
        self.mem_of_alf_p = alf_p

        if v_u_y == 0:  
            fi_p = kk.pi / 2 * crd
        else:
            fi_p = kk.atan2(v_u_z, -v_u_y) * crd  

        """___________________________________________________________________
                            find aero coef. with alf prostr
        ___________________________________________________________________"""
        # Подъемная сила
        Cy1_iz_korp_alf = tb.tab_3_4(self.mah, 1, 1)  # 2 / 57.3 * (kk.cos(tet)) ** 2 # для конуса
        Cy1_iz_korp = Cy1_iz_korp_alf * kk.sin(alf_p / crd)  

        f_y1_p = Cy1_iz_korp * self.q * self.s_f

        # Сила сопротивления
        # Пусть поток ламинарный и дозвуковой
        rei_f = self.v * len_full / tb.tab_atm(self.y, 5)
        x_t_ = 0
        c_f = tb.tab_4_3(self.mah, x_t_) * tb.tab_4_2(rei_f, x_t_) / 2
        cx_tr = c_f * S_omiv / self.s_f

        cx_nos = tb.tab_4_13(self.mah, len_full / self.d)

        """
        ___________________________________________________________
        """


        cx_0 = (cx_tr + cx_nos) * 1.2
        # len_full / self.d, 1
        cx_ind = (57.3 * Cy1_iz_korp_alf + 2 * tb.tab_4_40(self.mah, len_full / self.d, 1)) * kk.sin(alf_p / crd) ** 2
        Cx = cx_0  # + cx_ind

        """___________________________________________________________________
                                        momenti
        ___________________________________________________________________"""

        # tang

        mz_10 = 0  # форма идеального конуса -  в нейтральном положении при угле атаки 0 - обтекание равномерное

        x_f_alf = 0.006 # len_full - W_geom / self.s_f  # координата фокуса по углу атаки
        # print(x_f_alf, mz_alf)
        mz_alf = Cy1_iz_korp_alf * (x1_cm[krit_ust] - x_f_alf) / len_full
        #print(x_f_alf, mz_alf)
        # print(mz_alf, mz_alf * kk.sin(alf_p / crd), alf_p, x_f_alf, "xx", Cx * self.q* self.s_f, f_y1_p, Cx, cx_0, cx_nos)
        
        # для конуса lambd_nos = lambd_f -> x_c_ob = x_cm of full metal form
        mz_wz = -2 * (1 - x1_cm[krit_ust] / len_full + (x1_cm[krit_ust] / len_full) ** 2 - x1_cm[0] / len_full)
        
        # mz_alf_ = 0  # так как в модели отсутствуют крылья -> нет запаздывания скоса потока

        mz_p = mz_wz * self.wz * len_full / abs_v_flow + mz_alf * kk.sin(alf_p / crd)  # * crd #  + mz_wz * self.wz * len_full / abs_v_flow
        # mz_wz * self.wz * len_full / abs_v_flow + mz_wz * self.wz * len_full / abs_v_flow +
        Mz_p = mz_p * self.q * self.s_f * len_full

        # Mz_p = 0

        """___________________________________________________________________
                            from prostr to std body coord
        ___________________________________________________________________"""

        fx = -Cx * self.q * self.s_f
        fy = f_y1_p * kk.cos(fi_p / crd)
        fz = f_y1_p * kk.sin(fi_p / crd)

        Mx = 0
        My = Mz_p * kk.sin(fi_p/crd)
        Mz = Mz_p * kk.cos(fi_p/crd)

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

        der_wx = float(mxyz[0] / self.Jxyz[0] - (self.Jxyz[2] - self.Jxyz[1]) / self.Jxyz[0] * self.wy * self.wz)
        der_wy = float(mxyz[1] / self.Jxyz[1] - (self.Jxyz[0] - self.Jxyz[2]) / self.Jxyz[1] * self.wx * self.wz)
        der_wz = float(mxyz[2] / self.Jxyz[2] - (self.Jxyz[1] - self.Jxyz[0]) / self.Jxyz[2] * self.wx * self.wy)

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
        # print(self.v)

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

        # Euler ang.
        self.tang = kk.asin(2 * (LL1 * LL2 + LL0 * LL3))
        if fxyz[2] == 0:
            self.gamma = 0
        elif time >= 27.1:
            self.gamma = 0
        else:
            self.gamma = kk.atan2((LL0 * LL1 - LL2 * LL3), (LL0 ** 2 + LL2 ** 2 - 0.5))
        self.psi = kk.atan2((LL0 * LL2 - LL1 * LL3), (LL0 ** 2 + LL1 ** 2 - 0.5))

        # Earth normolized coords
        self.x = euler(self.x, der_xyz[1])
        self.y = euler(self.y, der_xyz[2])
        self.z = euler(self.z, der_xyz[3])


# model params
dt = 0.001
g = 9.81
wind_ = [[0, 0, 0], [0, 0, -20], [0, 20, 0], [0, 14, -14]]  # wind in [x, y, z] projections on earth normilized coords

# mass params
# Jx = [127.943 * 10 ** -6, 71.25 * 10 ** -6]  # кг * м^2
# Jy = [774.765 * 10 ** -6, 439.11 * 10 ** -6]  # кг * м^2
# Jz = [774.765 * 10 ** -6, 439.11 * 10 ** -6]  # кг * м^2
# m = [11.85 * 10 ** -3, 5.61 * 10 ** -3]  # кг для стали 10
# x1_cm = [30 * 10 ** -3, 25.1 * 10 ** -3]  # координата цм от носка

Jx = [102 * 10 ** -6,float(208 * 10 ** -6), 69.178 * 10 ** -6]  # кг * м^2
Jy = [102 * 10 ** -6, float(3643 * 10 ** -6), 1533.3 * 10 ** -6]  # кг * м^2
Jz = [102 * 10 ** -6, float(3643 * 10 ** -6), 1533.3 * 10 ** -6]  # кг * м^2
m = [0.0071,0.0193, 5.48 * 10 ** -3]  # кг для стали 10
x1_cm = [0.006, 0.058, 40. * 10 ** -3]  # координата цм от носка

# geometry
d = 0.012  # м
W_geom = 0.006  # 0.00000264  # 1507 * 10 ** - 9  # м^3
S_omiv = 0.000226  # 1324.31 * 10 ** -6  # м^2
krit_rasch = 0  # 0 - пуля, 1 - шар
krit_ust = 0  # 0 - статически неустойчивая, 1 - устойчивая

len_full = 0.012  # длина общая
tet = kk.atan(d/2 / len_full)  # deg - угол порураствора конуса

v_0 = 300  # м/с
tang_0 = 20  # deg


"""___________________________________________________________________
                            Main body
___________________________________________________________________"""
color = ['r', 'g', 'b', 'k']
label = ['отсутвие ветра', 'горизонтальный', 'вертикальный', 'смешанный']
'''plt.figure(figsize=(10 * 2, 1 * 3))
for ii in range(0, 4):

    bull_12 = Bullet(d, m[krit_ust], Jx[krit_ust], Jy[krit_ust], Jz[krit_ust], v_0, tang_0, wind_[ii])
    print(bull_12.alf, bull_12.betta)
    time = 0

    xx = [bull_12.x]
    yy = [bull_12.y]
    zz = [bull_12.z]

    ti = [0]

    while bull_12.y >= 0:  # time < 5: # bull_12.y >= 0:

        f_xyz, m_xyz = bull_12.aero(bull_12.windd(wind_[ii]))  # wind_speed(wind_, form_c_ib([bull_12.L0, bull_12.L1, bull_12.L2, bull_12.L3])))
        bull_12.dynamics(f_xyz, m_xyz)
        # print(bull_12.alf, bull_12.betta, bull_12.tang * 180 / kk.pi)
        # print()
        time += dt

        xx.append(bull_12.x)
        yy.append(bull_12.y)
        zz.append(bull_12.z)

        ti.append(time)

    plt.plot(xx, yy, linestyle='-',color=color[ii], label=label[ii])  # 'y(x)')
    plt.plot(xx, zz, linestyle='--',color=color[ii])  # label='z(x)')
    plt.legend(title="ветеровая обстановка")
    plt.axis([0, 3500, 0, 600])
    plt.xlabel('x, м', fontsize=15)
    plt.ylabel('y(x), z(x), м', fontsize=15)
    plt.grid(True)
    # plt.xticks(range(0, 2401, 200))

plt.show()'''


for ii in range(0, 4):

    bull_12 = Bullet(d, m[krit_ust], Jx[krit_ust], Jy[krit_ust], Jz[krit_ust], v_0, tang_0, wind_[ii])

    time = 0

    der_vx = [0]
    der_vy = [0]
    der_vz = [0]

    vxx = [bull_12.tang * 180 / kk.pi]
    vyy = [bull_12.vy1 * 180 / kk.pi]
    vzz = [bull_12.vz1 * 180 / kk.pi]

    LL0 = [bull_12.L0]
    LL1 = [bull_12.L1]
    LL2 = [bull_12.L2]
    LL3 = [bull_12.L3]

    ti = [0]

    while bull_12.y >= 0:  # time < 5: # bull_12.y >= 0:

        f_xyz, m_xyz = bull_12.aero(
            bull_12.windd(wind_[ii]))  # wind_speed(wind_, form_c_ib([bull_12.L0, bull_12.L1, bull_12.L2, bull_12.L3])))
        bull_12.dynamics(f_xyz, m_xyz)
        # print(bull_12.alf, bull_12.betta, bull_12.tang * 180 / kk.pi)
        # print()
        time += dt

        # vxx.append(bull_12.alf * 180 / kk.pi)  # bull_12.tang * 180 / kk.pi)
        # vxx.append(bull_12.tang * 180 / kk.pi)
        # vyy.append(bull_12.gamma * 180 / kk.pi)  # bull_12.gamma * 180 / kk.pi)
        # vzz.append(bull_12.psi * 180 / kk.pi)
        # vxx.append(bull_12.alf)
        # vyy.append(bull_12.betta)  # bull_12.gamma * 180 / kk.pi)

        vxx.append(bull_12.wx)
        vyy.append(bull_12.wy)
        vzz.append(bull_12.wz)

        # print(bull_12.wz)
        # LL0.append(bull_12.L0)
        # LL1.append(bull_12.L1)
        # LL2.append(bull_12.L2)
        # LL3.append(bull_12.L3)

        ti.append(time)
    # plt.plot(ti, LL0)
    # plt.plot(ti, LL1)
    # plt.plot(ti, LL2)
    # plt.plot(ti, LL3)
    # plt.show()
    # plt.plot(ti, vxx, linestyle='-', color=color[ii], label=label[ii])  # 'y(x)')
    # plt.plot(ti, vyy, linestyle='-', color=color[ii], label=label[ii])  # label='z(x)')
    # plt.plot(ti, vzz, linestyle='-', color=color[ii], label=label[ii])

    plt.plot(ti, vxx, linestyle='-', color=color[ii], label=label[ii])
    plt.plot(ti, vyy, linestyle='--', color=color[ii])  #, label=label[ii])  # label='z(x)')
    # plt.plot(ti, vzz, linestyle='-.', color=color[ii])  #, label=label[ii])
    plt.legend(title="ветеровая обстановка")
    plt.axis([0, 25, -0.1, 0.1])
    plt.xlabel('t, с', fontsize=15)
    # plt.ylabel('Vz, м/с', fontsize=15)
    # plt.ylabel(r'$\vartheta$, град', fontsize=15)
    # plt.ylabel(r'$\alpha$,  $\beta$, град', fontsize=15)
    plt.ylabel(r'$\omega_{x}$, $\omega_{y}$, $\omega_{z}$, рад/с', fontsize=15)
    # plt.ylabel(r'$\psi$, град', fontsize=15)
    plt.grid(True)

plt.show()

'''
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
plt.show()'''


'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, zz, yy)'''
