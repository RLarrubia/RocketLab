import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd

class rocket():
    
    def __init__(self, 
                 R:float,
                 R0:float, 
                 L:float, 
                 rg:float,
                 rs:float,
                 N_sc: int,
                 parameters: pd.DataFrame):
        # Variables del cohete
        self.R = R  # Radio final del combustible
        self.R0 = R0  # Radio inicial del combustible
        self.L = L # Longitud del combustible
        self.rg = rg # Radio de la garganta
        self.rs = rs  # Radio de salida de la tobera
        self.N_sc = N_sc # Puntos para guardar y representar

        # Parámetros del tipo de combustible
        self.Tc = parameters['Tc'].iloc[0]                  # Temperatura de la cámara (se considera constante y conocida)
        self.Rg = 8.314 / parameters['Masa_molar'].iloc[0]  # Constante del gas del propulsante
        self.gamma = parameters['gamma'].iloc[0]                                # Gamma del gas propulsante
        self.rho_pr = parameters['rho_pr'].iloc[0]                              # Densidad del propulsante sólido real
        self.Rend = parameters['Rend'].iloc[0]                          # Rendimiento del propulsante que pasa a estado gaseoso
        self.rho_p = self.Rend * self.rho_pr            # Densidad del propulsante

        # Parámetros del modelo de velocidad de recesión Rp = a * Pc^n
        self.a = parameters['a'].iloc[0]
        self.n = parameters['n'].iloc[0]

        # Parámetros derivados
        self.As = np.pi * rs ** 2  # Área de salida
        self.Ag = np.pi * rg ** 2  # Área de garganta
        self.e = self.As / self.Ag  # Relación de áreas de la tobera
        self.gammadegamma = np.sqrt(self.gamma) * (2 / (self.gamma + 1)) ** ((self.gamma + 1) / (2 * (self.gamma - 1)))
        self.c_estrella = np.sqrt(self.Rg * self.Tc) / self.gammadegamma  # C estrella

        f = lambda X: self.e - self.gammadegamma / (X ** (1 / self.gamma) * (2 * self.gamma / (self.gamma - 1) * (1 - X ** ((self.gamma - 1) / self.gamma))) ** 0.5)
        self.sub = scipy.optimize.fsolve(f, 0.9)[0]  # Relación Ps/Pc con tobera bloqueada para el caso subsónico
        self.sup = scipy.optimize.fsolve(f, 0.1)[0]  # Relación Ps/Pc con tobera bloqueada para el caso supersónico

        self.msup = np.sqrt(2 / (self.gamma - 1) * (1 - self.sup ** ((self.gamma - 1) / self.gamma)) / (self.sup ** ((self.gamma - 1) / self.gamma)))  # Mach a la salida supersónico
        self.msub = np.sqrt(2 / (self.gamma - 1) * (1 - self.sub ** ((self.gamma - 1) / self.gamma)) / (self.sub ** ((self.gamma - 1) / self.gamma)))  # Mach a la salida subsónico

        # Variables de diseño del cohete
        self.Mcp = parameters['Mcp'].iloc[0]        # Masa de la carga de pago
        self.Rcg_cp = parameters['Rcg_cp'].iloc[0]  # Posición del centro de masas de la carga de pago

        # Parámetros del cohete
        self.Cd = parameters['Cd'].iloc[0]  # Coeficiente de resistencia aerodinámica del cohete
        self.S = np.pi * self.R ** 2  # Superficie mojada del cohete

        # Parámetros para el modelo atmosférico
        self.Re = parameters['Re'].iloc[0]   # Radio de la Tierra
        self.g0 = parameters['g0'].iloc[0]   # Aceleración de la gravedad en la superficie
        self.Ra = parameters['Ra'].iloc[0]      # Constante del aire

        self.atm = np.array([
        [0, 0, -6.5e-3, 288.15, 101325, 1.224],         # Troposfera <11000
        [11000, 11019, 0, 266.65, 22632, 0.3639],       # Tropopausa <20000
        [20000, 20063, 1e-3, 216.65, 5474.9, 0.088],    # Estratosfera <32000
        [32000, 32162, 2.8e-3, 228.65, 868.02, 0.0132], # Estratosfera 2 <47000
        [47000, 47350, 0, 270.65, 110.91, 1.224],       # Estratopausa <51000
        [51000, 51413, -2.8e-3, 270.65, 66.939, 0],     # Mesosfera <71000
        [71000, 71802, -2e-3, 214.65, 3.9564, 0],       # Mesosfera 2 <84532
        [84852, 86000, 0, 186.87, 0.3734, 0]            # Mesopausa < Termosfera
        ])

        # Parámetros para el modelo de presurización
        self.Mc = parameters['Mc'].iloc[0]  # Masa de la carcasa del motor

        # Parámetros derivados
        self.Mp = self.rho_pr * np.pi * (self.R ** 2 - self.R0 ** 2) * self.L
        self.M0 = self.Mcp + self.Mc + self.Mp

        # VECTORES SOLUCIÓN
        # Discretización del intervalo temporal
        self.t0 = parameters['t0'].iloc[0]
        self.tf = parameters['tf'].iloc[0]
        self.N = parameters['N'].iloc[0]
        self.t = np.zeros(self.N + 1)
        self.dt = np.zeros(self.N)

        # TIPO DE DISTRIBUCIÓN CHEBYCHEV
        for i in range(self.N):
            self.t[self.N - i] = (self.tf - self.t0) / 2 + (self.tf - self.t0) / 2 * np.cos((np.pi / 2 + i * np.pi) * 1 / (self.N + 1))
            self.t[0] = 0

        for i in range(1, self.N):
            self.dt[i - 1] = self.t[i] - self.t[i - 1]

        # Vectores del modelo del motor
        self.Pc = np.zeros(self.N + 1)       # Presión de cámara en cada instante
        self.Ms = np.zeros(self.N + 1)       # Mach a la salida en cada instante
        self.Ts = np.zeros(self.N + 1)       # Temperatura a la salida en cada instante
        self.Vs = np.zeros(self.N + 1)       # Velocidad a la salida en cada instante
        self.E = np.zeros(self.N + 1)        # Empuje en cada instante
        self.rho_s = np.zeros(self.N + 1)    # Densidad a la salida

        self.Ge = np.zeros(self.N + 1)       # Gasto de combustible que entra en la cámara
        self.Gs = np.zeros(self.N + 1)       # Gasto que sale por la tobera
        self.Vc = np.zeros(self.N + 1)       # Volumen de la cámara en cada instante
        self.Rp = np.zeros(self.N + 1)       # Velocidad de recesión en cada instante
        self.r = np.zeros(self.N + 1)        # Radio del combustible en cada instante
        self.Ab = np.zeros(self.N + 1)       # Área lateral del combustible en cada instante

        self.color = np.zeros((self.N + 1, 3))  # Establece un color dependiendo del tipo de flujo

        # Vectores solución del modelo del cohete
        self.h = np.zeros(self.N + 1)        # Altura en cada instante
        self.der = np.zeros(self.N + 1)      # Derivadas primeras de la altura
        self.der2 = np.zeros(self.N + 1)     # Derivadas segundas de la altura

        self.pa = np.zeros(self.N + 1)       # Presión atmosférica en cada instante
        self.rho_ext = np.zeros(self.N + 1)  # Densidad exterior en cada instante
        self.Ta_ext = np.zeros(self.N + 1)   # Temperatura ambiente exterior en cada instante
        self.M = np.zeros(self.N + 1)        # Masa del cohete en cada instante
        self.Rcg = np.zeros(self.N + 1)      # Posición del centro de masas del cohete en cada instante respecto al área de salida
        self.Vcg = np.zeros(self.N + 1)      # Velocidad del centro de masas del cohete en cada instante respecto al área de salida
        self.Acg = np.zeros(self.N + 1)      # Aceleración del centro de gravedad en cada instante respecto al área de salida
        self.Gcg = np.zeros(self.N + 1)      # Variación del gasto como consecuencia del movimiento del centro de masas

        self.Fa = np.zeros(self.N + 1)       # Fuerza aerodinámica en cada instante
        self.g = np.zeros(self.N + 1)        # Aceleración de la gravedad en cada instante
        self.Ps = np.zeros(self.N + 1)       # Presión a la salida de la tobera

        # CONDICIONES INICIALES
        # Condiciones iniciales del motor
        self.Pc[0] = 101325
        self.Ts[0] = self.Tc
        self.r[0] = self.R0
        self.M[0] = self.M0
        # Condiciones iniciales del cohete
        self.h[0] = 0
        self.der[0] = 0

        self.actitude_simulation()
        self.save_to_excel()
    
    def rocket_engine(self, i:int):
        # Cálculo de las variables del motor
            self.Rp[i] = self.a * self.Pc[i] ** self.n
            self.Ab[i] = 2 * np.pi * self.L * self.r[i]
            self.Ge[i] = self.rho_p * self.Rp[i] * self.Ab[i]
            self.Vc[i] = np.pi * self.r[i] ** 2 * self.L
            dV = self.Ab[i] * self.Rp[i]
            self.r[i + 1] = self.r[i] + self.Rp[i] * self.dt[i]

            if self.r[i] > self.R:
                self.Rp[i] = 0
                self.Ge[i] = 0
                self.r[i + 1] = self.r[i]
                self.Ab[i] = self.Ab[i - 1]
                self.Vc[i] = self.Vc[i - 1]
                dV = 0
                self.Ts[i] = self.Ta_ext[i]

            m1n = self.msup
            m2n = np.sqrt((2 + (self.gamma - 1) * m1n ** 2) / (2 * self.gamma * m1n ** 2 - (self.gamma - 1)))
            p02 = self.pa[i] * (1 + (self.gamma - 1) / 2 * m2n ** 2) ** (self.gamma / (self.gamma - 1))
            salto_P0_OCN = (((2 * self.gamma * m1n ** 2 - (self.gamma - 1)) / (self.gamma + 1)) * ((2 + (self.gamma - 1) * m1n ** 2) / ((self.gamma + 1) * m1n ** 2)) ** self.gamma) ** (1 / (self.gamma - 1))
            Z = p02 * salto_P0_OCN

            X = self.pa[i] / self.sub

            if self.Pc[i] < X:
                self.Gs[i] = np.sqrt(self.gamma) * self.As * np.sqrt(2 / (self.gamma - 1)) * self.pa[i] / (np.sqrt(self.Rg * self.Tc)) * ((self.Pc[i] / self.pa[i]) ** (2 * (self.gamma - 1) / self.gamma) - (self.Pc[i] / self.pa[i]) ** ((self.gamma - 1) / self.gamma))
                self.Pc[i + 1] = self.Pc[i] + self.dt[i] * (self.Ge[i] - self.Gs[i]) * (self.Rg * self.Tc) / self.Vc[i] - dV / self.Vc[i]
                if self.Pc[i + 1] < self.pa[i]:
                    self.Pc[i + 1] = self.pa[i]

                self.Ms[i + 1] = np.sqrt(2 / (self.gamma - 1) * ((self.Pc[i + 1] / self.pa[i]) ** ((self.gamma - 1) / self.gamma) - 1))
                self.Ts[i + 1] = self.Tc / (1 + (self.gamma - 1) / 2 * self.Ms[i + 1] ** 2)
                self.Vs[i + 1] = self.Ms[i + 1] * np.sqrt(self.gamma * self.Rg * self.Ts[i + 1])
                self.E[i + 1] = self.Gs[i] * self.Vs[i + 1]

                self.rho_s[i] = self.pa[i] / (self.Rg * self.Ts[i])
                self.Ps[i] = self.pa[i]
                self.color[i, :] = [0, 1, 1]
            else:
                self.Gs[i] = self.Pc[i] * self.Ag / self.c_estrella
                self.Pc[i + 1] = self.Pc[i] + self.dt[i] * (self.Ge[i] - self.Gs[i]) * (self.Rg * self.Tc) / self.Vc[i] - dV / self.Vc[i]
                if self.Pc[i + 1] < self.pa[i]:
                    self.Pc[i + 1] = self.pa[i]

                if self.Pc[i] < Z:
                    self.Ms[i + 1] = self.msub
                    self.Ts[i + 1] = self.Tc / (1 + (self.gamma - 1) / 2 * self.Ms[i + 1] ** 2)
                    self.Vs[i + 1] = self.Ms[i + 1] * np.sqrt(self.gamma * self.Rg * self.Ts[i + 1])
                    self.E[i + 1] = self.Gs[i] * self.Vs[i + 1]

                    self.rho_s[i] = self.pa[i] / (self.Rg * self.Ts[i])
                    self.Ps[i] = self.pa[i]
                    self.color[i, :] = [0, 0, 0]
                
                else:
                    self.Ms[i + 1] = self.msup
                    self.Ts[i + 1] = self.Tc / (1 + (self.gamma - 1) / 2 * self.Ms[i + 1] ** 2)
                    self.Vs[i + 1] = self.Ms[i + 1] * np.sqrt(self.gamma * self.Rg * self.Ts[i + 1])
                    Ce = self.gammadegamma * np.sqrt(2 * self.gamma / (self.gamma - 1) * (1 - self.sup ** ((self.gamma - 1) / self.gamma))) + self.e * (self.sup - self.pa[i] / self.Pc[i + 1])
                    self.E[i + 1] = self.Pc[i + 1] * self.Ag * Ce

                    self.rho_s[i] = self.sup * self.Pc[i] / (self.Rg * self.Ts[i])
                    self.Ps[i] = self.Pc[i] * self.sup
                    self.color[i, :] = [1, 0, 0]

    def actitude_simulation(self):
        # Resolución
        for i in range(self.N - 1):
            # Determinación de la presión atmosférica en ese instante
            z = self.Re * self.h[i] / (self.Re + self.h[i])
            k = np.where(self.atm[:, 0] <= z)[0][-1]

            if self.atm[k, 2] == 0:
                self.pa[i] = self.atm[k, 4] * np.exp(-self.g0 / (self.Ra * self.atm[k, 3]) * (z - self.atm[k, 0]))
                self.rho_ext[i] = self.atm[k, 5] * np.exp(-self.g0 / (self.Ra * self.atm[k, 3]) * (z - self.atm[k, 0]))
                self.Ta_ext[i] = self.atm[k, 3]
            else:
                Text = self.atm[k, 3] + self.atm[k, 2] * (z - self.atm[k, 0])
                self.pa[i] = self.atm[k, 4] * (Text / self.atm[k, 3]) ** (-self.g0 / (self.atm[k, 2] * self.Ra))
                self.rho_ext[i] = self.atm[k, 5] * (Text / self.atm[k, 3]) ** (-self.g0 / (self.atm[k, 2] * self.Ra))
                self.Ta_ext[i] = self.atm[k, 3] + self.atm[k, 2] * (z - self.atm[k, 0])

            self.rocket_engine(i)

            self.M[i + 1] = self.M[i] - self.Gs[i] * self.dt[i]

            self.Rcg[i] = (self.Mcp * self.Rcg_cp + self.Mc * self.L / 2 + (self.M[i] - self.Mc - self.Mcp) * self.L / 2) / self.M[i]
            self.Rcg[i + 1] = (self.Mcp * self.Rcg_cp + self.Mc * self.L / 2 + (self.M[i + 1] - self.Mc - self.Mcp) * self.L / 2) / self.M[i + 1]

            self.Vcg[i] = self.Gs[i] * (self.Rcg[i] - self.L / 2) / self.M[i]

        
            if i == 0:
                self.Acg[i] = (2 * self.Rcg[i]) / (self.dt[i] * (self.dt[i] + self.dt[i])) + (2 * self.Rcg[i + 1]) / (self.dt[i] * (self.dt[i] + self.dt[i])) - (2 * self.Rcg[i]) / (self.dt[i] * self.dt[i])
            else:
                self.Acg[i] = (2 * self.Rcg[i - 1]) / (self.dt[i - 1] * (self.dt[i - 1] + self.dt[i])) + (2 * self.Rcg[i + 1]) / (self.dt[i] * (self.dt[i - 1] + self.dt[i])) - (2 * self.Rcg[i]) / (self.dt[i - 1] * self.dt[i])
            self.Gcg[i] = self.rho_s[i] * self.Vcg[i] * self.Vs[i] * self.As

            self.der[i] = self.h[i + 1] / self.dt[i] - self.h[i] / self.dt[i]

            self.Fa[i] = -np.sign(self.der[i]) * 0.5 * self.rho_ext[i] * self.Cd * self.S * self.der[i] ** 2
            self.g[i] = -self.g0 * self.Re ** 2 / (self.Re + self.h[i]) ** 2

            self.der2[i] = self.Acg[i] + self.g[i] + (self.Gcg[i] + self.E[i] + self.Fa[i]) / self.M[i]

            self.h[i + 2] = (-self.der2[i] + 2 * self.h[i] / (self.dt[i] * (self.dt[i] + self.dt[i + 1])) - 2 * self.h[i + 1] / (self.dt[i] * self.dt[i + 1])) * (-self.dt[i + 1] * (self.dt[i] + self.dt[i + 1]) / 2)

            if self.h[i + 2] < 0:
                self.h[i + 2] = 0

        if self.r[i] < self.R:
            pass
        else:
            self.r[self.N] = self.r[self.N - 1]
            self.Vc[self.N] = self.Vc[self.N - 1]
            self.Ts[self.N] = self.Ts[self.N - 1]

            self.Ps[self.N] = self.Ps[self.N - 1]
            self.Rcg[self.N] = self.Rcg[self.N - 1]
            self.Vcg[self.N] = self.Vcg[self.N - 1]
            self.Acg[self.N] = self.Acg[self.N - 1]
            self.Pc[self.N] = self.Pc[self.N - 1]
            self.der[self.N] = self.der[self.N - 1]
            self.der2[self.N] = self.der2[self.N - 1]
            self.color[self.N, :] = self.color[self.N - 1, :]
            

    def save_to_excel(self, filename="data"):
        # Datos de Engine
        Ns = self.N_sc
        indices = np.linspace(0, self.N, Ns, dtype=int)
        t_s = self.t[indices].ravel()
        Pc_s = self.Pc[indices].ravel()
        E_s = self.E[indices].ravel()
        Vs_s = self.Vs[indices].ravel()
        r_s = self.r[indices].ravel()
        Ge_s = self.Ge[indices].ravel()
        Gs_s = self.Gs[indices].ravel()
        Ms_s = self.Ms[indices].ravel()
        Ts_s = self.Ts[indices].ravel()
        Rp_s = self.Rp[indices].ravel()
        color_s = self.color[indices]
        color_s_columns = np.hsplit(color_s, color_s.shape[1])
        
        engine_scatter_data = {
            "t_s": t_s, "Pc_s": Pc_s, "E_s": E_s, "Vs_s": Vs_s, "r_s": r_s, "Ge_s": Ge_s, "Gs_s": Gs_s, 
            "Ms_s": Ms_s, "Ts_s": Ts_s, "Rp_s": Rp_s
        }

        for i, col in enumerate(color_s_columns):
            engine_scatter_data[f'color_s_{i+1}'] = col.ravel()

        engine_data = {
            "t": self.t.ravel(), "Pc": self.Pc.ravel(), "E": self.E.ravel(), "Vs": self.Vs.ravel(), "r": self.r.ravel(), 
            "Ge": self.Ge.ravel(), "Gs": self.Gs.ravel(), "Ms": self.Ms.ravel(), "Ts": self.Ts.ravel(), "Rp": self.Rp.ravel()
        }

        color_columns = np.hsplit(self.color, self.color.shape[1])
        for i, col in enumerate(color_columns):
            engine_data[f'color_{i+1}'] = col.ravel()

        # Datos de Attitude
        Ns2 = self.N_sc
        indices2 = np.linspace(0, self.N, Ns2, dtype=int)
        t_s2 = self.t[indices2].ravel()
        h_s = self.h[indices2].ravel()
        der_s = self.der[indices2].ravel()
        der2_s = self.der2[indices2].ravel()
        Fa_s = self.Fa[indices2].ravel()
        g_s = self.g[indices2].ravel()
        color_s2 = self.color[indices2]
        color_s2_columns = np.hsplit(color_s2, color_s2.shape[1])

        attitude_scatter_data = {
            "t_s2": t_s2, "h_s": h_s, "der_s": der_s, "der2_s": der2_s, "Fa_s": Fa_s, "g_s": g_s
        }

        for i, col in enumerate(color_s2_columns):
            attitude_scatter_data[f'color_s2_{i+1}'] = col.ravel()

        attitude_data = {
            "t2": self.t.ravel(), "h": self.h.ravel(), "der": self.der.ravel(), "der2": self.der2.ravel(), 
            "Fa": self.Fa.ravel(), "g": self.g.ravel()
        }

        color_columns2 = np.hsplit(self.color, self.color.shape[1])
        for i, col in enumerate(color_columns2):
            attitude_data[f'color2_{i+1}'] = col.ravel()

        if not os.path.exists('temp'):
            os.makedirs('temp')

        
        with pd.ExcelWriter(f'temp/{filename}.xlsx') as writer:
            # pd.DataFrame(engine_data).to_excel(writer, sheet_name="engine", index=False)
            pd.DataFrame(engine_scatter_data).to_excel(writer, sheet_name="engine_scatter", index=False)
            # pd.DataFrame(attitude_data).to_excel(writer, sheet_name="attitude", index=False)
            pd.DataFrame(attitude_scatter_data).to_excel(writer, sheet_name="attitude_scatter", index=False)
            
    

    def graphs(self):
        
        # GRÁFICAS
        # Número de puntos para mostrar en el scatter
        Ns = self.N_sc
        indices = np.linspace(0, self.N, Ns, dtype=int)
        t_s = self.t[indices]
        Pc_s = self.Pc[indices]
        E_s = self.E[indices]
        Vs_s = self.Vs[indices]
        r_s = self.r[indices]
        Ge_s = self.Ge[indices]
        Gs_s = self.Gs[indices]
        Ms_s = self.Ms[indices]
        Ts_s = self.Ts[indices]
        Rp_s = self.Rp[indices]
        color_s = self.color[indices]

        fig, axs = plt.subplots(3, 3)

        # Presión en la cámara
        axs[0, 0].plot(t_s, Pc_s / 101325, linewidth=2)
        axs[0, 0].scatter(t_s, Pc_s / 101325, c=color_s)
        axs[0, 0].set_title('Presión de cámara')
        axs[0, 0].set_ylabel('Pc (bar)')
        axs[0, 0].set_xlabel('t (s)')

        # Empuje
        axs[0, 1].plot(t_s, E_s, linewidth=2)
        axs[0, 1].scatter(t_s, E_s, c=color_s)
        axs[0, 1].set_title('Empuje')
        axs[0, 1].set_ylabel('E (N)')
        axs[0, 1].set_xlabel('t (s)')

        # Velocidad de salida
        axs[0, 2].plot(t_s, Vs_s, linewidth=2)
        axs[0, 2].scatter(t_s, Vs_s, c=color_s)
        axs[0, 2].set_title('Velocidad de salida')
        axs[0, 2].set_ylabel('Vs (m/s)')
        axs[0, 2].set_xlabel('t (s)')

        # Avance del frente de combustión
        axs[1, 0].plot(t_s, r_s * 1000, linewidth=2)
        axs[1, 0].set_title('Radio del combustible')
        axs[1, 0].set_ylabel('r (mm)')
        axs[1, 0].set_xlabel('t (s)')

        # Comparación de gastos
        axs[1, 1].plot(t_s, Gs_s, linewidth=2)
        axs[1, 1].plot(t_s, Ge_s, linewidth=2)
        axs[1, 1].set_title('Gastos')
        axs[1, 1].set_ylabel('G (kg/s)')
        axs[1, 1].set_xlabel('t (s)')
        axs[1, 1].legend(['Gasto tobera', 'Gasto entrante combustible'])

        # Mach a la salida
        axs[1, 2].plot(t_s, Ms_s, linewidth=2)
        axs[1, 2].scatter(t_s, Ms_s, c=color_s)
        axs[1, 2].set_title('Mach a la salida')
        axs[1, 2].set_ylabel('Ms')
        axs[1, 2].set_xlabel('t (s)')

        # Temperatura a la salida
        axs[2, 0].plot(t_s, Ts_s, linewidth=2)
        axs[2, 0].scatter(t_s, Ts_s, c=color_s)
        axs[2, 0].set_title('Temperatura a la salida')
        axs[2, 0].set_ylabel('Ts (K)')
        axs[2, 0].set_xlabel('t (s)')

        # Velocidad de recesión
        axs[2, 1].plot(t_s, Rp_s * 1000, linewidth=2)
        axs[2, 1].set_title('Velocidad de recesión')
        axs[2, 1].set_ylabel('Rp (mm/s)')
        axs[2, 1].set_xlabel('t (s)')

        plt.tight_layout()  # Corrección aquí
        # Gráficas cohete
        fig, axs = plt.subplots(2, 3)
        Ns2 = 50
        indices2 = np.linspace(0, self.N, Ns2, dtype=int)
        t_s = self.t[indices2]
        h_s = self.h[indices2]
        der_s = self.der[indices2]
        der2_s = self.der2[indices2]
        Fa_s = self.Fa[indices2]
        g_s = self.g[indices2]
        color_s = self.color[indices2]

        # Altura
        axs[0, 0].plot(self.t, self.h, linewidth=2)
        axs[0, 0].scatter(t_s, h_s, c=color_s)
        axs[0, 0].set_title('Altura de cámara')
        axs[0, 0].set_ylabel('H (m)')
        axs[0, 0].set_xlabel('t (s)')

        # Velocidad
        axs[0, 1].plot(self.t, self.der, linewidth=2)
        axs[0, 1].scatter(t_s, der_s, c=color_s)
        axs[0, 1].set_title('Velocidad (m/s)')
        axs[0, 1].set_ylabel('V (m/s)')
        axs[0, 1].set_xlabel('t (s)')

        # Aceleración total
        axs[0, 2].plot(self.t, self.der2, linewidth=2)
        axs[0, 2].scatter(t_s, der2_s, c=color_s)
        axs[0, 2].set_title('Aceleración total')
        axs[0, 2].set_ylabel('a (m/s²)')
        axs[0, 2].set_xlabel('t (s)')

        # Fuerza aerodinámica
        axs[1, 0].plot(self.t, self.Fa, linewidth=2)
        axs[1, 0].scatter(t_s, Fa_s, c=color_s)
        axs[1, 0].set_title('Fuerza Aerodinámica')
        axs[1, 0].set_ylabel('Fa (N)')
        axs[1, 0].set_xlabel('t (s)')

        # Posición, velocidad y aceleración del cg del cohete respecto al área de salida
        axs[1, 1].plot(self.t, self.Rcg, linewidth=2)
        axs[1, 1].plot(self.t, self.Vcg, linewidth=2)
        axs[1, 1].plot(self.t, self.Acg, linewidth=2)
        axs[1, 1].set_title('Posición, velocidad y aceleración del cg del cohete')
        axs[1, 1].set_ylabel('Rcg (m), Vcg (m/s), Acg (m/s²)')
        axs[1, 1].set_xlabel('t (s)')
        axs[1, 1].legend(['Rcg', 'Vcg', 'Acg'])

        # Presión a la salida y atmosférica
        axs[1, 2].plot(self.t, self.Ps / 101325, linewidth=2)
        axs[1, 2].plot(self.t, self.pa / 101325, linewidth=2)
        axs[1, 2].set_title('Variación de la presión de salida y atmosférica')
        axs[1, 2].set_ylabel('Ps, Pa (bar)')
        axs[1, 2].set_xlabel('t (s)')
        axs[1, 2].legend(['Pa', 'Ps'])

        plt.tight_layout()  # Corrección aquí

        plt.show()

if __name__ == '__main__':
    
    R = 2.25e-2  # Radio final del combustible
    R0 = 0.5e-2  # Radio inicial del combustible
    L = 8e-2  # Longitud del combustible
    rg = 0.5e-2  # Radio de la garganta
    rs = 0.5e-2  # Radio de salida de la tobera


    # Definición de las variables numéricas
    Tc = 1000  # Temperatura de la cámara (se considera constante y conocida)
    M_molar = 41.98e-3  # Masa molar del gas del propulsante
    gamma = 1.3  # Gamma del gas propulsante
    rho_pr = 1800  # Densidad del propulsante sólido real
    Rend = 1 - 0.4237  # Rendimiento del propulsante que pasa a estado gaseoso
    rho_p = Rend * rho_pr  # Densidad del propulsante

    # Parámetros del modelo de velocidad de recesión Rp = a * Pc^n
    a = 6e-5
    n = 0.32

    # Variables de diseño del cohete
    Mcp = 0.1  # Masa de la carga de pago # Longitud del cohete, debe definir este valor
    Rcg_cp = 5 * 10 / 4  # Posición del centro de masas de la carga de pago

    # Parámetros del cohete
    Cd = 1  # Coeficiente de resistencia aerodinámica del cohete

    # Parámetros para el modelo atmosférico
    Re = 6.37e6  # Radio de la Tierra
    g0 = 9.80665  # Aceleración de la gravedad en la superficie
    Ra = 287  # Constante del aire

    # Parámetros para el modelo de presurización
    Mc = 0.2  # Masa de la carcasa del motor

    # VECTORES SOLUCIÓN
    # Discretización del intervalo temporal
    t0 = 0
    tf = 10
    N = 50000
    N_sc = 50
    # Creación del DataFrame
    parameters = pd.DataFrame([{
        'Tc': Tc,
        'Masa_molar': M_molar,
        'gamma': gamma,
        'rho_pr': rho_pr,
        'Rend': Rend,
        'a': a,
        'n': n,
        'Mcp': Mcp,
        'Rcg_cp': Rcg_cp,
        'Cd': Cd,
        'Re': Re,
        'g0': g0,
        'Ra': Ra,
        'Mc': Mc,
        't0': t0,
        'tf': tf,
        'N': N,
    }])
    simulation = rocket(R, R0, L, rg, rs, N_sc, parameters)
    