import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler

# ------------------------------------------
# Impostazioni grafiche stile ROOT
# ------------------------------------------
plt.style.use(hep.style.ROOT)
plt.rcParams.update({
    'legend.fontsize': '10',
    'legend.loc': 'lower right',
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.facecolor': 'w',
    'legend.edgecolor': 'w',
    'figure.figsize': (6, 4),
    'axes.labelsize': '13',
    'figure.titlesize': '17',
    'axes.titlesize': '15',
    'xtick.labelsize': '13',
    'ytick.labelsize': '13',
    'lines.linewidth': '1',
    'text.usetex': True,
    'axes.formatter.min_exponent': '2',
    'figure.subplot.left': '0.125',
    'figure.subplot.bottom': '0.125',
    'figure.subplot.right': '0.925',
    'figure.subplot.top': '0.925',
    'figure.subplot.wspace': '0.1',
    'figure.subplot.hspace': '0.1',
    'axes.labelpad': 8.0,
    'xtick.major.pad': 5.0,
    'ytick.major.pad': 5.0,
})
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# ------------------------------
# Funzioni di fit
# ------------------------------
def fit_lin(x, m, q):
    """Modello lineare: y = m*x + q"""
    return m * x + q

def fit_lin_fixed_slope(x, q):
    """Modello lineare con pendenza fissata a -20"""
    return -20.0 * x + q

# ------------------------------
# Parametri iniziali
# ------------------------------
file = 'RCF_PB'
inputname = file + '.txt'

# Frequenza di taglio stimata a priori (solo per definire il range delle alte frequenze)
ft_init = 4392.0   # Hz
scan = 14000.0     # come nel vecchio codice

# Errori di lettura strumentali
y_errfix = 0.1
y_errper = 0.03

# ------------------------------
# Caricamento dati da file
# ------------------------------
data = np.loadtxt(inputname).T

f = 1.0 / (data[0] * 1e-6)      # frequenza in Hz
Vin = data[1]
Vout = data[2]
V_fs = data[3]
delta_t = data[4]
phi_fs = data[5]

A = Vout / Vin
phi = (4 * delta_t * f * 1e-6)   # fase normalizzata a 1 (non utilizzata nel Bode)
phi_fs = 2 * np.pi * f * phi_fs * 1e-6 / (np.pi / 2.)

# ------------------------------
# Calcolo errori su A (per Bode)
# ------------------------------
Vin_errL = 0.041
V_errL = (V_fs / 10.0) * 0.41

A_err = A * np.sqrt(
    (V_errL / Vout)**2 +
    (Vin_errL / Vin)**2 +
    2.0 * (0.03 * 0.41)**2
)
A_err[(f>10) & (f<10600)] = A[(f>10) & (f<10600)] * np.sqrt(
    (V_errL[(f>10) & (f<10600)] / Vout[(f>10) & (f<10600)])**2 +
    (Vin_errL / Vin[(f>10) & (f<10600)])**2
)

# ============================================================
# SELEZIONE REGIONE ASINTOTICA (come nel vecchio codice)
# ============================================================
f_min = ft_init + scan
f_max = ft_init + 55.0 * scan
mask_bode = (f > f_min) & (f < f_max)

x_bode = np.log10(f[mask_bode])                # log10(f) – nessuna normalizzazione
y_bode = 20.0 * np.log10(A[mask_bode])         # 20*log10(A) in dB
y_err_bode = 20.0 * A_err[mask_bode] / (A[mask_bode] * np.log(10.0))

N_bode = len(x_bode)

print(f"\nRegione Bode: {f_min:.0f} Hz -- {f_max:.0f} Hz")

# ============================================================
# FIT 1: pendenza libera (BODE 1)
# ============================================================
popt_bode1, pcov_bode1 = curve_fit(
    fit_lin, x_bode, y_bode,
    p0=[-20.0, 0.0],
    method='lm',
    sigma=y_err_bode,
    absolute_sigma=True
)
m1, q1 = popt_bode1
perr_m1, perr_q1 = np.sqrt(np.diag(pcov_bode1))

# ============================================================
# FIT 2: pendenza fissata a -20 (BODE 2)
# ============================================================
popt_bode2, pcov_bode2 = curve_fit(
    fit_lin_fixed_slope, x_bode, y_bode,
    p0=[0.0],
    method='lm',
    sigma=y_err_bode,
    absolute_sigma=True
)
q2 = popt_bode2[0]
perr_q2 = np.sqrt(np.diag(pcov_bode2))[0]

# ============================================================
# Residui e bontà del fit
# ============================================================
residui1 = y_bode - fit_lin(x_bode, *popt_bode1)
residui2 = y_bode - fit_lin_fixed_slope(x_bode, *popt_bode2)

chisq1 = np.sum((residui1 / y_err_bode)**2)
chisq2 = np.sum((residui2 / y_err_bode)**2)
dof1 = N_bode - 2
dof2 = N_bode - 1

# ============================================================
# Calcolo di ft dai due metodi
# ============================================================
# BODE 1: intersezione con 0 dB → ft = 10^(-q/m)
ft_bode1 = 10.0**(-q1 / m1)
err_ft_bode1 = np.sqrt(
    (ft_bode1 * np.log(10.0) * q1 / m1**2)**2 * perr_m1**2 +
    (ft_bode1 * np.log(10.0) / m1)**2 * perr_q1**2 +
    2.0 * (ft_bode1 * np.log(10.0))**2 * q1 / m1**3 * pcov_bode1[0,1]
)

# BODE 2: intercetta con pendenza -20 → ft = 10^(q/20)
ft_bode2 = 10.0**(q2 / 20.0)
err_ft_bode2 = ft_bode2 * np.log(10.0) / 20.0 * perr_q2

# ============================================================
# Stampa risultati
# ============================================================
print("\n ============== BEST FIT BODE ====================")
print("\n ================== BODE 1 (pendenza libera) ==================")
print(f"  slope m = {m1:.3f} +/- {perr_m1:.3f}")
print(f"  intercept q = {q1:.3f} +/- {perr_q1:.3f} dB")
print(f"  chisq/ndf = {chisq1/dof1:.2f}")
print(f"  frequenza di taglio BODE1 = {ft_bode1:.0f} +/- {err_ft_bode1:.0f} Hz")
print("\n ================== BODE 2 (pendenza fissata a -20) ===========")
print(f"  intercept q = {q2:.3f} +/- {perr_q2:.3f} dB")
print(f"  chisq/ndf = {chisq2/dof2:.2f}")
print(f"  frequenza di taglio BODE2 = {ft_bode2:.0f} +/- {err_ft_bode2:.0f} Hz")
print("=================================================================\n")

# ============================================================
# GRAFICO DI BODE
# ============================================================
x_fit = np.linspace(np.log10(f_min), np.log10(f_max), 1000)

fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])

# ---------- Pannello superiore: dati e rette di fit ----------
ax[0].errorbar(np.log10(f), 20*np.log10(A),
               yerr=20.0 * A_err / (A * np.log(10.0)),
               fmt='o', ms=2, color='black', label='Dati')
# BODE 1 in ROSSO
ax[0].plot(x_fit, fit_lin(x_fit, *popt_bode1),
           '--', color='red', label='BODE 1 (m libero)')
# BODE 2 in BLU
ax[0].plot(x_fit, fit_lin_fixed_slope(x_fit, *popt_bode2),
           '--', color='blue', label='BODE 2 (m = -20)')
ax[0].set_ylabel(r'$20\log_{10}|A|$ [dB]')
ax[0].legend(loc='lower left')
# Limiti asse x ristretti alla regione mostrata
ax[0].set_xlim(3, 6)

# ---------- Pannello inferiore: residui e differenze tra modelli ----------
# Punti residui (neri) – senza legenda
ax[1].errorbar(x_bode, residui1, yerr=y_err_bode, fmt='o', ms=2, color='black')
# Differenza BODE2 - BODE1 (blu)
ax[1].plot(x_fit, fit_lin_fixed_slope(x_fit, *popt_bode2) - fit_lin(x_fit, *popt_bode1),
           linestyle='--', color='blue')
# Differenza BODE1 - BODE2 (rossa)
ax[1].plot(x_fit, fit_lin(x_fit, *popt_bode1) - fit_lin_fixed_slope(x_fit, *popt_bode2),
           linestyle='--', color='red')
ax[1].axhline(0, color='gray', linestyle='-', linewidth=0.8)
ax[1].set_ylabel('Residui [dB]')
ax[1].set_xlabel(r'$\log_{10}(f)$')
# Nessuna legenda nel pannello residui

# Limiti y dei residui
max_err = np.max(y_err_bode)
R_ylim = 2 * max(np.std(residui1), np.std(residui2)) + max_err
ax[1].set_ylim(-R_ylim, R_ylim)

# Salvataggio
plt.savefig(file + '_BODE.png',
            bbox_inches='tight',
            pad_inches=1,
            transparent=True,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            dpi='figure')
plt.show()