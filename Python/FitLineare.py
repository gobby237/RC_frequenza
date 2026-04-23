import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler

# ------------------------------------------
# Impostazioni grafiche stile ROOT (opzionale)
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
})
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# ------------------------------
# Funzioni di fit
# ------------------------------
def fit_lin(x, m, q):
    """Modello lineare: y = m*x + q"""
    return m * x + q

def fit_nlin_A(x, ft):
    """Modello teorico ampiezza filtro RC passa-basso"""
    return 1.0 / np.sqrt(1.0 + (x / ft)**2)

def fit_nlin_phi(x, ft):
    """Modello teorico fase normalizzata filtro RC"""
    return (2.0 / np.pi) * np.arctan(x / ft)

# ------------------------------
# Parametri iniziali
# ------------------------------
file = 'RCF_PB'
inputname = file + '.txt'

# Frequenza di taglio iniziale attorno alla quale eseguire il fit locale
ft_init = 4392.0   # Hz

# --- INTERVALLI DI FIT INDIPENDENTI (modifica qui i valori) ---
scan_A   = 2200.0   # semi‑intervallo per il modulo A
scan_phi = 1800.0    # semi‑intervallo per la fase

# Errori di lettura strumentali (valori fissi come nell'originale)
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
phi = (4 * delta_t * f * 1e-6)   # normalizzato a 1
phi_fs = 2 * np.pi * f * phi_fs * 1e-6 / (np.pi / 2.)

# ------------------------------
# Calcolo errori (propagazione)
# ------------------------------
Vin_errL = 0.041
V_errL = (V_fs / 10.0) * 0.41

phi_errL = (phi_fs / 10.0) * 0.41 * np.sqrt(2.0)
mask_high = f > 10600.0
phi_errL[mask_high] = (phi_fs[mask_high] * 1e-6 / 10.0) * 0.58 * np.sqrt(2.0)

A_err = A * np.sqrt(
    (V_errL / Vout)**2 +
    (Vin_errL / Vin)**2 +
    2.0 * (0.03 * 0.41)**2
)
A_err[(f>10) & (f<10600)] = A[(f>10) & (f<10600)] * np.sqrt(
    (V_errL[(f>10) & (f<10600)] / Vout[(f>10) & (f<10600)])**2 +
    (Vin_errL / Vin[(f>10) & (f<10600)])**2
)

# ------------------------------
# Maschere separate per i due fit
# ------------------------------
f_min_A = ft_init - scan_A
f_max_A = ft_init + scan_A
mask_A = (f > f_min_A) & (f < f_max_A)
x_A = f[mask_A]
y_A = A[mask_A]
y_err_A = A_err[mask_A]

f_min_phi = ft_init - scan_phi
f_max_phi = ft_init + scan_phi
mask_phi = (f > f_min_phi) & (f < f_max_phi)
x_phi = f[mask_phi]
z_phi = phi[mask_phi]
z_err_phi = phi_errL[mask_phi]

print(f"Intervallo fit A:   {f_min_A:.1f} -- {f_max_A:.1f} Hz")
print(f"Intervallo fit phi: {f_min_phi:.1f} -- {f_max_phi:.1f} Hz")

N_A = len(x_A)
N_phi = len(x_phi)

# ------------------------------
# Fit lineare ponderato su A e phi
# ------------------------------
popt_A, pcov_A = curve_fit(
    fit_lin, x_A, y_A,
    p0=[-0.1, 1.0],
    method='lm',
    sigma=y_err_A,
    absolute_sigma=True
)

popt_phi, pcov_phi = curve_fit(
    fit_lin, x_phi, z_phi,
    p0=[0.1, 0.0],
    method='lm',
    sigma=z_err_phi,
    absolute_sigma=True
)

# ------------------------------
# Analisi dei residui e bontà del fit
# ------------------------------
residu_A = y_A - fit_lin(x_A, *popt_A)
residu_phi = z_phi - fit_lin(x_phi, *popt_phi)

perr_A = np.sqrt(np.diag(pcov_A))
perr_phi = np.sqrt(np.diag(pcov_phi))

chisq_A = np.sum((residu_A / y_err_A)**2)
chisq_phi = np.sum((residu_phi / z_err_phi)**2)
dof_A = N_A - 2
dof_phi = N_phi - 2

# ------------------------------
# Calcolo frequenza di taglio dai fit lineari
# ------------------------------
ft_A = (1.0 / np.sqrt(2.0) - popt_A[1]) / popt_A[0]
ft_phi = (0.5 - popt_phi[1]) / popt_phi[0]

err_ft_A = np.sqrt(
    (ft_A / popt_A[0])**2 * perr_A[0]**2 +
    (1.0 / popt_A[0])**2 * perr_A[1]**2 +
    2.0 * ft_A / popt_A[0]**2 * pcov_A[0, 1]
)
err_ft_phi = np.sqrt(
    (ft_phi / popt_phi[0])**2 * perr_phi[0]**2 +
    (1.0 / popt_phi[0])**2 * perr_phi[1]**2 +
    2.0 * ft_phi / popt_phi[0]**2 * pcov_phi[0, 1]
)

# ------------------------------
# Stampa risultati
# ------------------------------
print("\n ============== BEST FIT Lineare Locale ====================")
print("\n ================== Modulo A ========================")
print(f"  slope m = {popt_A[0]:.3e} +/- {perr_A[0]:.1e} s")
print(f"  intercept q = {popt_A[1]:.3f} +/- {perr_A[1]:.3f}")
print(f"  chisq/ndf = {chisq_A/dof_A:.2f}")
print(f"  frequenza di taglio = {ft_A:.0f} +/- {err_ft_A:.0f} Hz")
print("\n ================== Fase phi ========================")
print(f"  slope m = {popt_phi[0]:.3e} +/- {perr_phi[0]:.1e} s")
print(f"  intercept q = {popt_phi[1]:.3f} +/- {perr_phi[1]:.3f}")
print(f"  chisq/ndf = {chisq_phi/dof_phi:.2f}")
print(f"  frequenza di taglio = {ft_phi:.0f} +/- {err_ft_phi:.0f} Hz")
print("=============================================================\n")

ft_stima = [
    ['A,loc', ft_A, err_ft_A],
    ['phi,loc', ft_phi, err_ft_phi]
]

# ------------------------------
# Grafico dati + fit + residui
# ------------------------------
x_fit_A = np.linspace(f_min_A, f_max_A, 500)
x_fit_phi = np.linspace(f_min_phi, f_max_phi, 500)

fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])

# Pannello superiore: dati e rette di fit (colori modificati: A=rosso, phi=blu)
ax[0].errorbar(f, A, yerr=A_err, fmt='o', ms=2, color='black', label='Modulo')
ax[0].errorbar(f, phi, yerr=phi_errL, fmt='o', ms=2, color='red', label='Fase')
ax[0].plot(x_fit_A, fit_lin(x_fit_A, *popt_A), '--', color='black', label='Fit A')
ax[0].plot(x_fit_phi, fit_lin(x_fit_phi, *popt_phi), '--', color='red', label='Fit $\phi$')

# Bande di sfondo più sbiadite (alpha=0.05)
ax[0].axvspan(f_min_A, f_max_A, alpha=0.05, color='black')
ax[0].axvspan(f_min_phi, f_max_phi, alpha=0.05, color='red')

ax[0].set_xlim(min(f_min_A, f_min_phi) - max(scan_A, scan_phi)/2,
               max(f_max_A, f_max_phi) + max(scan_A, scan_phi)/2)
ax[0].set_ylim(0, 1)
ax[0].set_ylabel(r'$\left|A\right|$ / $\phi_{norm}$')
ax[0].legend(loc='lower right')

# Pannello inferiore: residui (colori rosso e blu) + bande di sfondo
ax[1].errorbar(x_A, residu_A, yerr=y_err_A, fmt='o', ms=2, color='black')
ax[1].errorbar(x_phi, residu_phi, yerr=z_err_phi, fmt='o', ms=2, color='red')
ax[1].axhline(0, color='gray', linestyle='-', linewidth=0.8)

# Stesse bande di sfondo anche nel pannello dei residui
ax[1].axvspan(f_min_A, f_max_A, alpha=0.05, color='black')
ax[1].axvspan(f_min_phi, f_max_phi, alpha=0.05, color='red')

max_err = max(np.max(y_err_A) if len(y_err_A)>0 else 0,
              np.max(z_err_phi) if len(z_err_phi)>0 else 0)
R_ylim = 2 * max(np.std(residu_A), np.std(residu_phi)) + max_err
ax[1].set_ylim(-R_ylim, R_ylim)
ax[1].set_ylabel('Residui')
ax[1].set_xlabel('Frequenza [Hz]')

plt.savefig(file + '_loc.png',
            bbox_inches='tight',
            pad_inches=1,
            transparent=True,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            dpi='figure')
plt.show()