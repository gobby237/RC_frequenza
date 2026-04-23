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
# Modelli di fit (con parametri aggiuntivi)
# ------------------------------
def fit_A_offset(f, ft, A0):
    """Ampiezza con fattore di scala A0"""
    return A0 / np.sqrt(1.0 + (f / ft)**2)

def fit_phi_offset(f, ft, phi0):
    """Fase normalizzata con possibile offset costante"""
    return (2.0 / np.pi) * np.arctan(f / ft) + phi0

# ------------------------------
# Parametri iniziali
# ------------------------------
file = 'RCF_PB'
inputname = file + '.txt'

ft_guess = 4392.0      # Hz
A0_guess = 1.0         # valore atteso asintotico
phi0_guess = 0.0       # offset iniziale

# ------------------------------
# Caricamento dati
# ------------------------------
data = np.loadtxt(inputname).T

f = 1.0 / (data[0] * 1e-6)       # frequenza in Hz
Vin = data[1]
Vout = data[2]
V_fs = data[3]
delta_t = data[4]                # µs
phi_fs = 2 * np.pi * f * np.array(data[5]) * 1e-6 / (np.pi / 2.)

A = Vout / Vin
phi = 4.0 * delta_t * f * 1e-6   # fase normalizzata (0 a 1)

# ------------------------------
# Errori (formule originali)
# ------------------------------
Vin_errL = 0.041
V_errL = (V_fs / 10.0) * 0.41

phi_errL = (phi_fs / 10.0) * 0.41 * np.sqrt(2.0)
mask_high = f > 10600.0
phi_errL[mask_high] = (phi_fs[mask_high] / 10.0) * 0.58 * np.sqrt(2.0)

A_err = A * np.sqrt(
    (V_errL / Vout)**2 +
    (Vin_errL / Vin)**2 +
    2.0 * (0.03 * 0.41)**2
)
A_err[(f > 10) & (f < 10600)] = A[(f > 10) & (f < 10600)] * np.sqrt(
    (V_errL[(f > 10) & (f < 10600)] / Vout[(f > 10) & (f < 10600)])**2 +
    (Vin_errL / Vin[(f > 10) & (f < 10600)])**2
)

# ------------------------------
# Fit NON LINEARE con parametri di scala e offset
# ------------------------------
# Fit ampiezza con A0
popt_A, pcov_A = curve_fit(
    fit_A_offset, f, A,
    p0=[ft_guess, A0_guess],
    sigma=A_err,
    absolute_sigma=True
)

# Fit fase con offset
popt_phi, pcov_phi = curve_fit(
    fit_phi_offset, f, phi,
    p0=[ft_guess, phi0_guess],
    sigma=phi_errL,
    absolute_sigma=True
)

ft_A, A0 = popt_A
ft_phi, phi0 = popt_phi

# Errori sui parametri
err_ft_A = np.sqrt(pcov_A[0, 0])
err_A0   = np.sqrt(pcov_A[1, 1])
err_ft_phi = np.sqrt(pcov_phi[0, 0])
err_phi0  = np.sqrt(pcov_phi[1, 1])

# Residui e chi2
res_A = A - fit_A_offset(f, *popt_A)
res_phi = phi - fit_phi_offset(f, *popt_phi)

chi2_A = np.sum((res_A / A_err)**2)
chi2_phi = np.sum((res_phi / phi_errL)**2)

N = len(f)
dof_A = N - 2      # due parametri: ft, A0
dof_phi = N - 2    # due parametri: ft, phi0

# ------------------------------
# Stampa risultati
# ------------------------------
print("\n============= FIT NON LINEARE CON PARAMETRI AGGIUNTIVI =============")
print("\n--- Ampiezza (modello con A0) ---")
print(f"  ft = {ft_A:.1f} ± {err_ft_A:.1f} Hz")
print(f"  A0 = {A0:.4f} ± {err_A0:.4f}")
print(f"  χ²/ndf = {chi2_A/dof_A:.2f}")
print("  Matrice di covarianza:")
print(pcov_A)

print("\n--- Fase (modello con offset φ₀) ---")
print(f"  ft  = {ft_phi:.1f} ± {err_ft_phi:.1f} Hz")
print(f"  φ₀  = {phi0:.4f} ± {err_phi0:.4f}")
print(f"  χ²/ndf = {chi2_phi/dof_phi:.2f}")
print("  Matrice di covarianza:")
print(pcov_phi)
print("========================================================\n")

# ------------------------------
# Grafico
# ------------------------------
f_fit = np.logspace(np.log10(80), np.log10(250000), 2000)

fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])

# Pannello superiore: dati e curve di fit
ax[0].errorbar(f, A, yerr=A_err, fmt='o', ms=2, color='red', label='Modulo')
ax[0].errorbar(f, phi, yerr=phi_errL, fmt='o', ms=2, color='blue', label='Fase')
ax[0].plot(f_fit, fit_A_offset(f_fit, *popt_A), '--', color='red',
           label=r'Fit $A$ (con $A_0$)')
ax[0].plot(f_fit, fit_phi_offset(f_fit, *popt_phi), '--', color='blue',
           label=r'Fit $\phi$ (con $\phi_0$)')
ax[0].set_xscale('log')
ax[0].set_xlim(80, 250000)
ax[0].set_ylim(0, 1)
ax[0].set_ylabel(r'$\left|A\right|$ / $\phi_{norm}$')
ax[0].legend(loc='lower right')

# Finestrella con le formule aggiornate
ax[0].text(0.05, 0.25,
           r'$\displaystyle A = \frac{A_0}{\sqrt{1+(f/f_t)^2}}$' + '\n'
           r'$\displaystyle \phi = \frac{2}{\pi}\arctan\frac{f}{f_t} + \phi_0$',
           transform=ax[0].transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Pannello inferiore: residui
ax[1].errorbar(f, res_A, yerr=A_err, fmt='o', ms=2, color='red')
ax[1].errorbar(f, res_phi, yerr=phi_errL, fmt='o', ms=2, color='blue')
ax[1].axhline(0, color='gray', linestyle='-', linewidth=0.8)
ax[1].set_xscale('log')
ax[1].set_ylabel('Residui')
ax[1].set_xlabel('Frequenza [Hz]')

max_err = max(np.max(A_err), np.max(phi_errL))
R = 5 * max(np.std(res_A), np.std(res_phi)) + max_err
ax[1].set_ylim(-R, R)

plt.savefig(file + '_NL_offset.png',
            bbox_inches='tight',
            pad_inches=1,
            transparent=True,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            dpi='figure')
plt.show()