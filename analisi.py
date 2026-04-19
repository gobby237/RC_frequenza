#!/usr/bin/env python3
"""
RC Filter Frequency Response Analysis - Python Version
=======================================================
Conversione completa da ROOT C++ con 6 metodi di stima della frequenza di taglio.

Struttura:
- CONFIGURAZIONE: Parametri adattabili
- CARICAMENTO: Dati sperimentali
- 6 METODI: Stima di f_t
- OUTPUT: Terminale ordinato + grafici
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

class Config:
    """Parametri per l'esperimento RCF (passa-basso)."""
    
    # Componenti misurati
    R_meas = 3.871e3        # Resistenza (Ohm)
    C_meas = 9.36e-9        # Capacità (F)
    
    # Incertezze strumentali (schema: sqrt((frac% * val)^2 + (abs)^2))
    R_frac_err = 0.9 / 100
    R_abs_err = 0.003e3
    C_frac_err = 1.9 / 100
    C_abs_err = 2e-9
    scale_factor = 0.58     # Fattore conservativo per errori combinati
    
    # Incertezze oscilloscopio (V/div e µs/div)
    voltage_scale_unc = 0.04  # V per divisione (incertezza strumentale)
    time_scale_unc = 0.04     # µs per divisione
    
    # Range frequenza
    xmin = 0.1              # Freq. minima (kHz)
    xmax = 20e3             # Freq. massima (kHz)
    
    # Fit lineare locale (intorno a f_t)
    fitfl0 = 2.5            # Freq. inizio (kHz)
    fitfl1 = 6.7            # Freq. fine (kHz)
    
    # Scansione chi-quadro
    NSTEP = 50              # Numero step scansione
    ft_margin = 0.1         # Margine ±10% attorno f_t teorica


# ============================================================================
# FUNZIONI AUSILIARIE
# ============================================================================

def print_section(title):
    """Stampa header di sezione."""
    width = 75
    print(f"\n{'='*width}")
    print(f"  {title:<{width-4}}")
    print(f"{'='*width}\n")


def print_subsection(title):
    """Stampa header di sottosezione."""
    print(f"\n{title}")
    print("-" * 75)


def calc_errors(config):
    """Calcola errori su R e C combinando componenti frazionaria e assoluta."""
    eR = config.scale_factor * np.sqrt(
        (config.R_frac_err * config.R_meas)**2 + config.R_abs_err**2
    )
    eC = config.scale_factor * np.sqrt(
        (config.C_frac_err * config.C_meas)**2 + config.C_abs_err**2
    )
    return eR, eC


def calc_theoretical_ft(R, C, eR, eC):
    """f_t = 1/(2π RC) con propagazione errore."""
    ft = 1.0 / (2 * np.pi * R * C)
    eft = ft * np.sqrt((eR/R)**2 + (eC/C)**2)
    return ft, eft


def load_data(filename, config):
    """
    Carica file dati. Formato:
    T(µs)  V_in(V)  V_out(V)  V_scale  ∆T(µs)  ∆T_scale
    """
    data = {
        'f': [], 'A': [], 'eA': [],
        'phi_norm': [], 'ephi_norm': [],
        'tan_phi': [], 'e_tan_phi': [],
        'cot_phi': [], 'e_cot_phi': [],
    }
    
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = [float(x) for x in line.split()]
            if len(parts) < 6:
                continue
            
            T_us, V_in, V_out, V_scale, DT_us, DT_scale = parts
            
            # Skip punti non validi
            if T_us <= 0 or V_scale <= 0 or V_in <= 0 or V_out <= 0:
                continue
            
            # Frequenza (1/T in µs → kHz)
            f_kHz = 1e-3 / (1e-6 * T_us)
            data['f'].append(f_kHz)
            
            # Ampiezza
            A = V_out / V_in
            data['A'].append(A)
            
            # Errore ampiezza: propagazione A = V_out/V_in
            eV = V_scale * config.voltage_scale_unc  # Errore su entrambi i canali
            eA = A * np.sqrt((eV/V_in)**2 + (eV/V_out)**2)
            data['eA'].append(eA)
            
            # Fase
            phi_deg = 360.0 * DT_us / T_us
            phi_norm = phi_deg / 90.0
            data['phi_norm'].append(phi_norm)
            
            # Errore fase
            eDT = config.time_scale_unc * DT_scale * np.sqrt(2)
            ephi_deg = phi_deg * eDT / DT_us if DT_us > 0 else 0
            ephi_norm = ephi_deg / 90.0
            data['ephi_norm'].append(ephi_norm)
            
            # tan(φ) e cotan(φ)
            phi_rad = np.radians(phi_deg)
            tan_phi = np.tan(phi_rad)
            cot_phi = 1.0 / tan_phi if abs(tan_phi) > 1e-6 else np.inf
            data['tan_phi'].append(tan_phi)
            data['cot_phi'].append(cot_phi)
            
            # Errore su tan(φ): δ(tan φ) = (1/cos²φ) * δφ [rad]
            ephi_rad = np.radians(ephi_deg)
            e_tan_phi = abs(1.0 / np.cos(phi_rad)**2 * ephi_rad)
            e_cot_phi = abs(1.0 / np.sin(phi_rad)**2 * ephi_rad)
            data['e_tan_phi'].append(e_tan_phi)
            data['e_cot_phi'].append(e_cot_phi)
    
    # Converti a array
    for key in data:
        data[key] = np.array(data[key])
    
    data['n'] = len(data['f'])
    return data


# ============================================================================
# MODELLI TEORICI (PASSA-BASSO)
# ============================================================================

def A_theory(f, ft):
    """Ampiezza passa-basso: A(f) = 1/√(1 + (f/f_t)²)"""
    return 1.0 / np.sqrt(1.0 + (f/ft)**2)


def phi_theory(f, ft):
    """Fase normalizzata (÷90°): φ_n(f) = (2/π) arctan(f/f_t)"""
    return (2.0 / np.pi) * np.arctan(f / ft)


def chi2(y, y_th, ey):
    """Calcola χ² = Σ((y_i - y_th_i) / ey_i)²"""
    mask = ey > 0
    return np.sum(((y[mask] - y_th[mask]) / ey[mask])**2)


# ============================================================================
# METODO 1: A⁻² vs f² (fit lineare)
# ============================================================================

def method_1_ampquad(data, config):
    """Metodo 1: Linearizza 1/A² = 1 + (f/f_t)²"""
    
    f = data['f']
    A = data['A']
    eA = data['eA']
    
    # Trasforma: x = f², y = 1/A², ey = (2/A³) * eA
    x = f**2
    y = 1.0 / (A**2)
    ey = (2.0 / A**3) * eA
    
    # Fit: y = m*x + q
    popt, pcov = curve_fit(lambda x, m, q: m*x + q, x, y, sigma=ey, absolute_sigma=True)
    m, q = popt
    em, eq = np.sqrt(np.diag(pcov))
    
    # Estrai f_t: f_t = 1/√m
    ft = 1.0 / np.sqrt(m)
    eft = 0.5 * ft * (em / m)
    
    # Chi-quadro
    y_th = m * x + q
    chi2_val = chi2(y, y_th, ey)
    
    return {
        'name': 'A⁻² vs f²',
        'ft': ft, 'eft': eft,
        'm': m, 'em': em, 'q': q, 'eq': eq,
        'chi2': chi2_val, 'dof': len(x) - 2,
    }


# ============================================================================
# METODO 2: χ² minimizzato su A (free parameter)
# ============================================================================

def method_2_chi2_amp(data, config, ft_exp):
    """Metodo 2: Scannerizza χ²(f_t) e fit parabolico attorno al minimo."""
    
    f = data['f']
    A = data['A']
    eA = data['eA']
    
    # Range di scansione: ±margin% attorno a f_t teorica
    ft_min = ft_exp * (1 - config.ft_margin) / 1e3  # Converti Hz → kHz
    ft_max = ft_exp * (1 + config.ft_margin) / 1e3
    
    # Scansione
    ft_scan = np.linspace(ft_min, ft_max, config.NSTEP)
    chi2_scan = []
    for ft in ft_scan:
        A_th = A_theory(f, ft)
        c2 = chi2(A, A_th, eA)
        chi2_scan.append(c2)
    chi2_scan = np.array(chi2_scan)
    
    # Trova minimo
    idx_min = np.argmin(chi2_scan)
    chi2_min = chi2_scan[idx_min]
    
    # Fit parabolico attorno al minimo
    coeffs = np.polyfit(ft_scan, chi2_scan, 2)
    a, b, c = coeffs
    
    # Vertice parabola: f_t = -b/(2a)
    ft_best = -b / (2*a)
    chi2_vertex = a * ft_best**2 + b * ft_best + c
    
    # Incertezza: dove χ²(f_t) = χ²_min + 1
    # Risolvi: a*ft² + b*ft + (c - chi2_vertex - 1) = 0
    disc = b**2 - 4*a*(c - chi2_vertex - 1)
    if disc >= 0:
        ft_p = (-b + np.sqrt(disc)) / (2*a)
        ft_m = (-b - np.sqrt(disc)) / (2*a)
        eft = abs(ft_p - ft_best)
    else:
        eft = np.inf
    
    return {
        'name': 'χ² amplitude (free param)',
        'ft': ft_best, 'eft': eft,
        'chi2': chi2_vertex, 'dof': len(f),
        'ft_scan': ft_scan, 'chi2_scan': chi2_scan,
    }


# ============================================================================
# METODO 3: Fit lineare intorno a f_t (A)
# ============================================================================

def method_3_linear_local_A(data, config):
    """Metodo 3: Fit lineare A(f) su intervallo [fitfl0, fitfl1]."""
    
    f = data['f']
    A = data['A']
    eA = data['eA']
    
    # Seleziona intervallo
    mask = (f >= config.fitfl0) & (f <= config.fitfl1)
    f_fit = f[mask]
    A_fit = A[mask]
    eA_fit = eA[mask]
    
    # Fit: A = a + b*f
    popt, pcov = curve_fit(lambda x, a, b: a + b*x, f_fit, A_fit, 
                           sigma=eA_fit, absolute_sigma=True)
    a, b = popt
    ea, eb = np.sqrt(np.diag(pcov))
    
    # Estrai f_t: dove A = 1/√2
    y0 = 1.0 / np.sqrt(2.0)
    ft = (y0 - a) / b
    cov_factor = 2 * np.cov([np.repeat(a, len(f_fit)), np.repeat(b, len(f_fit))])[0, 1] / (ea * eb)
    cov_factor = min(max(cov_factor, -1), 1)  # Clamp a [-1, 1]
    eft = ft * np.sqrt((ea/(y0-a))**2 + (eb/b)**2 - 2*cov_factor*(ea/(y0-a))*(eb/b))
    
    # Chi-quadro
    A_th = a + b * f_fit
    chi2_val = chi2(A_fit, A_th, eA_fit)
    
    return {
        'name': 'Linear local A(f)',
        'ft': ft, 'eft': eft,
        'a': a, 'ea': ea, 'b': b, 'eb': eb,
        'chi2': chi2_val, 'dof': len(f_fit) - 2,
    }


# ============================================================================
# METODO 4: Fit lineare tan(φ) vs f
# ============================================================================

def method_4_tanphi(data, config):
    """Metodo 4: Fit lineare tan(φ) = (1/f_t)*f + 0, quindi f_t = 1/m."""
    
    f = data['f']
    y = data['tan_phi']
    ey = data['e_tan_phi']
    
    # Fit: y = m*f + q
    popt, pcov = curve_fit(lambda x, m, q: m*x + q, f, y, sigma=ey, absolute_sigma=True)
    m, q = popt
    em, eq = np.sqrt(np.diag(pcov))
    
    # Estrai f_t: Per passa-basso, tan(φ) ≈ f/f_t, quindi pendenza m ≈ 1/f_t
    # Attenzione: se la pendenza è negativa, prendi il valore assoluto
    ft = 1.0 / abs(m)
    eft = ft * (em / abs(m))
    
    # Chi-quadro
    y_th = m * f + q
    chi2_val = chi2(y, y_th, ey)
    
    return {
        'name': 'tan(φ) linear fit',
        'ft': ft, 'eft': eft,
        'm': m, 'em': em, 'q': q, 'eq': eq,
        'chi2': chi2_val, 'dof': len(f) - 2,
    }


# ============================================================================
# METODO 5: χ² minimizzato su φ (free parameter)
# ============================================================================

def method_5_chi2_phase(data, config, ft_exp):
    """Metodo 5: Analogo al metodo 2, ma sulla fase normalizzata."""
    
    f = data['f']
    phi = data['phi_norm']
    ephi = data['ephi_norm']
    
    # Range di scansione
    ft_min = ft_exp * (1 - config.ft_margin) / 1e3
    ft_max = ft_exp * (1 + config.ft_margin) / 1e3
    
    # Scansione
    ft_scan = np.linspace(ft_min, ft_max, config.NSTEP)
    chi2_scan = []
    for ft in ft_scan:
        phi_th = phi_theory(f, ft)
        c2 = chi2(phi, phi_th, ephi)
        chi2_scan.append(c2)
    chi2_scan = np.array(chi2_scan)
    
    # Minimo
    idx_min = np.argmin(chi2_scan)
    chi2_min = chi2_scan[idx_min]
    
    # Fit parabolico
    coeffs = np.polyfit(ft_scan, chi2_scan, 2)
    a, b, c = coeffs
    
    ft_best = -b / (2*a)
    chi2_vertex = a * ft_best**2 + b * ft_best + c
    
    disc = b**2 - 4*a*(c - chi2_vertex - 1)
    if disc >= 0:
        ft_p = (-b + np.sqrt(disc)) / (2*a)
        eft = abs(ft_p - ft_best)
    else:
        eft = np.inf
    
    return {
        'name': 'χ² phase (free param)',
        'ft': ft_best, 'eft': eft,
        'chi2': chi2_vertex, 'dof': len(f),
        'ft_scan': ft_scan, 'chi2_scan': chi2_scan,
    }


# ============================================================================
# METODO 6: Fit lineare intorno a f_t (φ)
# ============================================================================

def method_6_linear_local_phi(data, config):
    """Metodo 6: Fit lineare φ(f) su intervallo [fitfl0, fitfl1]."""
    
    f = data['f']
    phi = data['phi_norm']
    ephi = data['ephi_norm']
    
    # Seleziona intervallo
    mask = (f >= config.fitfl0) & (f <= config.fitfl1)
    f_fit = f[mask]
    phi_fit = phi[mask]
    ephi_fit = ephi[mask]
    
    # Fit: φ = a + b*f
    popt, pcov = curve_fit(lambda x, a, b: a + b*x, f_fit, phi_fit,
                           sigma=ephi_fit, absolute_sigma=True)
    a, b = popt
    ea, eb = np.sqrt(np.diag(pcov))
    
    # Estrai f_t: dove φ = 0.5
    y0 = 0.5
    ft = (y0 - a) / b
    cov_factor = np.cov([np.repeat(a, len(f_fit)), np.repeat(b, len(f_fit))])[0, 1] / (ea * eb)
    cov_factor = min(max(cov_factor, -1), 1)
    eft = ft * np.sqrt((ea/(y0-a))**2 + (eb/b)**2 - 2*cov_factor*(ea/(y0-a))*(eb/b))
    
    # Chi-quadro
    phi_th = a + b * f_fit
    chi2_val = chi2(phi_fit, phi_th, ephi_fit)
    
    return {
        'name': 'Linear local φ(f)',
        'ft': ft, 'eft': eft,
        'a': a, 'ea': ea, 'b': b, 'eb': eb,
        'chi2': chi2_val, 'dof': len(f_fit) - 2,
    }


# ============================================================================
# VISUALIZZAZIONE
# ============================================================================

def plot_results(data, config, results, ft_exp, eft_exp):
    """Crea 2 figure: (1) A+φ+residui, (2) Confronto metodi."""
    
    f = data['f']
    A = data['A']
    eA = data['eA']
    phi_norm = data['phi_norm']
    ephi_norm = data['ephi_norm']
    
    # Prendi risultati dei metodi χ² (migliori)
    m2 = results[1]
    m5 = results[4]
    
    # ========== FIGURA 1: A + Φ + Residui ==========
    fig1 = plt.figure(figsize=(14, 11))
    
    # Sottofigura 1: A e Φ (stessi assi X)
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_xscale('log')
    
    # Curve teoriche
    A_th = A_theory(f, m2['ft'])
    phi_th = phi_theory(f, m5['ft'])
    
    # Ampiezza (asse sinistro)
    ax1.errorbar(f, A, yerr=eA, fmt='o', color='black', markersize=6,
                 label=f"A (meas.)", capsize=2, alpha=0.7)
    ax1.plot(f, A_th, '-k', linewidth=2.5, label=f"A fit: $f_t={m2['ft']:.3f}$ kHz")
    ax1.set_ylabel('Amplitude A', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, which='both', alpha=0.3)
    
    # Fase (asse destro)
    ax1b = ax1.twinx()
    ax1b.errorbar(f, phi_norm, yerr=ephi_norm, fmt='s', color='red', markersize=6,
                  label=f"φ norm. (meas.)", capsize=2, alpha=0.7)
    ax1b.plot(f, phi_th, '--r', linewidth=2.5, label=f"φ fit: $f_t={m5['ft']:.3f}$ kHz")
    ax1b.set_ylabel('Phase (×90°)', fontsize=11, fontweight='bold', color='red')
    ax1b.tick_params(axis='y', labelcolor='red')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='center right', fontsize=10)
    
    # Sottofigura 2: Residui A
    ax2 = plt.subplot(3, 1, 2)
    ax2.set_xscale('log')
    res_A = A - A_th
    ax2.errorbar(f, res_A, yerr=eA, fmt='o', color='black', markersize=6,
                 label='Residuals', capsize=2)
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    ax2.set_ylabel('A (meas. - fit)', fontsize=11, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Sottofigura 3: Residui φ
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_xscale('log')
    res_phi = phi_norm - phi_th
    ax3.errorbar(f, res_phi, yerr=ephi_norm, fmt='s', color='red', markersize=6,
                 label='Residuals', capsize=2)
    ax3.axhline(0, color='r', linestyle='--', linewidth=1)
    ax3.set_xlabel('Frequency (kHz)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Δφ/90° (meas. - fit)', fontsize=11, fontweight='bold', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    ax3.grid(True, which='both', alpha=0.3)
    ax3.legend(fontsize=10)
    
    fig1.suptitle('RC Filter Response: Amplitude + Phase + Residuals', 
                  fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # ========== FIGURA 2: Confronto 6 metodi ==========
    fig2 = plt.figure(figsize=(12, 9))
    
    names = [r['name'] for r in results]
    fts = np.array([r['ft'] for r in results])
    efts = np.array([r['eft'] for r in results])
    
    x_pos = np.arange(1, len(fts) + 1)
    
    # Pannello 1: Frequenza di taglio
    ax4 = plt.subplot(2, 1, 1)
    ax4.errorbar(x_pos, fts, yerr=efts, fmt='o', markersize=9, capsize=5,
                 color='blue', elinewidth=2.5, label='Measured')
    ax4.axhline(ft_exp/1e3, color='green', linestyle='--', linewidth=2.5,
                label=f'Theory: {ft_exp/1e3:.4f} kHz')
    ax4.fill_between([0.5, len(fts)+0.5],
                      (ft_exp-eft_exp)/1e3, (ft_exp+eft_exp)/1e3,
                      alpha=0.2, color='green')
    ax4.set_ylabel('$f_t$ (kHz)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=11, loc='best')
    ax4.set_xlim(0.5, len(fts)+0.5)
    
    # Pannello 2: Errori relativi
    ax5 = plt.subplot(2, 1, 2)
    err_rel = 100 * efts / fts
    colors = ['green' if e < 10 else 'orange' if e < 20 else 'red' for e in err_rel]
    bars = ax5.bar(x_pos, err_rel, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.axhline(100*eft_exp/ft_exp, color='purple', linestyle='--', linewidth=2.5,
                label=f'Theory: {100*eft_exp/ft_exp:.1f}%')
    ax5.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend(fontsize=11, loc='best')
    ax5.set_xlim(0.5, len(fts)+0.5)
    
    fig2.suptitle('Comparison of 6 Methods', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig1, fig2


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()
    
    print_section("RC FILTER FREQUENCY RESPONSE ANALYSIS")
    print("Python version: Complete 6-method comparison")
    print()
    
    # === Fase 1: Parametri teorici ===
    print_section("THEORETICAL PARAMETERS")
    eR, eC = calc_errors(config)
    ft_exp, eft_exp = calc_theoretical_ft(config.R_meas, config.C_meas, eR, eC)
    
    print(f"R = ({config.R_meas:.3e} ± {eR:.3e}) Ω")
    print(f"C = ({config.C_meas:.3e} ± {eC:.3e}) F")
    print(f"f_t (theory) = ({ft_exp/1e3:.4f} ± {eft_exp/1e3:.4f}) kHz")
    print()
    
    # === Fase 2: Caricamento dati ===
    print_section("DATA LOADING")
    datafile = "RCF_PB.txt"
    print(f"File: {datafile}")
    
    data = load_data(datafile, config)
    print(f"✓ Loaded {data['n']} points")
    print(f"  Frequency range: {data['f'].min():.3f} – {data['f'].max():.1f} kHz")
    print(f"  Amplitude range: {data['A'].min():.4f} – {data['A'].max():.4f}")
    print(f"  Phase range: {data['phi_norm'].min():.3f} – {data['phi_norm'].max():.3f} (×90°)")
    print()
    
    # === Fase 3: 6 Metodi ===
    print_section("6 METHODS FOR CUTOFF FREQUENCY ESTIMATION")
    
    results = []
    
    # Metodo 1
    print_subsection("[1/6] METHOD 1: Amplitude Squared (A⁻² vs f²)")
    m1 = method_1_ampquad(data, config)
    results.append(m1)
    print(f"  Slope: m = {m1['m']:.4e} ± {m1['em']:.4e}")
    print(f"  Intercept: q = {m1['q']:.4e} ± {m1['eq']:.4e}")
    print(f"  χ²/dof = {m1['chi2']:.2f}/{m1['dof']}")
    print(f"  → f_t = {m1['ft']:.4f} ± {m1['eft']:.4f} kHz  ({100*m1['eft']/m1['ft']:.1f}%)")
    
    # Metodo 2
    print_subsection("[2/6] METHOD 2: χ² Minimization on Amplitude")
    m2 = method_2_chi2_amp(data, config, ft_exp)
    results.append(m2)
    print(f"  Scan range: ±{config.ft_margin*100:.0f}% around theory")
    print(f"  χ²_min = {m2['chi2']:.2f}")
    print(f"  → f_t = {m2['ft']:.4f} ± {m2['eft']:.4f} kHz  ({100*m2['eft']/m2['ft']:.1f}%)")
    
    # Metodo 3
    print_subsection("[3/6] METHOD 3: Linear Local Fit (Amplitude)")
    m3 = method_3_linear_local_A(data, config)
    results.append(m3)
    print(f"  Fit range: {config.fitfl0:.2f} – {config.fitfl1:.2f} kHz")
    print(f"  Slope: b = {m3['b']:.4e} ± {m3['eb']:.4e}")
    print(f"  χ²/dof = {m3['chi2']:.2f}/{m3['dof']}")
    print(f"  → f_t = {m3['ft']:.4f} ± {m3['eft']:.4f} kHz  ({100*m3['eft']/m3['ft']:.1f}%)")
    
    # Metodo 4
    print_subsection("[4/6] METHOD 4: Linear tan(φ)")
    m4 = method_4_tanphi(data, config)
    results.append(m4)
    print(f"  Slope: m = {m4['m']:.4e} ± {m4['em']:.4e}")
    print(f"  χ²/dof = {m4['chi2']:.2f}/{m4['dof']}")
    print(f"  → f_t = {m4['ft']:.4f} ± {m4['eft']:.4f} kHz  ({100*m4['eft']/m4['ft']:.1f}%)")
    
    # Metodo 5
    print_subsection("[5/6] METHOD 5: χ² Minimization on Phase")
    m5 = method_5_chi2_phase(data, config, ft_exp)
    results.append(m5)
    print(f"  Scan range: ±{config.ft_margin*100:.0f}% around theory")
    print(f"  χ²_min = {m5['chi2']:.2f}")
    print(f"  → f_t = {m5['ft']:.4f} ± {m5['eft']:.4f} kHz  ({100*m5['eft']/m5['ft']:.1f}%)")
    
    # Metodo 6
    print_subsection("[6/6] METHOD 6: Linear Local Fit (Phase)")
    m6 = method_6_linear_local_phi(data, config)
    results.append(m6)
    print(f"  Fit range: {config.fitfl0:.2f} – {config.fitfl1:.2f} kHz")
    print(f"  Slope: b = {m6['b']:.4e} ± {m6['eb']:.4e}")
    print(f"  χ²/dof = {m6['chi2']:.2f}/{m6['dof']}")
    print(f"  → f_t = {m6['ft']:.4f} ± {m6['eft']:.4f} kHz  ({100*m6['eft']/m6['ft']:.1f}%)")
    
    # === Fase 4: Riepilogo ===
    print_section("SUMMARY AND RECOMMENDATIONS")
    
    fts = np.array([r['ft'] for r in results])
    efts = np.array([r['eft'] for r in results])
    err_rels = 100 * efts / fts
    
    print(f"{'Method':<35} {'f_t (kHz)':<22} {'Error (%)':<12}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']:<32} {r['ft']:>8.4f} ± {r['eft']:>8.4f}   {err_rels[i-1]:>8.2f}%")
    
    print()
    print(f"{'THEORY (R-C):':<35} {ft_exp/1e3:>8.4f} ± {eft_exp/1e3:>8.4f}   {100*eft_exp/ft_exp:>8.2f}%")
    
    best_idx = np.argmin(err_rels)
    print()
    print(f"{'='*70}")
    print(f"✓ BEST METHOD: #{best_idx+1} ({results[best_idx]['name']})")
    print(f"  Recommended: f_t = ({fts[best_idx]:.4f} ± {efts[best_idx]:.4f}) kHz")
    print(f"{'='*70}")
    print()
    
    # === Fase 5: Grafici ===
    print_section("PLOTTING")
    fig1, fig2 = plot_results(data, config, results, ft_exp, eft_exp)
    print("✓ Figure 1: Response (A + φ + residuals)")
    print("✓ Figure 2: Methods comparison")
    print()
    
    plt.show()
    
    print_section("ANALYSIS COMPLETE")
    print("Figures shown; modify Config class for different parameters.")


if __name__ == "__main__":
    main()