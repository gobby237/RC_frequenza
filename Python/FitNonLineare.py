#!/usr/bin/env python3
"""
RC Filter - χ² Minimization Analysis
=====================================
ONLY 2 methods: χ² minimization on Amplitude and Phase
Saves all plots to PNG and TXT files
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend non-interattivo: niente GUI, solo salvataggio file
import matplotlib.pyplot as plt
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

class Config:
    """Parametri dell'esperimento RCF."""
    
    # Componenti misurati
    R_meas = 3.871e3        # Resistenza (Ohm)
    C_meas = 9.36e-9        # Capacità (F)
    
    # Incertezze strumentali
    R_frac_err = 0.9 / 100
    R_abs_err = 0.003e3
    C_frac_err = 1.9 / 100
    C_abs_err = 2e-9
    scale_factor = 0.58
    
    # Incertezze oscilloscopio
    voltage_scale_unc = 0.04
    time_scale_unc = 0.04
    
    # Scansione chi-quadro
    NSTEP = 50
    ft_margin = 0.1


# ============================================================================
# FUNZIONI AUSILIARIE
# ============================================================================

def print_header(title):
    width = 75
    print(f"\n{'='*width}")
    print(f"  {title:<{width-4}}")
    print(f"{'='*width}\n")


def calc_errors(config):
    """Calcola errori su R e C."""
    eR = config.scale_factor * np.sqrt(
        (config.R_frac_err * config.R_meas)**2 + config.R_abs_err**2
    )
    eC = config.scale_factor * np.sqrt(
        (config.C_frac_err * config.C_meas)**2 + config.C_abs_err**2
    )
    return eR, eC


def calc_theoretical_ft(R, C, eR, eC):
    """Frequenza di taglio teorica: f_t = 1/(2π RC)."""
    ft = 1.0 / (2 * np.pi * R * C)
    eft = ft * np.sqrt((eR/R)**2 + (eC/C)**2)
    return ft, eft


def load_data(filename, config):
    """
    Carica dati dal file.
    Formato: T(µs) V_in V_out V_scale ∆T ∆T_scale
    
    ERRORI CALCOLATI:
    - eA (ampiezza): A * sqrt((eV/V_in)² + (eV/V_out)²)
      dove eV = V_scale * 0.04 (incertezza oscilloscopio)
    
    - ephi_norm (fase normalizzata): 
      ephi_deg = phi_deg * eDT / DT
      dove eDT = 0.04 * DT_scale * sqrt(2)
      ephi_norm = ephi_deg / 90
    """
    data = {
        'f': [], 'A': [], 'eA': [],
        'phi_norm': [], 'ephi_norm': [],
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
            
            # Skip punti invalidi
            if T_us <= 0 or V_scale <= 0 or V_in <= 0 or V_out <= 0:
                continue
            
            # === FREQUENZA ===
            f_kHz = 1e-3 / (1e-6 * T_us)
            data['f'].append(f_kHz)
            
            # === AMPIEZZA E SUO ERRORE ===
            # A = V_out / V_in
            # Propagazione: eA/A = sqrt((eV_out/V_out)² + (eV_in/V_in)²)
            # dove eV = V_scale * config.voltage_scale_unc
            A = V_out / V_in
            data['A'].append(A)
            
            eV = V_scale * config.voltage_scale_unc  # Errore per canale
            eA = A * np.sqrt((0.04/V_in)**2 + (eV/V_out)**2 + 2*pow(0.015,2))
            data['eA'].append(eA)
            
            # === FASE E SUO ERRORE ===
            # phi_deg = 360 * DT / T
            # Propagazione: ephi/phi = eDT / DT
            # dove eDT = time_scale_unc * DT_scale * sqrt(2)
            phi_deg = 360.0 * DT_us / T_us
            phi_norm = phi_deg / 90.0
            data['phi_norm'].append(phi_norm)
            
            eDT = config.time_scale_unc * DT_scale * np.sqrt(2)
            ephi_deg = phi_deg * eDT / DT_us if DT_us > 0 else 0
            ephi_norm = ephi_deg / 90.0
            data['ephi_norm'].append(ephi_norm)
    
    # Converti a numpy array
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
    """Fase normalizzata: φ_n(f) = (2/π) arctan(f/f_t)"""
    return (2.0 / np.pi) * np.arctan(f / ft)


def chi2(y, y_th, ey):
    """Calcola χ² = Σ((y_i - y_th_i) / ey_i)²"""
    mask = ey > 0
    return np.sum(((y[mask] - y_th[mask]) / ey[mask])**2)


# ============================================================================
# METODO 1: χ² MINIMIZZATO SU AMPIEZZA
# ============================================================================

def chi2_minimize_amplitude(data, config, ft_exp):
    """
    Minimizza χ² scansionando su f_t per l'ampiezza.
    Il loop su ft_scan è vettorizzato con numpy (broadcasting 2D).
    """

    f  = data['f']
    A  = data['A']
    eA = data['eA']

    ft_min = ft_exp * (1 - config.ft_margin) / 1e3
    ft_max = ft_exp * (1 + config.ft_margin) / 1e3

    ft_scan = np.linspace(ft_min, ft_max, config.NSTEP)

    # Vettorizzato: ft_scan (N,) × f (M,) → A_th (N, M)
    A_th = 1.0 / np.sqrt(1.0 + (f[np.newaxis, :] / ft_scan[:, np.newaxis])**2)
    mask = eA > 0
    chi2_scan = np.sum(((A[np.newaxis, mask] - A_th[:, mask]) / eA[np.newaxis, mask])**2, axis=1)

    # Fit parabolico
    coeffs = np.polyfit(ft_scan, chi2_scan, 2)
    a, b, c = coeffs

    ft_best     = -b / (2*a)
    chi2_vertex = a * ft_best**2 + b * ft_best + c

    discriminant = b**2 - 4*a*(c - chi2_vertex - 1)
    eft = abs((-b + np.sqrt(discriminant)) / (2*a) - ft_best) if discriminant >= 0 else np.inf

    return {
        'name':      'χ² Minimization - Amplitude',
        'ft':        ft_best,
        'eft':       eft,
        'chi2_min':  chi2_vertex,
        'dof':       len(f),
        'ft_scan':   ft_scan,
        'chi2_scan': chi2_scan,
        'coeffs':    coeffs,
    }


# ============================================================================
# METODO 2: χ² MINIMIZZATO SU FASE
# ============================================================================

def chi2_minimize_phase(data, config, ft_exp):
    """
    Minimizza χ² scansionando su f_t per la fase normalizzata.
    Il loop su ft_scan è vettorizzato con numpy (broadcasting 2D).
    """

    f    = data['f']
    phi  = data['phi_norm']
    ephi = data['ephi_norm']

    ft_min = ft_exp * (1 - config.ft_margin) / 1e3
    ft_max = ft_exp * (1 + config.ft_margin) / 1e3

    ft_scan = np.linspace(ft_min, ft_max, config.NSTEP)

    # Vettorizzato: ft_scan (N,) × f (M,) → phi_th (N, M)
    phi_th = (2.0 / np.pi) * np.arctan(f[np.newaxis, :] / ft_scan[:, np.newaxis])
    mask = ephi > 0
    chi2_scan = np.sum(((phi[np.newaxis, mask] - phi_th[:, mask]) / ephi[np.newaxis, mask])**2, axis=1)

    # Fit parabolico
    coeffs = np.polyfit(ft_scan, chi2_scan, 2)
    a, b, c = coeffs

    ft_best     = -b / (2*a)
    chi2_vertex = a * ft_best**2 + b * ft_best + c

    discriminant = b**2 - 4*a*(c - chi2_vertex - 1)
    eft = abs((-b + np.sqrt(discriminant)) / (2*a) - ft_best) if discriminant >= 0 else np.inf

    return {
        'name':      'χ² Minimization - Phase',
        'ft':        ft_best,
        'eft':       eft,
        'chi2_min':  chi2_vertex,
        'dof':       len(f),
        'ft_scan':   ft_scan,
        'chi2_scan': chi2_scan,
        'coeffs':    coeffs,
    }


# ============================================================================
# VISUALIZZAZIONE E SALVATAGGIO
# ============================================================================

# --- Parametri grafici centralizzati ---
# Dimensioni figure (larghezza, altezza) in pollici
FIG1_W, FIG1_H   = 14, 11   # Risposta + Residui
FIG2_W, FIG2_H   = 10,  6   # χ² Ampiezza
FIG3_W, FIG3_H   = 10,  6   # χ² Fase

# Font
FS_TITLE   = 24   # Titolo figura
FS_LABEL   = 20   # Etichette assi
FS_TICK    = 18   # Numeri sugli assi
FS_LEGEND  = 18   # Testo legenda

# Marker e linee
MS         = 7    # Dimensione marker (markersize)
LW_FIT     = 2.5  # Spessore curva di fit
LW_REF     = 1.2  # Spessore linee di riferimento (axhline/axvline)
CAPSIZE    = 3    # Dimensione cappucci barre di errore


def plot_and_save(data, config, m_amp, m_phase, ft_exp, eft_exp, output_dir="."):
    """
    Crea 3 figure e le salva come PNG:
    1. Risposta in ampiezza e fase + Residui
    2. Scansione χ² — Ampiezza
    3. Scansione χ² — Fase

    Notazione adottata (coerente in tutti i grafici):
      A            -- ampiezza normalizzata  V_out/V_in
      phi          -- sfasamento normalizzato (x90 gradi)
      f            -- frequenza (kHz)
      f_t          -- frequenza di taglio (kHz)
      chi2         -- chi quadro
    """
    os.makedirs(output_dir, exist_ok=True)

    f = data['f']
    A = data['A']
    eA = data['eA']
    phi_norm = data['phi_norm']
    ephi_norm = data['ephi_norm']

    # Ordina per frequenza crescente per evitare rette spurie nel fit
    sort_idx = np.argsort(f)
    f_s   = f[sort_idx]
    A_s   = A[sort_idx]
    eA_s  = eA[sort_idx]
    phi_s = phi_norm[sort_idx]
    ephi_s = ephi_norm[sort_idx]

    # Curve di fit su griglia densa
    f_fit = np.logspace(np.log10(f_s.min()), np.log10(f_s.max()), 500)
    A_fit_curve   = A_theory(f_fit,   m_amp['ft'])
    phi_fit_curve = phi_theory(f_fit, m_phase['ft'])

    # Residui sui punti misurati
    A_th_pts   = A_theory(f_s,   m_amp['ft'])
    phi_th_pts = phi_theory(f_s, m_phase['ft'])
    res_A   = A_s   - A_th_pts
    res_phi = phi_s - phi_th_pts

    # Colori coerenti con le due grandezze
    COLOR_AMP   = 'black'
    COLOR_PHASE = 'red'

    # Helper per impostare font dei tick uniformemente
    def _set_ticks(ax, labelcolor='black'):
        ax.tick_params(axis='both', labelsize=FS_TICK, labelcolor=labelcolor)

    # ========== FIGURA 1: Risposta + Residui ==========
    fig1, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG1_W, FIG1_H),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True                          # asse x condiviso
    )

    # Pannello superiore: dati + fit
    ax1.set_xscale('log')
    ax1.errorbar(f_s, A_s, yerr=eA_s, fmt='o', color=COLOR_AMP,
                 markersize=MS, capsize=CAPSIZE, alpha=0.85,
                 label=r"$A$ (misura)")
    ax1.plot(f_fit, A_fit_curve, '-', color=COLOR_AMP, linewidth=LW_FIT,
             label=rf"Fit $A$:  $f_t = {m_amp['ft']:.4f}$ kHz")
    ax1.set_ylabel(r'Ampiezza  $A = V_{{\rm out}}/V_{{\rm in}}$',
                   fontsize=FS_LABEL)
    ax1.tick_params(axis='y', labelcolor=COLOR_AMP, labelsize=FS_TICK)
    ax1.grid(True, which='both', alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.errorbar(f_s, phi_s, yerr=ephi_s, fmt='s', color=COLOR_PHASE,
                  markersize=MS, capsize=CAPSIZE, alpha=0.85,
                  label=r"$\varphi$ (misura)")
    ax1b.plot(f_fit, phi_fit_curve, '--', color=COLOR_PHASE, linewidth=LW_FIT,
              label=rf"Fit $\varphi$:  $f_t = {m_phase['ft']:.4f}$ kHz")
    ax1b.set_ylabel(r'Sfasamento  $\varphi$  (×90°)',
                    fontsize=FS_LABEL, color=COLOR_PHASE)
    ax1b.tick_params(axis='y', labelcolor=COLOR_PHASE, labelsize=FS_TICK)

    lines1,  labels1  = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b,
               loc='center right', fontsize=FS_LEGEND)

    # Pannello inferiore: residui di A e φ sovrapposti (senza legenda)
    ax2.errorbar(f_s, res_A, yerr=eA_s, fmt='o', color=COLOR_AMP,
                 markersize=MS, capsize=CAPSIZE, alpha=0.85)
    ax2.axhline(0, color=COLOR_AMP, linestyle='--', linewidth=LW_REF)
    ax2.set_ylabel(r'Residui  $A$', fontsize=FS_LABEL)
    ax2.tick_params(axis='y', labelcolor=COLOR_AMP, labelsize=FS_TICK)
    ax2.tick_params(axis='x', labelsize=FS_TICK)

    ax2b = ax2.twinx()
    ax2b.errorbar(f_s, res_phi, yerr=ephi_s, fmt='s', color=COLOR_PHASE,
                  markersize=MS, capsize=CAPSIZE, alpha=0.85)
    ax2b.axhline(0, color=COLOR_PHASE, linestyle='--', linewidth=LW_REF)
    ax2b.set_ylabel(r'Residui  $\varphi$ (×90°)',
                    fontsize=FS_LABEL, color=COLOR_PHASE)
    ax2b.tick_params(axis='y', labelcolor=COLOR_PHASE, labelsize=FS_TICK)

    ax2.set_xlabel(r'Frequenza  $f$  (kHz)', fontsize=FS_LABEL)
    ax2.grid(True, which='both', alpha=0.3)

    fig1.suptitle(r'Filtro RC — Risposta in ampiezza e fase con residui',
                  fontsize=FS_TITLE, fontweight='bold')
    plt.subplots_adjust(hspace=0.05)       # pannelli ravvicinati
    plt.tight_layout()
    path1 = os.path.join(output_dir, "rcf_01_risposta.png")
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\u2713 Salvato: {path1}")

    # ========== FIGURE 2 e 3: Scansione χ² ==========
    def _chi2_plot(m, color, titolo, filename):
        fig, ax = plt.subplots(figsize=(FIG2_W, FIG2_H) if 'Ampiezza' in titolo
                               else (FIG3_W, FIG3_H))

        a, b, c = m['coeffs']
        ft_parab   = np.linspace(m['ft_scan'].min(), m['ft_scan'].max(), 500)
        chi2_parab = a * ft_parab**2 + b * ft_parab + c
        ax.plot(ft_parab, chi2_parab, '-', color=color, linewidth=LW_FIT,
                label=r'Parabola di fit')

        ax.axvline(m['ft'], color='gray', linestyle='-', linewidth=LW_FIT,
                   label=rf"$f_t = {m['ft']:.4f}$ kHz")
        ax.axvline(m['ft'] - m['eft'], color='gray', linestyle=':', linewidth=LW_REF)
        ax.axvline(m['ft'] + m['eft'], color='gray', linestyle=':', linewidth=LW_REF,
                   label=rf"Incertezza:  $\pm{m['eft']:.4f}$ kHz")

        ax.axhline(m['chi2_min'],     color=color, linestyle='--',
                   linewidth=LW_REF, alpha=0.6,
                   label=rf"$\chi^2_{{\rm min}} = {m['chi2_min']:.2f}$")
        ax.axhline(m['chi2_min'] + 1, color=color, linestyle=':',
                   linewidth=LW_REF, alpha=0.6,
                   label=rf"$\chi^2_{{\rm min}} + 1 = {m['chi2_min']+1:.2f}$")

        # Zoom sul minimo: Δχ² ≤ 5
        chi2_zoom = 5
        ax.set_ylim(m['chi2_min'] - 0.5, m['chi2_min'] + chi2_zoom)

        discriminant_z = b**2 - 4*a*(c - m['chi2_min'] - chi2_zoom)
        if discriminant_z >= 0:
            x_lo = (-b - np.sqrt(discriminant_z)) / (2*a)
            x_hi = (-b + np.sqrt(discriminant_z)) / (2*a)
            margin = (x_hi - x_lo) * 0.15
            ax.set_xlim(x_lo - margin, x_hi + margin)

        ax.set_xlabel(r'Frequenza di taglio  $f_t$  (kHz)',
                      fontsize=FS_LABEL)
        ax.set_ylabel(r'$\chi^2$', fontsize=FS_LABEL)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FS_LEGEND)
        ax.set_title(titolo, fontsize=FS_TITLE, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\u2713 Salvato: {path}")

    _chi2_plot(m_amp,   COLOR_AMP,
               r'Scansione $\chi^2$ — Ampiezza',
               "rcf_02_chi2_ampiezza.png")
    _chi2_plot(m_phase, COLOR_PHASE,
               r'Scansione $\chi^2$ — Fase',
               "rcf_03_chi2_fase.png")


def save_results_txt(m_amp, m_phase, ft_exp, eft_exp, config, output_dir="."):
    """Salva i risultati in un file TXT."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    txt_path = os.path.join(output_dir, "rcf_results.txt")
    
    with open(txt_path, 'w') as f:
        f.write("="*75 + "\n")
        f.write("RC FILTER - CHI-SQUARE MINIMIZATION ANALYSIS\n")
        f.write("="*75 + "\n\n")
        
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration:\n")
        f.write(f"  R_meas = {config.R_meas:.3e} Ω\n")
        f.write(f"  C_meas = {config.C_meas:.3e} F\n")
        f.write(f"  Scan range: ±{config.ft_margin*100:.0f}% around theory\n")
        f.write(f"  Number of scan points: {config.NSTEP}\n\n")
        
        f.write("="*75 + "\n")
        f.write("METHOD 1: χ² MINIMIZATION ON AMPLITUDE\n")
        f.write("="*75 + "\n")
        f.write(f"χ² at minimum: {m_amp['chi2_min']:.2f}\n")
        f.write(f"Degrees of freedom: {m_amp['dof']}\n")
        f.write(f"Reduced χ² (χ²/dof): {m_amp['chi2_min']/m_amp['dof']:.2f}\n\n")
        f.write(f"RESULT:\n")
        f.write(f"  f_t = {m_amp['ft']:.6f} ± {m_amp['eft']:.6f} kHz\n")
        f.write(f"  Relative error: {100*m_amp['eft']/m_amp['ft']:.2f}%\n")
        f.write(f"  Difference from theory: {100*abs(m_amp['ft']-ft_exp/1e3)/(ft_exp/1e3):.2f}%\n\n")
        
        f.write("="*75 + "\n")
        f.write("METHOD 2: χ² MINIMIZATION ON PHASE\n")
        f.write("="*75 + "\n")
        f.write(f"χ² at minimum: {m_phase['chi2_min']:.2f}\n")
        f.write(f"Degrees of freedom: {m_phase['dof']}\n")
        f.write(f"Reduced χ² (χ²/dof): {m_phase['chi2_min']/m_phase['dof']:.2f}\n\n")
        f.write(f"RESULT:\n")
        f.write(f"  f_t = {m_phase['ft']:.6f} ± {m_phase['eft']:.6f} kHz\n")
        f.write(f"  Relative error: {100*m_phase['eft']/m_phase['ft']:.2f}%\n")
        f.write(f"  Difference from theory: {100*abs(m_phase['ft']-ft_exp/1e3)/(ft_exp/1e3):.2f}%\n\n")
        
        f.write("="*75 + "\n")
        f.write("THEORETICAL VALUE (from R and C)\n")
        f.write("="*75 + "\n")
        f.write(f"f_t (theory) = {ft_exp/1e3:.6f} ± {eft_exp/1e3:.6f} kHz\n\n")
        
        f.write("="*75 + "\n")
        f.write("ERROR CALCULATION DETAILS\n")
        f.write("="*75 + "\n\n")
        f.write("AMPLITUDE ERROR (eA):\n")
        f.write("  Formula: eA = A * sqrt((eV/V_in)² + (eV/V_out)²)\n")
        f.write("  where eV = V_scale * 0.04 (oscilloscope uncertainty)\n")
        f.write("  This assumes uncertainty on both V_in and V_out is the same\n\n")
        
        f.write("PHASE ERROR (ephi_norm):\n")
        f.write("  eDT = 0.04 * DT_scale * sqrt(2)  [µs]\n")
        f.write("  ephi_deg = phi_deg * eDT / DT  [degrees]\n")
        f.write("  ephi_norm = ephi_deg / 90  [normalized]\n")
        f.write("  Factor sqrt(2) is conservative (phase + amplitude uncertainty)\n\n")
        
        f.write("CHI-SQUARE FITTING:\n")
        f.write("  χ²(f_t) = Σ((y_meas - y_theory) / ey)²\n")
        f.write("  Parabola: χ²(f_t) = a*f_t² + b*f_t + c\n")
        f.write("  Best fit: f_t = -b/(2a)\n")
        f.write("  Uncertainty: where χ² = χ²_min + 1\n")
    
    print(f"✓ Saved: {txt_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()
    
    print_header("RC FILTER - χ² MINIMIZATION ANALYSIS")
    print("Computing cutoff frequency via chi-square minimization\n")
    
    # === Parametri teorici ===
    print_header("THEORETICAL PARAMETERS")
    eR, eC = calc_errors(config)
    ft_exp, eft_exp = calc_theoretical_ft(config.R_meas, config.C_meas, eR, eC)
    
    print(f"R = ({config.R_meas:.3e} ± {eR:.3e}) Ω")
    print(f"C = ({config.C_meas:.3e} ± {eC:.3e}) F")
    print(f"f_t (theory) = ({ft_exp/1e3:.4f} ± {eft_exp/1e3:.4f}) kHz\n")
    
    # === Caricamento dati ===
    print_header("DATA LOADING")
    
    # Auto-detect file
    possible_paths = [
        "RCF_PB.txt",
        "./RCF_PB.txt",
        os.path.expanduser("~/RC_frequenza/RCF_PB.txt"),
        os.path.expanduser("~/SecondoAnno/RC_frequenza/RCF_PB.txt"),
    ]
    
    datafile = None
    for path in possible_paths:
        if os.path.exists(path):
            datafile = path
            break
    
    if datafile is None:
        print("⚠️  File not found in standard locations:")
        for p in possible_paths:
            print(f"    {p}")
        datafile = input("\n📝 Enter file path: ").strip()
        if not os.path.exists(datafile):
            print(f"❌ File not found: {datafile}")
            return
    
    print(f"✓ File: {datafile}")
    
    data = load_data(datafile, config)
    print(f"✓ Loaded {data['n']} points")
    print(f"  Frequency range: {data['f'].min():.3f} – {data['f'].max():.1f} kHz")
    print(f"  Amplitude range: {data['A'].min():.4f} – {data['A'].max():.4f}")
    print(f"  Phase range: {data['phi_norm'].min():.3f} – {data['phi_norm'].max():.3f} (×90°)\n")
    
    # === Metodi χ²: ampiezza e fase in parallelo ===
    print_header("χ² MINIMIZATION (parallel)")
    print(f"Scanning cutoff frequency: ±{config.ft_margin*100:.0f}% around theory")
    print(f"Number of scan points: {config.NSTEP}\n")

    with ProcessPoolExecutor(max_workers=2) as pool:
        fut_amp   = pool.submit(chi2_minimize_amplitude, data, config, ft_exp)
        fut_phase = pool.submit(chi2_minimize_phase,     data, config, ft_exp)
        m_amp   = fut_amp.result()
        m_phase = fut_phase.result()

    print(f"[Amplitude]  χ²_min={m_amp['chi2_min']:.2f}  dof={m_amp['dof']}  "
          f"χ²_red={m_amp['chi2_min']/m_amp['dof']:.2f}")
    print(f"  f_t = {m_amp['ft']:.6f} ± {m_amp['eft']:.6f} kHz  "
          f"({100*m_amp['eft']/m_amp['ft']:.2f}% rel.)\n")
    print(f"[Phase]      χ²_min={m_phase['chi2_min']:.2f}  dof={m_phase['dof']}  "
          f"χ²_red={m_phase['chi2_min']/m_phase['dof']:.2f}")
    print(f"  f_t = {m_phase['ft']:.6f} ± {m_phase['eft']:.6f} kHz  "
          f"({100*m_phase['eft']/m_phase['ft']:.2f}% rel.)\n")
    
    # === Confronto ===
    print_header("COMPARISON")
    print(f"{'Method':<35} {'f_t (kHz)':<22} {'Error (%)':<12}")
    print("-" * 70)
    print(f"{'1. χ² Amplitude':<35} {m_amp['ft']:>8.6f} ± {m_amp['eft']:>8.6f}   {100*m_amp['eft']/m_amp['ft']:>8.2f}%")
    print(f"{'2. χ² Phase':<35} {m_phase['ft']:>8.6f} ± {m_phase['eft']:>8.6f}   {100*m_phase['eft']/m_phase['ft']:>8.2f}%")
    print()
    print(f"{'THEORY (R-C):':<35} {ft_exp/1e3:>8.6f} ± {eft_exp/1e3:>8.6f}   {100*eft_exp/ft_exp:>8.2f}%\n")
    
    diff_amp = 100 * abs(m_amp['ft'] - ft_exp/1e3) / (ft_exp/1e3)
    diff_phase = 100 * abs(m_phase['ft'] - ft_exp/1e3) / (ft_exp/1e3)
    
    print(f"Difference from theory:")
    print(f"  Amplitude: {diff_amp:.2f}%")
    print(f"  Phase: {diff_phase:.2f}%\n")
    
    # === Visualizzazione grafici ===
    output_dir = "rcf_output"
    print_header("SAVING PLOTS")
    plot_and_save(data, config, m_amp, m_phase, ft_exp, eft_exp, output_dir)
    print(f"\n\u2713 All plots saved in: {output_dir}/")
    
    print_header("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()