import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors


# global settings of the graph
#print(plt.style.available)
# selected the ROOT style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '10',
         'legend.loc': 'upper right',
          'legend.frameon':       'True',
          'legend.framealpha':    '0.8',      # legend patch transparency
          'legend.facecolor':     'w', # inherit from axes.facecolor; or color spec
          'legend.edgecolor':     'w',      # background patch boundary color
          'figure.figsize': (6, 4),
         'axes.labelsize': '10',
         'figure.titlesize' : '14',
         'axes.titlesize':'12',
         'xtick.labelsize':'10',
         'ytick.labelsize':'10',
         'lines.linewidth': '1',
         'text.usetex': False,
#         'axes.formatter.limits': '-5, -3',
         'axes.formatter.min_exponent': '2',
#         'axes.prop_cycle': cycler('color', 'bgrcmyk')
         'figure.subplot.left':'0.125',
         'figure.subplot.bottom':'0.125',
         'figure.subplot.right':'0.925',
         'figure.subplot.top':'0.925',
         'figure.subplot.wspace':'0.1',
         'figure.subplot.hspace':'0.1',
#         'figure.constrained_layout.use' : True
          }
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# Function definition
def fit_lin(x, m, q):
    fitval = m * x + q
    return fitval

def fit_nlin_A(x, ft):
    fitval = 1/np.sqrt(1+(x/ft)**2)
    return fitval

def fit_nlin_phi(x, ft):
    fitval = np.arctan(x/ft)*2/np.pi
    return fitval

# Input file name
file = 'RCF_PB'
inputname = file+'.txt'

# Initial parameter values
ft_init= 4392 # Hz 

# Assumed reading errors
y_errfix = 0.1
y_errper = 0.03


#### LOAD DATA
"""
Read data from the input file
We assume to have imposed a current on a resistor and measured the voltage drop in the resistor
on x-axis we have the current in mA and on the y-axis we have the voltage in mV
the slope is the resistance of the resistor
"""

# load from file
data = np.loadtxt(inputname).T
f = 1/(np.array(data[0])*1e-6) # Hz
Vin = np.array(data[1]) # V
Vout = np.array(data[2]) # V
A = Vout/Vin
V_fs = np.array(data[3]) # V
phi = 2*np.pi*f*np.array(data[4])*1e-6/(np.pi/2.) # norm to 1
phi_fs = 2*np.pi*f*np.array(data[5])*1e-6/(np.pi/2.)


# Number of points to fit
N = len(f)

# Calculate errors on y
Vin_errL = V_fs[0]/10*0.41
V_errL = V_fs/10*0.41
phi_errL = phi_fs/10*0.41*np.sqrt(2)
A_err = A*np.sqrt((V_errL/Vout)**2+(Vin_errL/Vin)**2 + 2*(0.03*0.41)**2)
A_err[(f>10) & (f<16600)] = A[(f>10) & (f<16600)]*np.sqrt((V_errL[(f>10) & (f<16600)]/Vout[(f>10) & (f<16600)])**2+(Vin_errL/Vin[(f>10) & (f<16600)])**2)
#A_err = []
#for i in range(len(V_errL)):
#    if V_errL[i] == Vin_errL :
#        temp = A[i]*np.sqrt((V_errL[i]/Vout[i])**2+(Vin_errL/Vin[i])**2)
#        A_err.append(temp)
#    else:
#        temp = A[i]*np.sqrt((V_errL[i]/Vout[i])**2+(Vin_errL/Vin[i])**2 + 2*(0.03*0.41)**2)
#        A_err.append(temp)
A_err = np.asarray(A_err)
print(Vin_errL,V_errL, A_err)

#---------------------------
# Fit lineare locale
#---------------------------

scan = 2200. # +/- intervallo linearita' di rispetto ft_init
f_min = ft_init-scan
f_max = ft_init+scan
print(f_min,f_max)

x = f[(f>f_min) & (f<f_max)]
y = A[(f>f_min) & (f<f_max)]
z = phi[(f>f_min) & (f<f_max)]
y_err_A = A_err[(f>f_min) & (f<f_max)]#  V_errL[(f>f_min) & (f<f_max)]*np.sqrt(2)
z_err_phi = phi_errL[(f>f_min) & (f<f_max)]

N = len(x)
#print(x,y,z)

popt_A, pcov_A = curve_fit(fit_lin, x, y, p0=[-0.1, 1], method='lm', sigma=y_err_A,absolute_sigma=True)
popt_phi, pcov_phi = curve_fit(fit_lin, x, z, p0=[0.1, 0], method='lm', sigma=z_err_phi,absolute_sigma=True)
print(pcov_A)

"""
POPT: Vettore con la stima dei parametri dal fit
PCOV: Matrice delle covarianze
"""


# build residuals data
residuA = y - fit_lin(x, *popt_A)
residuphi =  z - fit_lin(x, *popt_phi)

# variables error and chi2
perr_A = np.sqrt(np.diag(pcov_A))
chisq_A = np.sum((residuA/y_err_A)**2)
perr_phi = np.sqrt(np.diag(pcov_phi))
chisq_phi = np.sum((residuphi/z_err_phi)**2)
df = N - 2

# frequenza di taglio
ft_A = (1/np.sqrt(2)-popt_A[1])/popt_A[0]
ft_phi = (0.5-popt_phi[1])/popt_phi[0]
corr = -1*np.mean(x)/np.sqrt(np.var(x)+np.mean(x)**2)

err_ft_A = np.sqrt((ft_A/popt_A[0])**2*perr_A[0]**2+(1/popt_A[0])**2*perr_A[1]**2+2*ft_A/popt_A[0]**2*pcov_A[0,1])
err_ft_phi = np.sqrt((ft_phi/popt_phi[0])**2*perr_phi[0]**2+(1/popt_phi[0])**2*perr_phi[1]**2+2*ft_phi/popt_phi[0]**2*pcov_phi[0,1])

ft_stima = []
ft_stima.append(['A,loc', ft_A, err_ft_A])
ft_stima.append(['$\phi$,loc', ft_phi, err_ft_phi])


"""
# Extract and print best fit parameters and errors
"""

print("\n ============== BEST FIT Lin Loc - SciPy ====================")
print("\n ================== Per il modulo A ========================")
print( r' slope m = {a:.3e} +/- {b:.1e} s'.format(a=popt_A[0], b=perr_A[0]))
print( r' intercept q = {c:.3f} +/- {d:.3f} '.format(c=popt_A[1],d=perr_A[1]))
print(r' chisq/ndf = {e:.2f}'.format(e=chisq_A/df))
print(r' frequenza di taglio = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_A, d=err_ft_A))
print("\n ================== Per la fase phi ========================")
print( r' slope m = {a:.3e} +/- {b:.1e} s'.format(a=popt_phi[0], b=perr_phi[0]))
print( r' intercept q = {c:.3f} +/- {d:.3f} '.format(c=popt_phi[1],d=perr_phi[1]))
print(r' chisq/ndf = {e:.2f}'.format(e=chisq_phi/df))
print(r' frequenza di taglio = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_phi, d=err_ft_phi))
print("=============================================================\n")



# fit tracciato con mille punti fra la freq min e max
x_fit = np.linspace(f_min, f_max, 1000)

"""
# Plot data, fit and residuals
"""

fig, ax = plt.subplots(2, 1, figsize=(5, 4),sharex=True, constrained_layout = True, height_ratios=[2, 1])
ax[0].plot(x_fit, fit_lin(x_fit, *popt_A), label='modulo', linestyle='--', color='black')
ax[0].plot(x_fit,fit_lin(x_fit,*popt_phi), label='fase', linestyle='dashed', color='green')
ax[0].errorbar(f,A,yerr=A_err, fmt='o',ms=2,color='black') # , label=r'data'
ax[0].errorbar(f,phi,yerr=phi_errL, fmt='o',ms=2,color='green') #, label=r'data'
ax[0].set_xlim(f_min-scan/2,f_max+scan/2)
ax[0].set_ylim(0,1)
ax[0].legend(loc='upper right')
ax[0].set_ylabel(r'$\left|A\right|$ / $\phi_{norm \to 1}$')
#ax[0].set_xticks([2,3,4,5])
ax[0].text(1500,0.4,r'm = {a:.3e} $\pm$ {b:.1e} $s$'.format(a=popt_A[0], b=perr_A[0]), size=8, color='black')
ax[0].text(1500,0.3,r'q = {c:.3f} $\pm$ {d:.3f}'.format(c=popt_A[1],d=perr_A[1]), size=8, color='black')
ax[0].text(1500,0.2,r'm = {a:.3e} $\pm$ {b:.1e} $s$'.format(a=popt_phi[0], b=perr_phi[0]), size=8, color='green')
ax[0].text(1500,0.1,r'q = {c:.3f} $\pm$ {d:.3f}'.format(c=popt_phi[1],d=perr_phi[1]), size=8, color='green')

ax[0].text(1100,0.15,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_phi, d=err_ft_phi), size=10, color='green')
ax[0].text(1100,0.25,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_A, d=err_ft_A), size=10, color='black')

ax[1].errorbar(x,residuA,yerr=y_err_A, fmt='o', label=r'modulo',ms=2,color='black')
ax[1].errorbar(x,residuphi,yerr=z_err_phi, fmt='o', label=r'fase ',ms=2,color='green')
R_ylim = np.std(residuA)*5 + np.max(y_err_A)
ax[1].set_ylim([-1*R_ylim,R_ylim])
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'Frequenza (Hz)',loc='center')
ax[1].plot(x,np.zeros(N))

plt.savefig(file+'_loc.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 'figure')

plt.show()


#---------------------------
# Fit log BODE
#---------------------------

ft = ft_phi

scan = 5000. # +/- intervallo linearita' di rispetto ft_init
f_min = (ft+scan)
f_max = (ft+55*scan)
print(ft,f_min,f_max)

x = np.log10(f[(f>f_min) & (f<f_max)]/ft)
y = 20*np.log10(A[(f>f_min) & (f<f_max)])
y_err_logA = 20*1/A[(f>f_min) & (f<f_max)]*A_err[(f>f_min) & (f<f_max)]#(V_errL[(f>f_min) & (f<f_max)]/Vout[(f>f_min) & (f<f_max)]*A[(f>f_min) & (f<f_max)])
print(y,y_err_logA)

N = len(x)
#print(x,y,z)

popt_logA, pcov_logA = curve_fit(fit_lin, x, y, p0=[-20, 0], method='lm', sigma=y_err_logA,absolute_sigma=True)
custom_fit_lin = lambda var_x, q: fit_lin(var_x, -20, q)
popt_logA2, pcov_logA2 = curve_fit(custom_fit_lin, x, y, p0=[0], method='lm', sigma=y_err_logA,absolute_sigma=True)

"""
POPT: Vettore con la stima dei parametri dal fit
PCOV: Matrice delle covarianze
"""

# build residuals data
residulogA = y - fit_lin(x, *popt_logA)
residulogA2 = y - custom_fit_lin(x, *popt_logA2)

# variables error and chi2
perr_logA = np.sqrt(np.diag(pcov_logA))
chisq_logA = np.sum((residulogA/y_err_logA)**2)
perr_logA2 = np.sqrt(np.diag(pcov_logA2))
chisq_logA2 = np.sum((residulogA2/y_err_logA)**2)
df = N - 2

ft_bode1 = 10**(-1*popt_logA[1]/popt_logA[0])*ft
ft_bode2 = 10**(popt_logA2[0])*ft
err_ft_bode1 = np.sqrt((ft_bode1/popt_logA[0]*np.log(10))**2*perr_logA[0]**2+(ft_bode1*popt_logA[1]/popt_logA[0]**2*np.log(10))**2*perr_logA[1]**2-2*(ft_bode1/popt_logA[0]*np.log(10))*(ft_bode1*popt_logA[1]/popt_logA[0]**2*np.log(10))*pcov_logA[0,1])
err_ft_bode2 = ft_bode2*np.log(10)*perr_logA2[0]


ft_stima.append(['Bode', ft_bode1, err_ft_bode1])


"""
# Extract and print best fit parameters and errors
"""


print("\n ============== BEST FIT BODE - SciPy ====================")
print("\n ================== BODE 1 ========================")
print( r' slope m = {a:.3e} +/- {b:.1e}'.format(a=popt_logA[0], b=perr_logA[0]))
print( r' intercept q = {c:.3f} +/- {d:.3f} '.format(c=popt_logA[1],d=perr_logA[1]))
print(r' chisq/ndf = {e:.2f}'.format(e=chisq_logA/df))
print(r' frequenza di taglio BODE1= {e:.3f} +/- {d:.0f} Hz'.format(e=ft_bode1, d=err_ft_bode1))
print("\n ================== BODE 2 ========================")
print( r' intercept q = {c:.3f} +/- {d:.3f} '.format(c=popt_logA2[0],d=perr_logA2[0]))
print(r' chisq/ndf = {e:.2f}'.format(e=chisq_logA2/df))
print(r' frequenza di taglio BODE2= {e:.3f} +/- {d:.0f} Hz'.format(e=ft_bode2, d=err_ft_bode2))
print("=============================================================\n")

# fit tracciato con mille punti fra la freq min e max
x_fit = np.linspace(np.log10(f_min/ft), np.log10(f_max/ft), 1000)

"""
# Plot data, fit and residuals
"""

fig, ax = plt.subplots(2, 1, figsize=(5, 4),sharex=True, constrained_layout = True, height_ratios=[2, 1])
ax[0].plot(x_fit, fit_lin(x_fit, *popt_logA), label='BODE 1', linestyle='--', color='green')
ax[0].plot(x_fit, custom_fit_lin(x_fit, *popt_logA2), label='BODE 2', linestyle='--', color='blue')
ax[0].errorbar(np.log10(f/ft), 20*np.log10(A),yerr=20*1/A*A_err, fmt='o',ms=1,color='black') # , label=r'data'
#ax[0].set_xlim(np.log(f_min-scan/2/ft),np.log(f_max+scan/2/ft))
#ax[0].set_ylim(0,1)
ax[0].legend(loc='upper right')
ax[0].set_ylabel(r'$\left|A\right|~(dB)$')
#ax[0].set_xticks([2,3,4,5])
#ax[0].text(1.0,-10,r'm = {a:.3e} $\pm$ {b:.1e}'.format(a=popt_logA[0], b=perr_logA[0]), size=8, color='green')
#ax[0].text(1.0,-15,r'q = {c:.3f} $\pm$ {d:.3f}'.format(c=popt_logA[1],d=perr_logA[1]), size=8, color='green')

ax[0].text(-1,-30,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_bode1, d=err_ft_bode1), size=10, color='green')
ax[0].text(-1,-35,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_bode2, d=err_ft_bode2), size=10, color='blue')

ax[1].errorbar(x,residulogA,yerr=y_err_logA, fmt='o',ms=2,color='black')
ax[1].plot(x_fit, custom_fit_lin(x_fit, *popt_logA2)-fit_lin(x_fit, *popt_logA), linestyle='--', color='blue')
R_ylim = np.std(residulogA)*5 + np.max(y_err_logA)
ax[1].set_ylim([-1*R_ylim,R_ylim])
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'$\log{\frac{f}{f_t}}$')
ax[1].plot(x,np.zeros(N), color='green',linestyle='--')

plt.savefig(file+'_BODE.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 'figure')

plt.show()


#---------------------------
# Fit non lineare
#---------------------------

ft = ft_phi

x = f
y = A
z = phi
y_err_A = A_err

scan = 2200 # +/- intervallo preso con stessa sonda Vout e Vin
f_min = ft_init-scan
f_max = ft_init+scan
y_err_A[(f>f_min) & (f<f_max)] = A_err[(f>f_min) & (f<f_max)]
z_err_phi = phi_errL

N = len(x)

popt_nl_A, pcov_nl_A = curve_fit(fit_nlin_A, x, y, p0=[ft], method='lm', sigma=y_err_A,absolute_sigma=True)
popt_nl_phi, pcov_nl_phi = curve_fit(fit_nlin_phi, x, z, p0=[ft], method='lm', sigma=z_err_phi,absolute_sigma=True)

"""
POPT: Vettore con la stima dei parametri dal fit
PCOV: Matrice delle covarianze
"""


# build residuals data
residu_nl_A = y - fit_nlin_A(x, *popt_nl_A)
residu_nl_phi =  z - fit_nlin_phi(x, *popt_nl_phi)

# variables error and chi2
perr_nl_A = np.sqrt(np.diag(pcov_nl_A))
chisq_nl_A = np.sum((residu_nl_A/y_err_A)**2)
perr_nl_phi = np.sqrt(np.diag(pcov_nl_phi))
chisq_nl_phi = np.sum((residu_nl_phi/z_err_phi)**2)
df = N - 2

# frequenza di taglio
ft_nl_A = popt_nl_A[0]
ft_nl_phi = popt_nl_phi[0]
err_ft_A = perr_nl_A[0]
err_ft_phi = perr_nl_phi[0]


ft_stima.append(['A,nl', ft_nl_A, err_ft_A])
ft_stima.append(['$\phi$,nl', ft_nl_phi, err_ft_phi])


"""
# Extract and print best fit parameters and errors
"""


print("\n ============== BEST FIT Non Lineare - SciPy ====================")
print("\n ================== Modulo ========================")
print(r' chisq/ndf = {e:.2f}'.format(e=chisq_nl_A/df))
print(r' frequenza di taglio = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_nl_A, d=err_ft_A))
print("\n ================== fase ========================")
print(r' chisq/ndf = {e:.2f}'.format(e=chisq_nl_phi/df))
print(r' frequenza di taglio = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_nl_phi, d=err_ft_phi))
print("=============================================================\n")

# fit tracciato con mille punti fra la freq min e max
x_fit = np.linspace(np.min(x), np.max(x), 1000)

"""
# Plot data, fit and residuals
"""

fig, ax = plt.subplots(2, 1, figsize=(5, 4),sharex=True, constrained_layout = True, height_ratios=[2, 1])
ax[0].plot(x_fit, fit_nlin_A(x_fit, *popt_nl_A), label='modulo', linestyle='--', color='blue')
ax[0].plot(x_fit,fit_nlin_phi(x_fit,*popt_nl_phi), label='fase', linestyle='--', color='green')
ax[0].errorbar(f,A,yerr=A_err, fmt='o',ms=2,color='blue') # , label=r'data'
ax[0].errorbar(f,phi,yerr=phi_errL, fmt='o',ms=2,color='green') #, label=r'data'
ax[0].set_xlim(80,250000)
ax[0].set_ylim(0,1)
ax[0].set_xscale('log')
ax[0].legend(loc='upper right')
ax[0].set_ylabel(r'$\left|A\right|$ / $\phi_{norm \to 1}$')
#ax[0].set_xticks([2,3,4,5])

ax[0].text(10000,0.5,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_nl_phi, d=err_ft_phi), size=10, color='green')
ax[0].text(10000,0.6,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_nl_A, d=err_ft_A), size=10, color='blue')

ax[1].errorbar(x,residu_nl_A,yerr=y_err_A, fmt='o', label=r'modulo',ms=2,color='blue')
ax[1].errorbar(x,residu_nl_phi,yerr=z_err_phi, fmt='o', label=r'fase ',ms=2,color='green')
R_ylim = np.std(residuA)*5 + np.max(y_err_A)+0.05
ax[1].set_ylim([-1*R_ylim,R_ylim])
ax[1].set_xscale('log')
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'Frequenza (Hz)',loc='center')
ax[1].plot(x,np.zeros(N),color='black')

plt.savefig(file+'_NL.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 'figure')

plt.show()




#---------------------------
# Fit linearizzato
#---------------------------

ft = ft_phi


# tolgo dati ad alta frequenza, perche' con maggiore errore.
# definisco la scolita

soglia = 4.5e3
soglia2 = 4.5e3


x2 = f[(f<soglia)]**2
x = f[(f<soglia2)]
y = 1/A[(f<soglia)]**2
z = np.tan(phi[(f<soglia2)]*np.pi/2)

y_err_A2 = 2*y/A[(f<soglia)]*A_err[(f<soglia)]
z_err_phi = (1+z**2)*phi_errL[(f<soglia2)]

N = len(x)


custom_fit_lin1 = lambda var_x, m: fit_lin(var_x, m, 1)
custom_fit_lin2 = lambda var_x, m: fit_lin(var_x, m, 0)

popt_lin_A, pcov_lin_A = curve_fit(custom_fit_lin1, x2, y, p0=[1/ft**2], method='lm', sigma=y_err_A2,absolute_sigma=True)
popt_lin_phi, pcov_lin_phi = curve_fit(custom_fit_lin2, x, z, p0=[-1/ft], method='lm', sigma=z_err_phi,absolute_sigma=True)

"""
POPT: Vettore con la stima dei parametri dal fit
PCOV: Matrice delle covarianze
"""


# build residuals data
residu_lin_A = y - custom_fit_lin1(x2, *popt_lin_A)
residu_lin_phi =  z - custom_fit_lin2(x, *popt_lin_phi)

# variables error and chi2
perr_lin_A = np.sqrt(np.diag(pcov_lin_A))
chisq_lin_A = np.sum((residu_lin_A/y_err_A2)**2)
perr_lin_phi = np.sqrt(np.diag(pcov_lin_phi))
chisq_lin_phi = np.sum((residu_lin_phi/z_err_phi)**2)
df = N - 2

# frequenza di taglio
ft_lin_A = np.sqrt(1/popt_lin_A[0])
ft_lin_phi = 1/popt_lin_phi[0]
err_ft_A = 1/popt_lin_A[0]*np.sqrt(1/popt_lin_A[0])*perr_lin_A[0]
err_ft_phi = 1/popt_lin_phi[0]**2*perr_lin_phi[0]
    

ft_stima.append(['A,lin', ft_lin_A, err_ft_A])
ft_stima.append(['$\phi$,lin', ft_lin_phi, err_ft_phi])

"""
# Extract and print best fit parameters and errors
"""


print("\n ============== BEST FIT linearizzato - SciPy ====================")
print("\n ================== Modulo ========================")
print(r' chisq/ndf = {e:.2f}'.format(e=chisq_nl_A/df))
print( r' slope m = {a:.3e} +/- {b:.1e}'.format(a=popt_lin_A[0], b=perr_lin_A[0]))
print(r' frequenza di taglio = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_lin_A, d=err_ft_A))
print("\n ================== fase ========================")
print(r' chisq/ndf = {e:.2f}'.format(e=chisq_nl_phi/df))
print( r' slope m = {a:.3e} +/- {b:.1e}'.format(a=popt_lin_phi[0], b=perr_lin_phi[0]))
print(r' frequenza di taglio = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_lin_phi, d=err_ft_phi))
print("=============================================================\n")

# fit tracciato con mille punti fra la freq min e max
x_fit = np.linspace(np.min(x), np.max(x), 1000)
x_fit2 = np.linspace(np.min(x2), np.max(x2), 1000)

"""
# Plot data, fit and residuals
"""

fig, ax = plt.subplots(2, 2, figsize=(7, 4),sharex='col', constrained_layout = True, height_ratios=[2, 1])
ax[0,0].plot(x_fit2, custom_fit_lin1(x_fit2, *popt_lin_A), label='modulo', linestyle='--', color='blue')
ax[0,1].plot(x_fit,custom_fit_lin2(x_fit,*popt_lin_phi), label='fase', linestyle='--', color='green')
ax[0,0].errorbar(x2,y,yerr=y_err_A2, fmt='o',ms=2,color='blue') # , label=r'data'
ax[0,1].errorbar(x,z,yerr=z_err_phi, fmt='o',ms=2,color='green') #, label=r'data'
ax[0,0].set_xlim(5e3,1e8)
#ax[0,0].set_ylim(0,1)
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].legend(loc='upper right')
ax[0,0].set_ylabel(r'$\left|A\right|$ / $\phi_{norm \to 1}$')
#ax[0].set_xticks([2,3,4,5])

ax[0,1].text(500,20,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_lin_phi, d=err_ft_phi), size=10, color='green')
ax[0,0].text(30000,5,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_lin_A, d=err_ft_A), size=10, color='blue')

ax[1,0].errorbar(x2,residu_lin_A,yerr=y_err_A2, fmt='o', label=r'modulo',ms=2,color='blue')
ax[1,1].errorbar(x,residu_lin_phi,yerr=z_err_phi, fmt='o', label=r'fase ',ms=2,color='green')
R_ylim = np.std(residuA)*5 + np.max(y_err_A2)
ax[1,0].set_ylim([-1*R_ylim,R_ylim])
ax[1,0].set_xscale('log')
ax[1,0].set_ylabel(r'Residui')
ax[1,0].set_xlabel(r'Frequenza (Hz)',loc='center')
ax[1,0].plot(x,np.zeros(N),color='black')
ax[1,1].plot(x,np.zeros(N),color='black')

plt.savefig(file+'_lin.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 'figure')

plt.show()


"""
# Plot results
"""

ft_stima = [list(row) for row in zip(*ft_stima)]
ft_stima[1] = np.asarray(ft_stima[1])*1e-3
ft_stima[2] = np.asarray(ft_stima[2])*1e-3

fig, ax = plt.subplots(2, 1, figsize=(5, 4),sharex=True, constrained_layout = True, height_ratios=[1, 1])

ax[0].errorbar(ft_stima[0], ft_stima[1], yerr=ft_stima[2], fmt='o',ms=2)

res_media = np.average(ft_stima[1], weights = ft_stima[2])
err_res_media = 1/np.sqrt(1/ft_stima[2][0]**2+1/ft_stima[2][1]**2+1/ft_stima[2][2]**2+1/ft_stima[2][3]**2+1/ft_stima[2][4]**2+1/ft_stima[2][5]**2+1/ft_stima[2][6]**2)
ax[0].plot(ft_stima[0],[res_media]*len(ft_stima[0]),linestyle = '--', color = 'red')
res_ylim = np.std(ft_stima[1])*5
ax[0].set_ylim([res_media-1*res_ylim,res_media+res_ylim])
ax[0].text(ft_stima[0][0],res_media+0.025,r'$f_t$ = {e:.3f} +/- {d:.3f} kHz'.format(e=res_media, d=err_res_media), size=10, color='red')



C_ = 1/(2*np.pi*ft_stima[1]*1e3*3871.3)*1e9
err_C_ = C_*ft_stima[2]/ft_stima[1]
C__media = np.average(C_, weights = 1/err_C_**2)
err_C__media = 1/np.sqrt(1/err_C_[0]**2+1/err_C_[1]**2+1/err_C_[2]**2+1/err_C_[3]**2+1/err_C_[4]**2+1/err_C_[5]**2+1/err_C_[6]**2)

C__media_nl  =  (C_[3]/err_C_[3]**2+C_[4]/err_C_[4]**2)/(1/err_C_[3]**2+1/err_C_[4]**2) 
err_C__media_nl =  1/np.sqrt(1/err_C_[3]**2+1/err_C_[4]**2) # = np.sqrt(1/np.sum(1/err_C_[3:4]**2))
print(C__media_nl,err_C__media_nl)

ax[1].plot(ft_stima[0],[C__media]*len(ft_stima[0]),linestyle = '--', color = 'red')
ax[1].text(ft_stima[0][1],C__media-5,r'C = {e:.1f} +/- {d:.1f} nF'.format(e=C__media, d=err_C__media), size=10, color='red')

ax[1].errorbar(ft_stima[0], C_, yerr=err_C_, fmt='o',ms=2)
C__ylim = np.std(C_)*10
ax[1].set_ylim([C__media-1*C__ylim,C__media+C__ylim])


#ax[0].legend(loc='upper right')
ax[0].set_ylabel(r'Frequenza di taglio (kHz)')
#ax[0].set_xticks([2,3,4,5])
#ax[0].text(1500,0.4,r'm = {a:.3e} $\pm$ {b:.1e} $s$'.format(a=popt_A[0], b=perr_A[0]), size=8, color='black')
#ax[0].text(1500,0.3,r'q = {c:.3f} $\pm$ {d:.3f}'.format(c=popt_A[1],d=perr_A[1]), size=8, color='black')
#ax[0].text(1500,0.2,r'm = {a:.3e} $\pm$ {b:.1e} $s$'.format(a=popt_phi[0], b=perr_phi[0]), size=8, color='green')
#ax[0].text(1500,0.1,r'q = {c:.3f} $\pm$ {d:.3f}'.format(c=popt_phi[1],d=perr_phi[1]), size=8, color='green')

#ax[0].text(1100,0.15,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_phi, d=err_ft_phi), size=10, color='green')
#ax[0].text(1100,0.25,r'$f_t$ = {e:.0f} +/- {d:.0f} Hz'.format(e=ft_A, d=err_ft_A), size=10, color='black')

#ax[1].errorbar(x,residuA,yerr=y_err_A, fmt='o', label=r'modulo',ms=2,color='black')
#ax[1].errorbar(x,residuphi,yerr=z_err_phi, fmt='o', label=r'fase ',ms=2,color='green')
#R_ylim = np.std(residuA)*5 + np.max(y_err_A)
#ax[1].set_ylim([-1*R_ylim,R_ylim])
ax[1].set_ylabel(r'Capacitance (nF)')
#ax[1].set_xlabel(r'Frequenza (Hz)',loc='center')
#ax[1].plot(x,np.zeros(N))

plt.savefig(file+'_res.png',
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="w",
            edgecolor ='w',
            orientation ='Portrait',
            dpi = 'figure')

plt.show()

