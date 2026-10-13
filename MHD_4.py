# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:07:41 2026

@author: Jannik
"""
# Hier möchte ich das Magnetfeld mit betrachten
#==============================================================================
# Importe
#==============================================================================
import glob
import os
from scipy.io import FortranFile
import numpy as np
import matplotlib.pyplot as plt
import numerical_tools as nt
#import cont_opac_3 as cont_opac

# =============================================================================
# Konstanten (cgs)
# =============================================================================
g = 2.74E04    # Gravitationsbeschleunigung der Sonne
K_B = 1.38E-16 # Boltzmann-Konstante
m_H = 1.67E-24 # Masse eines Wasserstoffatomes
mu = 1.3       # Mittleres Molekulargewicht
lambda0 = 5000 # Ångström

PIXEL_ZU_KM = 8.0
ANZAHL_Z_PIXEL = 192
FORTRAN_SHAPE = (512, 512, 192) # [Z, Y, X]

# =============================================================================
# Fortran Daten laden
# =============================================================================
dateien = glob.glob("*.dat") # Alle Daten mit .dat Endung im aktuellen Ordner
alle_daten = {}

for datei in dateien:
    name = os.path.splitext(datei)[0]
    try:
        with FortranFile(datei, 'r') as f:
            daten_flach = f.read_reals(dtype=np.float32)
            alle_daten[name] = daten_flach.reshape(FORTRAN_SHAPE, order='F').transpose(2, 1, 0)
                        
    except Exception as e:
        print(f"Fehler bei Binärdatei {datei}: {e}")
        
T = alle_daten["temp_f90_binned_noz_nx512_ny512_nz192"]
P = alle_daten["pres_f90_binned_noz_nx512_ny512_nz192"]
tau = alle_daten["tau5_f90_binned_noz_nx512_ny512_nz192"]
Bx = alle_daten["magx_f90_binned_noz_nx512_ny512_nz192"]
By = alle_daten["magy_f90_binned_noz_nx512_ny512_nz192"]
Bz = alle_daten["magz_f90_binned_noz_nx512_ny512_nz192"]

B_sq = Bx**2 + By**2 + Bz**2

# =============================================================================
# Fortran Daten aufarbeiten
# =============================================================================
# Gitter in km und cm aufbauen
z_km = np.arange(ANZAHL_Z_PIXEL) * PIXEL_ZU_KM
z_cm = z_km * 1E5
z_max_cm = max(z_cm)
z_min_cm = min(z_cm)
dz = z_cm[1] - z_cm[0]
l = len(z_cm)

# Berechnen unsere Durchschnittsdaten erst QS, dann Sunspot := SP
# QS
# Mittelwert über die gesamten Z-Werte
T_QS = np.mean(T[:, 482:512, 0:30], axis=(1, 2))
tau_QS = np.mean(tau[:, 482:512, 0:30], axis=(1, 2))
#+++++ Der Druck +++++
P_log10_QS = np.log10(P[:, 482:512, 0:30])
P_log10_mean_QS = np.mean(P_log10_QS, axis=(1, 2))
P_QS = 10 ** P_log10_mean_QS

# SP
# Mittelwert über die gesamten Z-Werte
T_SP = np.mean(T[:, 241:271, 241:271], axis=(1, 2))
tau_SP = np.mean(tau[:, 241:271, 241:271], axis=(1, 2))
#+++++ Der Druck +++++
P_log10_SP = np.log10(P[:, 241:271, 241:271])
P_log10_mean_SP = np.mean(P_log10_SP, axis=(1, 2))
P_SP_data = 10 ** P_log10_mean_SP
#+++++ Magentfeld +++++
B_sq_SP = np.mean(B_sq[:, 241:271, 241:271], axis=(1, 2))
P_mag_profil = B_sq_SP / (8 * np.pi)

#=============================================================================
# Wollen einen Nullpunkt finden
#=============================================================================
tau_target = 1 # Tau = 1 sichtbare Oberfläche

# Tau-Achse und z-Achse miteinander verbinden (Arrays umdrehen wegen Monotonie)
z_alt_tau1_QS = np.interp(tau_target, tau_QS[::-1], z_km[::-1])
z_alt_tau1_SP = np.interp(tau_target, tau_SP[::-1], z_km[::-1])
z_alt_tau1_SP_I = np.interp(tau_target, tau_SP[::-1], z_km[::-1])

# Setzen nun unseren Nullpunkt bei dem gefundenem Wert
z_km_0 = z_km - z_alt_tau1_QS

# Referenz von der aktuellen z-Achse im Bezug auf Tau
z_tau1_QS = 0  # Per Definition 
z_tau1_SP = z_alt_tau1_SP - z_alt_tau1_QS  # Das ist genau minus die Wilson-Depression


# Berechnen unser neues cm Gitter, da der Nullpunkt verschoben wurde
z_cm = z_km_0 * 1E5
dz = z_cm[1] - z_cm[0]
l = len(z_cm)

# Index der physikalischen Oberfläche (z=0) bestimmen
idx_z0 = np.argmin(np.abs(z_km_0))
idx_SP_0 = np.argmin(np.abs(z_km_0 - z_tau1_SP))

# Unser Druck an der Oberfläche z=0 und tau = 1 als Referenzdruck
P_QS_Oberfläche = P_QS[idx_z0]



print("\n================== Unsere aktuellen Daten ==================")
print(f"Geometrischer Nullpunkt verschoben um:              {z_alt_tau1_QS:.1f} km")
print(f"Druck an der Oberfläche der QS bei z=0; (tau = 1):  {P_QS_Oberfläche:.2E}")

# =============================================================================
# LGS Lösen
# =============================================================================
C = - (mu * m_H * g) / K_B 
H = (K_B * T_QS[0] ) / (mu * m_H * g)
Quotient_H = z_max_cm / H

integrand_QS = C / T_QS

#P_QS_Oberfläche = P_QS[0]  # Referenz-Gasdruck der ungestörten Sonne bei z=0
#integral_boden_QS = np.trapezoid(integrand_QS[0:idx_z0+1], z_cm[0:idx_z0+1])
#P_0_QS = P_QS_Oberfläche * np.exp(-integral_boden_QS)
P_0_QS = P_QS[idx_SP_0]

#integral_deckel_QS = np.trapezoid(integrand_QS[idx_z0:], z_cm[idx_z0:])
#P_N_QS = P_QS_Oberfläche * np.exp(integral_deckel_QS)
P_N_QS = P_QS[-1]

#+++ Magentfeld +++
P_mag = P_mag_profil[idx_SP_0]

P_0_SP = P_0_QS - P_mag
P_N_SP = P_N_QS

print(f"Berechneter Gitterboden-Druck (QS):                 {P_0_QS:.2E}")
print(f"Magnetischer Druck (P_mag):                         {P_mag:.2E}")
print(f"Berechneter Gitterboden-Druck (SP):                 {P_0_SP:.2E}")
print("============================================================")

#+++ Numerisch lösen +++
# Die neue Länge
z_cm_cut = z_cm[idx_SP_0:]

# Quellterme aus den REALEN T(z)-Verläufen ableiten
#g_z_QS = C * nt.berechne_gradient_2nd_order((1 / T_QS), z_cm)
g_z_SP = C * nt.berechne_gradient_2nd_order((1 / T_SP), z_cm)
g_z_SP_cut = g_z_SP[idx_SP_0:]
l_SP = len(z_cm_cut)

integrand_SP = C / T_SP[idx_SP_0:]
integral = - nt.berechne_integration(integrand_SP[::-1], z_cm_cut[::-1])
P_SP_integrated_rev = P_N_QS * np.exp(integral)
P_SP_integrated = P_SP_integrated_rev[::-1]

# Matrix-DGL lösen für Quiet Sun
#A_QS, b_QS = nt.Stencil_Poisson_DC(l, dz, np.log(P_0_QS), np.log(P_N_QS), -g_z_QS)
#log_P_QS_model = nt.gaus(A_QS, b_QS)
#P_QS_model = np.exp(log_P_QS_model)

# Matrix-DGL lösen für Sunspot
A_SP, b_SP = nt.Stencil_Poisson_DC(l_SP, dz, np.log(P_0_SP), np.log(P_N_SP), -g_z_SP_cut)
log_P_SP_model = nt.gaus(A_SP, b_SP)
P_SP_model = np.exp(log_P_SP_model)


target_pressure = P_SP_data[idx_SP_0] 
idx_new_tau1 = np.argmin(np.abs(P_SP_integrated - target_pressure))

# 2. Jetzt ist z_tau1_SP_dynamic dein neuer, korrekter Punkt für das Modell!
z_tau1_SP_I = z_km_0[idx_SP_0 + idx_new_tau1]

# =============================================================================
# Plotten mit geeichter Achse
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ------------------------------------------------------------------------------
# LINKER PLOT: Druckvergleich (Optimiert im Stil des rechten Plots)
# ------------------------------------------------------------------------------
ax1.plot(z_km_0[idx_SP_0:], P_QS[idx_SP_0:], color="red", lw=2.5, label="Quiet Sun")
#ax1.plot(z_km_0, P_SP_data, color="blue", lw=2.0, alpha=0.3, linestyle="--", label="Sunspot (Daten komplett)")
ax1.plot(z_km_0[idx_SP_0:], P_SP_integrated, color="cyan", lw=2.5, label="SP Integriert")
ax1.plot(z_km_0[idx_SP_0:], P_SP_model, color="blue", lw=2.5, label="SP")
ax1.set_yscale("log")

# Markierungen für die photosphärischen Oberflächen (Tau = 1)
ax1.scatter(z_tau1_QS, np.interp(z_tau1_QS, z_km_0, P_QS), color="darkred", marker="X", s=60, zorder=5, label=r"$\tau=1$ QS (0 km)")
ax1.scatter(z_tau1_SP_I, P_SP_integrated[idx_new_tau1], color="darkturquoise", marker="X", s=60, zorder=5, label=fr"$\tau=1$ SP {z_tau1_SP_I:.1f} km)")
ax1.scatter(z_tau1_SP, P_SP_model[0], color="darkblue", marker="X", s=60, zorder=5, label=fr"$\tau=1$ SP {z_tau1_SP:.1f} km)")

ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
ax1.axvline(x=z_tau1_SP, color="gray", linestyle=":", alpha=0.4)
ax1.axvline(x=z_tau1_SP_I, color="gray", linestyle=":", alpha=0.4)
ax1.set_title("Druckprofile QS und SP", fontsize=18, fontweight="bold")
ax1.set_xlabel("$z$ (km)", fontsize=18)
ax1.set_ylabel("Durchschnittsdruck (dyn/cm²)", fontsize=18)
ax1.grid(True, which="both", linestyle=":", alpha=0.6)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=14, loc="best")

# ------------------------------------------------------------------------------
# RECHTER PLOT: Temperaturprofile auf geeichter Achse
# ------------------------------------------------------------------------------
ax2.plot(z_km_0, T_QS, color="red", lw=2, linestyle="-", label="Quiet Sun")
ax2.plot(z_km_0, T_SP, color="blue", lw=2.5, label="Sunspot (Umbra)")

# Tau = 1 Punkte auf den Temperaturkurven einzeichnen
ax2.scatter(z_tau1_QS, np.interp(z_tau1_QS, z_km_0, T_QS), color="darkred", marker="X", s=50, zorder=5)
ax2.scatter(z_tau1_SP, np.interp(z_tau1_SP, z_km_0, T_SP), color="darkblue", marker="X", s=50, zorder=5)

ax2.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
ax2.axvline(x=z_tau1_SP, color="gray", linestyle=":", alpha=0.4)

ax2.set_title("Temperaturprofile QS und SP", fontsize=18, fontweight="bold")
ax2.set_xlabel("$z$ (km)", fontsize=18)
ax2.set_ylabel("Durchschnittstemperatur", fontsize=18)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.legend(fontsize=14, loc="best")

# ------------------------------------------------------------------------------
# Gesamt-Layout
# ------------------------------------------------------------------------------
fig.suptitle(f"Druck- und Temperaturvergleich mit Wilson-Depression: {abs(z_tau1_SP):.1f} km)", fontsize=14, fontweight="bold", y=0.98)

plt.tight_layout()



# =============================================================================
# 2D Plot eines horizontalen Schnitts (X-Y Ebene) für eine feste Höhe z
# =============================================================================
# Import für die automatische Skalierung der Colorbars
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 1. Wähle den Index für die gewünschte Höhe
z_index = idx_z0 
aktuelle_hoehe_km = z_km_0[z_index]

# 2. Extrahiere den horizontalen Schnitt (X-Y Ebene)
T_xy = T[z_index, :, :]
B_sq_xy = B_sq[z_index, :, :] # Das quadratische Magnetfeld auf derselben Höhe


# Achsen für X und Y in km anlegen (beide 512 Pixel lang)
x_km = np.arange(512) * PIXEL_ZU_KM
y_km = np.arange(512) * PIXEL_ZU_KM

# Neues Figure mit zwei Subplots nebeneinander erstellen
fig_xy, (ax_T, ax_B) = plt.subplots(1, 2, figsize=(14, 6))

# ------------------------------------------------------------------------------
# LINKER PLOT: Temperatur
# ------------------------------------------------------------------------------
im_T = ax_T.imshow(T_xy, 
                   origin='lower', 
                   cmap='hot', 
                   extent=[x_km[0], x_km[-1], y_km[0], y_km[-1]])

#ax_T.set_title(f"Temperatur ($z = {aktuelle_hoehe_km:.1f}$ km)", fontsize=14, fontweight="bold")
ax_T.set_xlabel("$x$ (km)", fontsize=18)
ax_T.set_ylabel("$y$ (km)", fontsize=18)

# --- Colorbar exakt an Plot-Höhe anpassen ---
divider_T = make_axes_locatable(ax_T)
cax_T = divider_T.append_axes("right", size="5%", pad=0.1) # 5% Breite des Plots, 0.1 Abstand
cbar_T = fig_xy.colorbar(im_T, cax=cax_T)
cbar_T.set_label("Temperature (K)", fontsize=18)

# ------------------------------------------------------------------------------
# RECHTER PLOT: Magnetfeld
# ------------------------------------------------------------------------------
im_B = ax_B.imshow(B_sq_xy, 
                   origin='lower', 
                   cmap='viridis', 
                   extent=[x_km[0], x_km[-1], y_km[0], y_km[-1]])

#ax_B.set_title(f"Magnetfeld $B^2$ ($z = {aktuelle_hoehe_km:.1f}$ km)", fontsize=14, fontweight="bold")
ax_B.set_xlabel("$x$ (km)", fontsize=18)
ax_B.set_ylabel("$y$ (km)", fontsize=18)

# --- Colorbar exakt an Plot-Höhe anpassen ---
divider_B = make_axes_locatable(ax_B)
cax_B = divider_B.append_axes("right", size="5%", pad=0.1) # 5% Breite des Plots, 0.1 Abstand
cbar_B = fig_xy.colorbar(im_B, cax=cax_B)
cbar_B.set_label("Magnetic Pressure (dyn/cm²)", fontsize=18)

# ------------------------------------------------------------------------------
# Layout optimieren und als PDF speichern
# ------------------------------------------------------------------------------
plt.tight_layout()

# Als Vektorgrafik (PDF) speichern. 
# bbox_inches="tight" verhindert, dass Achsenbeschriftungen abgeschnitten werden.
plt.savefig("horizontaler_schnitt_z0.pdf", format="pdf", bbox_inches="tight")

# =============================================================================
# Plotten nur mit Quiet Sun und SP Integriert
# =============================================================================
fig, ax1 = plt.subplots(figsize=(8, 6))

# Nur Quiet Sun und SP Integriert plotten
ax1.plot(z_km_0[idx_SP_0:], P_QS[idx_SP_0:], color="red", lw=2.5, label="Quiet Sun")
ax1.plot(z_km_0[idx_SP_0:], P_SP_integrated, color="cyan", lw=2.5, label="SP Integriert")
ax1.set_yscale("log")

# Markierungen für die photosphärischen Oberflächen (Tau = 1)
ax1.scatter(z_tau1_QS, np.interp(z_tau1_QS, z_km_0, P_QS), color="darkred", marker="X", s=60, zorder=5, label=r"$\tau=1$ QS (0 km)")
ax1.scatter(z_tau1_SP_I, P_SP_integrated[idx_new_tau1], color="darkturquoise", marker="X", s=60, zorder=5, label=fr"$\tau=1$ SP {z_tau1_SP_I:.1f} km")

ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
ax1.axvline(x=z_tau1_SP_I, color="gray", linestyle=":", alpha=0.4)

ax1.set_title("Druckprofil: Quiet Sun vs. SP Integriert", fontsize=18, fontweight="bold")
ax1.set_xlabel("$z$ (km)", fontsize=18)
ax1.set_ylabel("Durchschnittsdruck (dyn/cm²)", fontsize=18)
ax1.grid(True, which="both", linestyle=":", alpha=0.6)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=14, loc="best")

plt.tight_layout()


# =============================================================================
# Plotten nur mit Quiet Sun und SP Modell (dunkelblau)
# =============================================================================
fig, ax1 = plt.subplots(figsize=(8, 6))

# Nur Quiet Sun und SP Modell plotten
ax1.plot(z_km_0[idx_SP_0:], P_QS[idx_SP_0:], color="red", lw=2.5, label="Quiet Sun")
ax1.plot(z_km_0[idx_SP_0:], P_SP_model, color="blue", lw=2.5, label="Sunspot (Modell)")
ax1.set_yscale("log")

# Markierungen für die photosphärischen Oberflächen (Tau = 1)
ax1.scatter(z_tau1_QS, np.interp(z_tau1_QS, z_km_0, P_QS), color="darkred", marker="X", s=60, zorder=5, label=r"$\tau=1$ QS (0 km)")
ax1.scatter(z_tau1_SP, P_SP_model[0], color="darkblue", marker="X", s=60, zorder=5, label=fr"$\tau=1$ SP {z_tau1_SP:.1f} km")

ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
ax1.axvline(x=z_tau1_SP, color="gray", linestyle=":", alpha=0.4)

ax1.set_title("Druckprofil: Quiet Sun vs. Sunspot Modell", fontsize=18, fontweight="bold")
ax1.set_xlabel("$z$ (km)", fontsize=18)
ax1.set_ylabel("Durchschnittsdruck (dyn/cm²)", fontsize=18)
ax1.grid(True, which="both", linestyle=":", alpha=0.6)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=14, loc="best")

plt.tight_layout()
plt.show()
plt.show()

plt.show()