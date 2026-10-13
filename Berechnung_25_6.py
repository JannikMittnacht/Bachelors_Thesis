# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:47:05 2026

@author: Jannik
"""
import numpy as np
import numerical_tools as nt
import cont_opac_new_new as cont_opac
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

#==============================================================================
# Konstanten (cgs)
#==============================================================================
g = 2.74E04    # Gravitationsbeschleunigung der Sonne
K_B = 1.38E-16 # Boltzmann-Konstante
m_H = 1.67E-24 # Masse eines Wasserstoffatomes
mu = 1.3       # Mittleres Molekulargewicht
lambda0 = 5000 # Ångström

#==============================================================================
# Festsetzungen & Gitter
#==============================================================================
z_min_km = -500 
z_max_km = 500
Gitterpunkte = 1000

z_km = np.linspace(z_min_km, z_max_km, Gitterpunkte)
z_cm = z_km * 1E5
dz = z_cm[1] - z_cm[0]  
l = len(z_cm)
z_min_cm = z_cm[0]
z_max_cm = z_cm[-1]
#Abstand_z_min_zu_0 = abs(z_min_cm)

# Physikalische Randparameter an der sichtbaren Oberfläche
P_QS_Oberfläche = 1.5E05  # Referenz-Gasdruck der ungestörten Sonne bei z=0
#wilson_depression_cm = abs(z_min_cm) # Geometrischer Abstand vom Gitterboden bis z=0

#==============================================================================
# Temperaturprofile (Jetzt beide dynamisch & geschichtet)
#==============================================================================
# Quiet Sun Temperatur
T_unten_QS = 7000  
T_oben_QS = 4000   
D_1_QS = z_max_cm * ((T_oben_QS) / (T_unten_QS - T_oben_QS))
D_0_QS = T_unten_QS * D_1_QS
T_z_QS = D_0_QS / (z_cm + D_1_QS)

# 1. Umbra-Temperatur (In der tiefen Schicht nähert sie sich der QS an!)
T_unten_U = 4000  
T_oben_U = 3000   
D_1_U = z_max_cm * ((T_oben_U) / (T_unten_U - T_oben_U))
D_0_U = T_unten_U * D_1_U
T_z_U = D_0_U / (z_cm + D_1_U)

#==============================================================================
# Lösen DGL
#==============================================================================
C = - (mu * m_H * g) / K_B 
H = (K_B * T_z_QS[0] ) / (mu * m_H * g)
Quotient_H = z_max_cm / H

B = 3000
P_mag = (B**2) / (8 * np.pi)

integrand = C / T_z_QS

#Skalenhöhe berechnen für z=0
idx_z0 = np.argmin(np.abs(z_km))

integral_boden = np.trapezoid(integrand[0:idx_z0+1], z_cm[0:idx_z0+1])
P_0_QS = P_QS_Oberfläche * np.exp(-integral_boden)

integral_deckel = np.trapezoid(integrand[idx_z0:], z_cm[idx_z0:])
P_N_QS = P_QS_Oberfläche * np.exp(integral_deckel)

P_0_U = P_0_QS - P_mag
P_N_U = P_N_QS

# Dirichlet-Randbedingungen (DC) für die Quiet Sun (QS)
#P_0_QS = P_QS_Oberfläche * np.exp(Abstand_z_min_zu_0 / H_QS_z0) #* np.exp(Quotient_H)#1.5E05 #P_QS_oberfläche * np.exp(wilson_depression_cm / H_QS_boden)
#P_N_QS = 1.3E03  # Oberer chromosphärischer Randdruck

# Dirichlet-Randbedingungen (DC) für die Umbra (UMBRA)
#P_0_U = (P_QS_Oberfläche * np.exp(Abstand_z_min_zu_0 / H_U_z0)) - P_mag #(P_0_QS * np.exp(Quotient_H)) - P_mag  # Horizontales Gleichgewicht am Boden
#P_N_U = 1.3E03  

# Quellterme aus den analytischen T(z)-Verläufen ableiten
g_z_QS = C * nt.berechne_gradient_2nd_order((1 / T_z_QS), z_cm)
g_z_U  = C * nt.berechne_gradient_2nd_order((1 / T_z_U), z_cm)

# Lösen numerisch
# Quiet Sun
A_QS, b_QS = nt.Stencil_Poisson_DC(l, dz, np.log(P_0_QS), np.log(P_N_QS), g_z_QS)
log_P_QS = nt.gaus(A_QS, b_QS)
#log_P_QS = np.linalg.solve(A_QS, b_QS)
P_QS = np.exp(log_P_QS)

# Umbra
A_U, b_U = nt.Stencil_Poisson_DC(l, dz, np.log(P_0_U), np.log(P_N_U), g_z_U)
log_P_U = nt.gaus(A_U, b_U)
#log_P_U = np.linalg.solve(A_U, b_U)
P_U = np.exp(log_P_U)

# =============================================================================
# AUTOMATISCHE BERECHNUNG DER WILSON-DEPRESSION
# =============================================================================
print("\n=== MATHEMATISCHER VALUE-CHECK ===")
print(f"Zielwert P_gas_QS(z=0):               {P_QS_Oberfläche:.2e} dyn/cm²")
print(f"P_total_umbra am Boden  (z = -500km): {P_U[0]:.2e} dyn/cm²")
print(f"P_total_umbra Oben      (z =  500km): {P_U[-1]:.2e} dyn/cm²")

# Wir suchen die Höhe 'z' in der Umbra, auf der P_total_umbra genau P_target erreicht.
# Da die Drücke mit der Höhe streng monoton fallen, müssen wir die Arrays für np.interp umdrehen.
z_surface_umbra = np.interp(P_QS_Oberfläche , P_U[::-1], z_km[::-1])

# Die Wilson-Depression ist der Abstand von z=0 (QS-Oberfläche) bis zu dieser Schicht
wilson_depression_berechnet = z_surface_umbra

print("\n--- ERGEBNIS DER BERECHNUNG ---")
print(f"Die sichtbare Umbra-Oberfläche liegt bei z = {z_surface_umbra:.1f} km")
print(f"Berechnete Wilson-Depression: {wilson_depression_berechnet:.1f} km\n")

# =============================================================================
# Berechnen die Trübung nutzen dafür Cont Opac von Andreu
# =============================================================================
# Bauen unsere leeren arrays
opac_QS = np.zeros(l)

opac_U = np.zeros(l)


#++++++++++++++++++++++++++++++++++
# QS and Sunspot
for c in range(0, l): # get opacity
    opac_values_QS = cont_opac.get_continuum_opacity(T_z_QS[c], P_QS[c], lambda0)
    opac_QS[c] = opac_values_QS[0]
    
    opac_values_U = cont_opac.get_continuum_opacity(T_z_U[c], P_U[c], lambda0)
    opac_U[c] = opac_values_U[0]

#+++++++++++++++++++++++++++++++++
tau_oben = 10**-4
z_cm_rev = z_cm[::-1]

# Berechnen Tau QS
tau_QS = cumulative_trapezoid(opac_QS[::-1], z_cm_rev)
tau_profile_QS = np.concatenate(([tau_oben], tau_oben - tau_QS))[::-1]

# Berechnen Tau Sunspot
tau_U = cumulative_trapezoid(opac_U[::-1], z_cm_rev)
tau_profile_U = np.concatenate(([tau_oben], tau_oben - tau_U))[::-1]

#+++++++++++++++++++++++++++++
# Suchen Tau = 1
tau_target = 1.0

# Jetzt übergeben wir die perfekt sortierten Arrays an np.interp
z_tau1_QS = np.interp(tau_target, tau_profile_QS[::-1], z_km[::-1])
z_tau1_U  = np.interp(tau_target, tau_profile_U[::-1], z_km[::-1])

wilson_depression_tau = z_tau1_QS - z_tau1_U

print("=== OPTISCHER VALUE-CHECK (Tau = 1) ===")
print(f"Sichtbare QS-Oberfläche (tau=1) bei z:      {z_tau1_QS:.2f} km")
print(f"Sichtbare Umbra-Oberfläche (tau=1) bei z:   {z_tau1_U:.2f} km")
print(f"Wilson-Depression (via tau=1 bestimmt):     {wilson_depression_tau:.1f} km\n")

#==============================================================================
# Plotten im direkten Vergleich
#==============================================================================
# =============================================================================
# plt.figure(figsize=(14, 6))
#  
# # Linker Plot: Druckvergleich (Log-Skala)
# plt.subplot(1, 2, 1)
# plt.plot(z_km, P_QS, 'r-', label="Quiet Sun (QS)", linewidth=2)
# plt.plot(z_km, P_U, 'b--', label="Umbra (Fleck)", linewidth=2)
# # Indikator-Linien für die Wilson-Depression einzeichnen
# plt.scatter(0, P_QS_Oberfläche, color='salmon', s=12, )#label=f"{P_QS_Oberfläche}")
# plt.scatter(x=z_surface_umbra, y=P_QS_Oberfläche, color='blue', s=12, )
# plt.scatter(z_max_km, P_N_QS , color='salmon', s=12, )#label=f"{P_N_QS}")
# plt.scatter(z_min_km, P_0_U, color='blue', s=12, )#label=f"{P_0_U}")
# plt.scatter(z_max_km, P_N_U , color='blue', s=12, )#label=f"{P_N_U}")
# plt.axvline(x=0, color='red', linestyle=':', alpha=0.7, label='Sichtbare QS-Oberfläche ($z=0$ km)')
# plt.axvline(x=z_surface_umbra, color='blue', linestyle=':', alpha=0.7, label=f'Sichtbare Umbra-Oberfläche ({z_surface_umbra:.1f} km)')
# plt.axhline(y=P_QS_Oberfläche, color='gray', linestyle=':', alpha=0.7)
# plt.xlabel(r'Höhe $z$ [km]', fontsize=12)
# plt.ylabel(r'Druck $P$ [dyn cm$^{-2}$]', fontsize=12)
# plt.yscale('log')
# plt.grid(True, which="both", ls="-", alpha=0.1)
# plt.title("Druckprofil: QS vs. UMBRA")
# plt.legend(fontsize=10, loc="lower left")
# 
# 
# 
# 
# # Rechter Plot: Temperaturvergleich
# plt.subplot(1, 2, 2)
# plt.plot(z_km, T_z_QS, 'r-', label="Quiet Sun (QS)", linewidth=2)
# plt.plot(z_km, T_z_U, 'b--', label="Umbra (Fleck)", linewidth=2)
# plt.xlabel(r'Höhe $z$ [km]', fontsize=12)
# plt.ylabel(r'Temperatur $T$ [K]', fontsize=12)
# plt.grid(True, which="both", alpha=0.2)
# plt.scatter(z_max_km - z_max_km, T_unten_QS, color='salmon', s=12, label=f"{T_unten_QS} K")
# plt.scatter(z_max_km, T_oben_QS, color='salmon', s=12, label=f"{T_oben_QS} K")
# plt.axhline(y=T_oben_QS, color='gray', linestyle=':', alpha=0.7)
# plt.scatter(z_max_km - z_max_km, T_unten_U, color='blue', s=12, label=f"{T_unten_U} K")
# plt.scatter(z_max_km, T_oben_U, color='blue', s=12, label=f"{T_oben_U} K")
# plt.title("Temperaturprofil: QS vs. UMBRA")
# plt.legend(fontsize=11)
# 
# plt.suptitle("...", fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()
# =============================================================================

plt.figure(figsize=(14, 6))
 
# Linker Plot: Druckvergleich (Log-Skala)
plt.subplot(1, 2, 1)
plt.plot(z_km, P_QS, 'r-', label="Quiet Sun (QS)", linewidth=2)
plt.plot(z_km, P_U, 'b--', label="Sunspot (Umbra)", linewidth=2)

# Indikator-Linien für die Wilson-Depression (Druck-Basis) einzeichnen
plt.scatter(0, P_QS_Oberfläche, color='salmon', s=30, zorder=5)
plt.scatter(x=z_surface_umbra, y=P_QS_Oberfläche, color='blue', s=30, zorder=5)
plt.axvline(x=z_tau1_QS, color='red', linestyle=':', alpha=0.5, label=f'Visible QS-Surface ({z_tau1_QS:.1f} km)')
plt.axvline(x=0, color='salmon', linestyle=':', alpha=0.5, label='Visible QS-Surface ($P_{gas}(z=0)$)')
plt.axvline(x=z_surface_umbra, color='blue', linestyle=':', alpha=0.5, label=f'Visible Umbra-Surface ({z_surface_umbra:.1f} km)')
plt.axhline(y=P_QS_Oberfläche, color='gray', linestyle=':', alpha=0.5)

# NEU: Optische Oberfläche (Tau = 1) in den Druckplot einbauen
# Wir holen uns den Druck an der Stelle, wo tau=1 ist, für die optischen Punkte
P_tau1_QS = np.interp(z_tau1_QS, z_km, P_QS)
P_tau1_U  = np.interp(z_tau1_U, z_km, P_U)
plt.scatter(z_tau1_QS, P_tau1_QS, color='darkred', marker='X', s=50, zorder=6, label=r'$\tau=1$ (QS)')
plt.scatter(z_tau1_U, P_tau1_U, color='darkblue', marker='X', s=50, zorder=6, label=r'$\tau=1$ (Umbra)')

plt.xlabel(r'Height $z$ [km]', fontsize=18)
plt.ylabel(r'Pressure $P$ [dyn cm$^{-2}$]', fontsize=18)
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.1)
#plt.title("Pressure profile : QS vs. UMBRA")
plt.legend(fontsize=14, loc="lower left")
plt.tick_params(axis='both', which='major', labelsize=14)
 
# Rechter Plot: Temperaturvergleich
plt.subplot(1, 2, 2)
plt.plot(z_km, T_z_QS, 'r-', label="Quiet Sun (QS)", linewidth=2)
plt.plot(z_km, T_z_U, 'b--', label="Sunspot (Umbra)", linewidth=2)

# NEU: Vertikale Linien für die reale, optische Photosphäre (Tau = 1)
plt.axvline(x=z_tau1_QS, color='darkred', linestyle=':', alpha=0.7, label=fr'QS $ \tau = 1$ ({z_tau1_QS:.1f} km)')
plt.axvline(x=z_tau1_U, color='darkblue', linestyle=':', alpha=0.7, label=fr'Umbra $ \tau = 1$ ({z_tau1_U:.1f} km)')
plt.axhline(y=T_oben_QS, color='gray', linestyle=':', alpha=0.5)


# Optische Punkte auf den Temperaturkurven markieren
T_tau1_QS = np.interp(z_tau1_QS, z_km, T_z_QS)
T_tau1_U  = np.interp(z_tau1_U, z_km, T_z_U)
plt.scatter(z_tau1_QS, T_tau1_QS, color='darkred', marker='X', s=60, zorder=5)
plt.scatter(z_tau1_U, T_tau1_U, color='darkblue', marker='X', s=60, zorder=5)

plt.xlabel(r'Height $z$ [km]', fontsize=18)
plt.ylabel(r'Temperature $T$ [K]', fontsize=18)
plt.grid(True, which="both", alpha=0.2)
#plt.title("Temperature Profile: QS vs. UMBRA")
plt.legend(fontsize=18, loc="upper right")
plt.tick_params(axis='both', which='major', labelsize=14)

#plt.suptitle(f"Modeling the Wilson Depression (Visually: {wilson_depression_tau:.1f} km)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()