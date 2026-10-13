# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:54:57 2026

@author: Jannik
"""
# Wollen hier alle unsere Numerischen Tools sammeln.
#==============================================================================
# Importe
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
#==============================================================================
# Die eigentlichen Funktionen
#==============================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gauß-Eliminations_Verfahren
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def gaus(A, b):
    # A die Matrix
    # b die rechte Seite
    ########
    n = len(b)
    # Erweiterte Matrix [A|b] erstellen
    M = np.column_stack((A.astype(float), b.astype(float)))
    ########
    for i in range(n):
        # 1. Pivotisierung: Suche das größte Element in der aktuellen Spalte
        max_row = np.argmax(np.abs(M[i:, i])) + i
        M[[i, max_row]] = M[[max_row, i]]
        
        # 2. Elimination
        for j in range(i + 1, n):
            faktor = M[j, i] / M[i, i]
            M[j, i:] -= faktor * M[i, i:]

    # 3. Rückwärtssubstitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
        
    return x
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Finite Differenzen
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def berechne_gradient_2nd_order(P, Gitter):
    n = len(Gitter)
    dz = Gitter[1] - Gitter[0]
    grad = np.zeros(n)
    
    # 1. Anfang (Forward 2. Ordnung)
    grad[0] = (-3*P[0] + 4*P[1] - P[2]) / (2*dz)
    
    # 2. Mitte (Zentral 2. Ordnung)
    for i in range(1, n-1):
        grad[i] = (P[i+1] - P[i-1]) / (2*dz)
        
    # 3. Ende (Backward 2. Ordnung)
    grad[-1] = (3*P[-1] - 4*P[-2] + P[-3]) / (2*dz)
    
    return grad

def berechne_gradient_2nd_order_variable(P, Gitter):
    n = len(Gitter)
    grad = np.zeros(n)
    h = np.diff(Gitter) 
    h_l = h[:-1]  # Abstände nach links (für i = 1 bis n-2)
    h_r = h[1:]   # Abstände nach rechts (für i = 1 bis n-2)
    
    Nenner = h_l * h_r * (h_l + h_r)
    grad[1:-1] = (h_l**2 * P[2:] + (h_r**2 - h_l**2) * P[1:-1] - h_r**2 * P[:-2]) / Nenner

    h1 = h[0]  
    h2 = h[1]  
    w0 = (-2 * h1 - h2) / (h1 * (h1 + h2))
    w1 = (h1 + h2) / (h1 * h2)
    w2 = -h1 / (h2 * (h1 + h2))
    grad[0] = w0 * P[0] + w1 * P[1] + w2 * P[2]

    hA = h[-1] 
    hB = h[-2] 
    w_last = (2 * hA + hB) / (hA * (hA + hB))
    w_prev1 = -(hA + hB) / (hA * hB)
    w_prev2 = hA / (hB * (hA + hB))
    grad[-1] = w_last * P[-1] + w_prev1 * P[-2] + w_prev2 * P[-3]

    return grad
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Trapezregel zum lösen des Integrales
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def berechne_integration(Werte_über_die_Integriert_wird, Gitter):
    # x sind die Stützstellen (das Gitter)
    # y sind die Werte die Integritert werden sollen
    ##########
    # Werte die wir benötigen
    x = Gitter
    y = Werte_über_die_Integriert_wird
    l_kum = len(Werte_über_die_Integriert_wird)
    tau = np.zeros(l_kum)
    ##########
    # Startwert
    tau[0] = 0 # Also komplette Durchlässigkeit
    ##########
    # Bauen unsere Schleife
    for i in range(0, l_kum-1):
        # Brauchen eine Schrittweite
        dx = np.abs(x[i+1] - x[i])
        # Berechnen die Fläche unter dem Trapez (der Kurve)
        Flaeche = 0.5 * (y[i] + y[i+1]) * dx
        # Nun aufsummieren
        tau[i+1] = tau[i] + Flaeche
    
    # Geben unseren gewünschten tau wert zurück
    return tau
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stencil mit P_N bekannt => Nutzen Forward Stencil. Erste Ableitung; Zweite Ordnung die passen nicht
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Passt nicht
def Stencil_forward_backward(Anzahl_Stützstellen, Abstand_der_Stützstellen, Analytische_Werte, Anfangswert_Druck):
    size = Anzahl_Stützstellen - 1
    A = np.zeros((size, size))
    b = np.zeros(size)
    
    A[0, :2] = [4, -1]
    b[0] = Analytische_Werte[0] * 2 * Abstand_der_Stützstellen + 3 * Anfangswert_Druck
    
    for i in range(1, size-3):
        A[i,i] = 4
        A[i, i+1] = -1
        A[i, i-1] = -3
        b[i]= Analytische_Werte[i] * 2 * Abstand_der_Stützstellen       
        
    #######
    # Letzter Forward
    A[-3, -3] = -3 
    A[-3, -2] = 4
    A[-3, -1] = -1
    b[-3] = Analytische_Werte[-3] * 2 * Abstand_der_Stützstellen
        
    # Erster Backward
    A[-2, -2] = 3
    A[-2, -3] = -4 
    A[-2, -4] = 1
    b[-2] = Analytische_Werte[-2] * 2 * Abstand_der_Stützstellen
    
    # Zweiter Backward   
    A[-1, -1] = 3
    A[-1, -2] = -4 
    A[-1, -3] = 1
    b[-1] = Analytische_Werte[-1] * 2 * Abstand_der_Stützstellen
       
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stencil mit P_N bekannt => Nutzen Backward Stencil. Erste Ableitung; Zweite Ordnung die passen nicht
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Passt nicht
def Stencil_backward_forward(Anzahl_Stützstellen, Abstand_der_Stützstellen, Analytische_Werte, Endwert_Druck):
    size = Anzahl_Stützstellen - 1
    A = np.zeros((size, size))
    b = np.zeros(size)
    
    A[0, :3] = [-3, 4, -1]
    b[0] = Analytische_Werte[0] * 2 * Abstand_der_Stützstellen
    
    A[1, 1] = -3
    A[1, 2] = 4
    A[1, 3] = -1
    b[1] = Analytische_Werte[1] * 2 * Abstand_der_Stützstellen

    for i in range(2, size-2):
        A[i,i-1] = -4
        A[i, i-2] = 1
        A[i, i] = 3
        b[i]= Analytische_Werte[i] * 2 * Abstand_der_Stützstellen
        
    A[-2, -2] = 3
    A[-2, -3] = -4
    A[-2, -4] = 1
    b[-2] = Analytische_Werte[-2] * 2 * Abstand_der_Stützstellen
         
    A[-1, -1] = -4
    A[-1, -2] = 1
    b[-1] = Analytische_Werte[-1] * 2 * Abstand_der_Stützstellen - 3 * Endwert_Druck
       
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stencil mit P_0 bekannt => Nutzen Zentralen Stencil, am Ende Backward Stencil. Erste Ableitung; Zweite Ordnung
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_central_backward(Anzahl_Stützstellen, Abstand_der_Stützstellen, Analytische_Werte, Anfangswert_Druck):
    # Wollen eine N-1 große Matrix, da der erste Punkt bekannt ist.
    size = Anzahl_Stützstellen - 1
    # Bauen unsere leere MAtrix / unser leeres b
    A = np.zeros((size, size))
    b = np.zeros(size)   
    #########
    # Unser P_0 ist bekannt, somit ist die Gleichung ugestellt gültig
    A[0, 1] = 1
    b[0] = Analytische_Werte[1] * 2 * Abstand_der_Stützstellen + Anfangswert_Druck
    #########
    # Für den zantralen Stencil
    for i in range(1, size-1):
        A[i, i-1] = -1  
        A[i, i+1] =  1  
        b[i] = Analytische_Werte[i+1] * 2 * Abstand_der_Stützstellen
    #########
    # Backward, da keine Ghost Cell "hinten"
    A[-1, -3:] = [1, -4, 3]
    b[-1] = Analytische_Werte[-1] * 2 * Abstand_der_Stützstellen
    
    # Geben unsere Werte zurück
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stencil mit P_N bekannt => Nutzen Zentralen Stencil, am Anfang Forward Stencil. Erste Ableitung; Zweite Ordnung
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_forward_central(Anzahl_Stützstellen, Abstand_der_Stützstellen, Analytische_Werte, Endwert_Druck):
    # Wollen eine N-1 große Matrix, da der letzte Punkt bekannt ist.
    size = Anzahl_Stützstellen - 1
    # Bauen unsere leere MAtrix / unser leeres b
    A = np.zeros((size, size))
    b = np.zeros(size)   
    #########
    # Forward, da keine Ghost Cell "vorne"
    A[0, :3] = [-3, 4, -1]
    b[0] = Analytische_Werte[0] * 2 * Abstand_der_Stützstellen
    #########
    # Für den zantralen Stencil
    for i in range(1, size-1):
        A[i, i-1] = -1  
        A[i, i+1] =  1  
        b[i] = Analytische_Werte[i] * 2 * Abstand_der_Stützstellen
    #########
    # Unser P_N ist bekannt, somit ist die Gleichung umgestellt gültig
    A[-1, -2] = -1
    b[-1] = Analytische_Werte[-2] * 2 * Abstand_der_Stützstellen - Endwert_Druck
    
    # Geben unsere Werte zurück
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stencil mit Ghost-Cell => Nutzen Zentralen Stencil. Erste Ableitung; Zweite Ordnung Neumann BC Ende
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_central_ghost_cells_begin(Anzahl_Stützstellen, Abstand_der_Stützstellen, Analytische_Werte, Anfangswert_Druck):
    # Wollen eine N große Matrix, da der erste Punkt P_0 bekannt ist und zwei Ghost-Cells.
    size = Anzahl_Stützstellen 
    # Bauen unsere leere MAtrix / unser leeres b
    A = np.zeros((size, size))
    b = np.zeros(size)   
    #########
    A[0, 1] = 2
    b[0] = 2 * Abstand_der_Stützstellen * Analytische_Werte[0] + 2 * Anfangswert_Druck
    
    # Überall Zentral
    for i in range(1, size - 1):
        # (P[i+1] - P[i-1]) / (2*dz) = P'[i]
        A[i, i-1] = -1
        A[i, i+1] =  1
        b[i] = Analytische_Werte[i] * 2 * Abstand_der_Stützstellen
        
    A[-1, -2:] = [-2, 2]
    b[-1] = 2 * Abstand_der_Stützstellen * Analytische_Werte[-1]
    
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stencil mit Ghost-Cell => Nutzen Zentralen Stencil. Erste Ableitung; Zweite Ordnung Neumann BC Anfang
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_central_ghost_cells_end(Anzahl_Stützstellen, Abstand_der_Stützstellen, Analytische_Werte, Endwert_Druck):
    # Wollen eine N große Matrix, da der letzte Punkt P_N bekannt ist und zwei Ghost-Cells.
    size = Anzahl_Stützstellen
    # Bauen unsere leere MAtrix / unser leeres b
    A = np.zeros((size, size))
    b = np.zeros(size)   
    #########
    A[0, :2] = [2, -2]
    b[0] = Analytische_Werte[0] * Abstand_der_Stützstellen
    
    # Überall Zentral
    for i in range(1, size - 1):
        A[i,i] = 1
        A[i, i-1] = -1.0
        A[i, i+1] =  1.0
        b[i] = Analytische_Werte[i] * 2 * Abstand_der_Stützstellen
        
    A[-1, -2] = -2
    b[-1] = 2 * Abstand_der_Stützstellen * Analytische_Werte[-1] - 2 * Endwert_Druck
    
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stencil zweite Ableitung zweite Ordnung DC Bedinungen, oben und unten, da zweiter Ordnung
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_zweite_Ableitung_DC(Anzahl_Stützstellen, Abstand_der_Stützstellen, Anfangswert_Druck, Endwert_Druck, alpha):
    size = Anzahl_Stützstellen
    A = np.zeros((size, size))
    b = np.zeros(size)
    k_minus = (1 - ((alpha * Abstand_der_Stützstellen) / 2))
    k_plus = (1 + ((alpha * Abstand_der_Stützstellen) / 2))
    
    
    A[0, 0] = 1
    b[0] = Anfangswert_Druck
    
    A[-1, -1] = 1
    b[-1] = Endwert_Druck
    
    for i in range(1, size-1):
        A[i, i-1] = k_minus
        A[i,i] = -2
        A[i, i+1] = k_plus
        b[i]= 0 
    
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stencil zweite Ableitung zweite Ordnung NBC Bedinungen, oben und unten, da zweiter Ordnung
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_zweite_Ableitung_NBC_DC(Anzahl_Stützstellen, Abstand_der_Stützstellen, alpha, Ableitung_Ende, Endwert_Druck):
    size = Anzahl_Stützstellen
    A = np.zeros((size, size))
    b = np.zeros(size)
    k_minus = (1 - ((alpha * Abstand_der_Stützstellen) / 2))
    k_plus = (1 + ((alpha * Abstand_der_Stützstellen) / 2))
    
    
    A[0, :2] = [-2, (k_minus + k_plus)] 
    b[0] = 2 * Abstand_der_Stützstellen * k_minus * Ableitung_Ende
    
    A[-1, -1] = 1
    b[-1] = Endwert_Druck
    
    for i in range(1, size-1):
        A[i, i-1] = k_minus
        A[i,i] = -2
        A[i, i+1] = k_plus
        b[i]= 0 
    
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stencil zweite Ableitung zweite Ordnung DC unten und NBC oben Bedinungen
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_zweite_Ableitung_DC_NBC(Anzahl_Stützstellen, Abstand_der_Stützstellen, alpha, Anfangswert_Druck, Ableitung_Ende):
    size = Anzahl_Stützstellen
    A = np.zeros((size, size))
    b = np.zeros(size)
    k_minus = (1 - ((alpha * Abstand_der_Stützstellen) / 2))
    k_plus = (1 + ((alpha * Abstand_der_Stützstellen) / 2))
    
    
    A[0, 0] = 1
    b[0] = Anfangswert_Druck
    
    A[-1, -2:] = [(k_minus + k_plus), -2] 
    b[-1] = -2 * Abstand_der_Stützstellen * k_plus * Ableitung_Ende
    
    for i in range(1, size-1):
        A[i, i-1] = k_minus
        A[i,i] = -2
        A[i, i+1] = k_plus
        b[i]= 0 
    
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Poissongleichungs Löser DC
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_Poisson_DC(Anzahl_Stützstellen, Abstand_der_Stützstellen, Anfangswert_Druck, Endwert_Druck, quellterm):
    size = Anzahl_Stützstellen
    dz = Abstand_der_Stützstellen
    A = np.zeros((size, size))
    b = np.zeros(size)
    
    A[0, 0] = 1
    b[0] = Anfangswert_Druck
    
    A[-1, -1] = 1
    b[-1] = Endwert_Druck
    
    for i in range(1, size-1):
        A[i, i-1] = 1 #* (dz**2)   
        A[i, i]   = -2 #* (dz**2)  
        A[i, i+1] = 1 #* (dz**2)   
        b[i] = quellterm[i] * (dz**2)  
        
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Poissongleichungs Löser NBC_DC
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_Poisson_NBC_DC(Anzahl_Stützstellen, Abstand_der_Stützstellen, Ableitung_Anfang, Endwert_Druck, quellterm):
    size = Anzahl_Stützstellen
    dz = Abstand_der_Stützstellen
    A = np.zeros((size, size))
    b = np.zeros(size)
    
    A[0, :2] = [-2, 2]
    b[0] = Abstand_der_Stützstellen * ( quellterm[0] * Abstand_der_Stützstellen + 2 * Ableitung_Anfang)
    
    A[-1, -1] = 1
    b[-1] = Endwert_Druck
    
    for i in range(1, size-1):
        A[i, i-1] = 1  
        A[i, i]   = -2   
        A[i, i+1] = 1    
        b[i] = quellterm[i] * (dz**2)  
        
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Poissongleichungs Löser DC_NBC
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_Poisson_DC_NBC(Anzahl_Stützstellen, Abstand_der_Stützstellen, Anfangswert_Druck, Ableitung_Ende, quellterm):
    size = Anzahl_Stützstellen
    dz = Abstand_der_Stützstellen
    A = np.zeros((size, size))
    b = np.zeros(size)
    
    A[0, 0] = 1
    b[0] = Anfangswert_Druck
    
    A[-1, -2:] = [2, -2]
    b[-1] = Abstand_der_Stützstellen * (quellterm[-1] * Abstand_der_Stützstellen - 2 * Ableitung_Ende)
    
    for i in range(1, size-1):
        A[i, i-1] = 1    
        A[i, i]   = -2   
        A[i, i+1] = 1  
        b[i] = quellterm[i] * (dz**2)  
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Poissongleichungs Löser DC mit variablem Abstand
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Stencil_Poisson_DC_variable(Anzahl_Stützstellen, Stützstellen_Gitter, Anfangswert_Druck, Endwert_Druck, quellterm):
    size = Anzahl_Stützstellen
    A = np.zeros((size, size))
    b = np.zeros(size)
    
    A[0, 0] = 1
    b[0] = Anfangswert_Druck
    
    A[-1, -1] = 1
    b[-1] = Endwert_Druck
    
    for i in range(1, size-1):
        dz_1 = Stützstellen_Gitter[i] - Stützstellen_Gitter[i-1]
        dz_2 = Stützstellen_Gitter[i+1] - Stützstellen_Gitter[i]
        
        A[i, i-1] = 2 / (dz_1 * (dz_1 + dz_2))
        A[i, i]   = -2 / (dz_1 * dz_2)
        A[i, i+1] = 2 / (dz_2 * (dz_1 + dz_2))
        b[i] = quellterm[i]
        
    return A, b
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Iteration für den Druck
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#def Iteration(Start_Druck, )






#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Validierung der Daten
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_physikalische_diagnose(Data_files, DATA_DIR, model_labels):
    # Wir erstellen ein 4x4 Gitter (4 Zeilen: P, T, B, rho)
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    
    for i, file_name in enumerate(Data_files):
        try:
            data = np.loadtxt(DATA_DIR / file_name, usecols=range(7))
            if data[0, 0] > data[-1, 0]: data = np.flipud(data)
            
            log_tau = data[:, 0] if np.any(data[:, 0] < 0) else np.log10(np.where(data[:, 0] > 0, data[:, 0], 1E-12))
            T, P_g, B, z, rho = data[:, 1], data[:, 3], data[:, 4], data[:, 5], data[:, 6]
            
            lbl = model_labels.get(file_name, file_name)
            
            # 1. Zeile: Gasdruck
            axs[0, i].plot(log_tau, P_g, color='black')
            axs[0, i].set_yscale('log')
            axs[0, i].set_title(f"P_g: {lbl}")
            
            # 2. Zeile: Temperatur
            axs[1, i].plot(log_tau, T, color='red')
            axs[1, i].set_title(f"T: {lbl}")
            
            # 3. Zeile: Magnetfeld
            axs[2, i].plot(log_tau, B, color='blue')
            axs[2, i].set_title(f"B: {lbl}")
            
            # 4. Zeile: Dichte
            axs[3, i].plot(log_tau, rho, color='green')
            axs[3, i].set_title(f"rho: {lbl}")
            axs[3, i].set_xlabel(r"$log\tau$")
            
            # Raster für alle
            for row in range(4):
                axs[row, i].grid(True, linestyle="--")
                
        except Exception as e:
            print(f"Fehler bei {file_name}: {e}")
            
    plt.tight_layout()
    plt.show()