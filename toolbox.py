import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

def build_Dx_Dy(img_original):
    nb_lignes, nb_colonnes, nb_color = img_original.shape
    nb_pixels = nb_lignes * nb_colonnes 
    dx_diff = -np.ones(nb_pixels)
    dx_diag = np.ones(nb_pixels)

    # gestion des bords : On met donc la différence à 0 pour la dernière colonne de chaque ligne.
    dx_diff[nb_colonnes-1::nb_colonnes] = 0
    dx_diag[nb_colonnes-1::nb_colonnes] = 0
    
    # Création de la matrice Dx
    Dx = diags([dx_diag, dx_diff], [0, 1], shape=(nb_pixels, nb_pixels))

    dy_diff = -np.ones(nb_pixels)
    dy_diag = np.ones(nb_pixels)

    #gestionn des bords : on mets la dernière ligne à zéro
    dy_diag[-nb_colonnes:] = 0 

    # Création de la matrice Dy
    Dy = diags([dy_diag, dy_diff], [0, nb_colonnes], shape=(nb_pixels, nb_pixels))

    return Dx.astype(np.float32), Dy.astype(np.float32)

def comparer_gradients(img_original, Dx, Dy):
    """
    Calcule et compare les gradients via matrices d'opérateurs et différences finies.
    """
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32)/255.0

    nb_lignes, nb_colonnes = img_gray.shape
    img_gray_flat = img_gray.flatten()

    # 1. Calcul via matrices d'opérateurs (le produit matriciel @)
    grad_x_1D = Dx @ img_gray_flat
    grad_y_1D = Dy @ img_gray_flat
    
    grad_x_2D = grad_x_1D.reshape(nb_lignes, nb_colonnes)
    grad_y_2D = grad_y_1D.reshape(nb_lignes, nb_colonnes)

    # 2. Calcul par différences finies manuelles
    zero_col = np.zeros((nb_lignes, 1))
    img_gray_dx = np.hstack((img_gray[:, :-1] - img_gray[:, 1:], zero_col))

    zero_lig = np.zeros((1, nb_colonnes))
    img_gray_dy = np.vstack((img_gray[:-1, :] - img_gray[1:, :], zero_lig)) 

    # 3. Vérification de la précision (arrondi à 10^-6)
    verif_Dx = np.around(grad_x_2D.astype(np.double), 6) == np.around(img_gray_dx.astype(np.double), 6)
    verif_Dy = np.around(grad_y_2D.astype(np.double), 6) == np.around(img_gray_dy.astype(np.double), 6)

    # 4. Affichage
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten() 

    # Ligne X
    axes[0].imshow(grad_x_2D, cmap='gray')
    axes[0].set_title("Dx (Matrice)")
    axes[1].imshow(img_gray_dx, cmap='gray')
    axes[1].set_title("Dx (Diff. Finie)")
    axes[2].imshow(verif_Dx, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title("Comparaison Dx")

    # Ligne Y
    axes[3].imshow(grad_y_2D, cmap='gray')
    axes[3].set_title("Dy (Matrice)")
    axes[4].imshow(img_gray_dy, cmap='gray')
    axes[4].set_title("Dy (Diff. Finie)")
    axes[5].imshow(verif_Dy, cmap='viridis', vmin=0, vmax=1)
    axes[5].set_title("Comparaison Dy")

    plt.tight_layout()
    plt.show()
    
    return

def WLS_gray(img_original, Dx, Dy, alpha, eps, lamb):
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_gray_flat = img_gray.flatten()
    log_lum_flat = (np.log(img_gray + eps)).flatten()

    nb_lignes, nb_colonnes = img_gray.shape
    nb_pixels = nb_lignes * nb_colonnes

    grad_x_1D = Dx @ log_lum_flat
    grad_y_1D = Dy @ log_lum_flat

    ax = 1 / (np.abs(grad_x_1D)**alpha + eps)
    ay = 1 / (np.abs(grad_y_1D)**alpha + eps)

    Ax = diags([ax], [0], shape=(nb_pixels, nb_pixels))
    Ay = diags([ay], [0], shape=(nb_pixels, nb_pixels))

    Lg = Dx.T @ Ax @ Dx + Dy.T @ Ay @ Dy

    A_sys = eye(nb_pixels) + lamb * Lg
    A_sys = A_sys.tocsr()

    img_lisse_flat = spsolve(A_sys, img_gray_flat)
    img_lisse = img_lisse_flat.reshape((nb_lignes, nb_colonnes))

    return img_lisse

def WLS_RGB(img_original, Dx, Dy, alpha, eps, lamb):
    nb_lignes, nb_colonnes, nb_color = img_original.shape
    nb_pixels = nb_lignes * nb_colonnes

    luminance = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    log_lum_flat = (np.log(luminance + eps)).flatten()
    
    grad_x_1D = Dx @ log_lum_flat
    grad_y_1D = Dy @ log_lum_flat

    ax = 1 / (np.abs(grad_x_1D)**alpha + eps)
    ay = 1 / (np.abs(grad_y_1D)**alpha + eps)

    Ax = diags(ax)
    Ay = diags(ay)

    Lg = Dx.T @ Ax @ Dx + Dy.T @ Ay @ Dy

    A_sys = (eye(nb_pixels) + lamb * Lg).tocsr()

    img_lisse = np.zeros_like(img_original, dtype=float)

    for i in range(nb_color):
        img_flat = img_original[:,:,i].flatten()
        img_lisse_flat = spsolve(A_sys, img_flat)
        img_lisse[:,:,i] = img_lisse_flat.reshape((nb_lignes, nb_colonnes))

    return img_lisse


