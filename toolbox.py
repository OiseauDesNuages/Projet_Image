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
    verif_Dx = grad_x_2D.astype(np.double) == img_gray_dx.astype(np.double)
    verif_Dy = grad_y_2D.astype(np.double) == img_gray_dy.astype(np.double)

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

def Tone_Mapping_iter(image, list_lamb):
    # Méthode itérative
    n, m, c = image.shape
    Dx, Dy = build_Dx_Dy(image)
    Dx = Dx.tocsr()
    Dy = Dy.tocsr()

    list_img = np.zeros((n, m, c, len(list_lamb)))

    for i in range(len(list_lamb)):
        list_img[:,:,:,i] = WLS_RGB(image, Dx, Dy, alpha=1.2, eps=1e-4, lamb=list_lamb[i])
    
    u = image
    d1 = list_img[:,:,:,0]
    d2 = list_img[:,:,:,1]
    d3 = list_img[:,:,:,2]

    kc = 0.75
    km = 0.75
    kf = 0.75

    base = d3
    coarse = d3 + kc*(d2-d3) 
    medium = coarse + km*(d1-d2)
    fine = medium + kf*(u-d1)

    return base, coarse, medium, fine

def Tone_Mapping_rec(image):
    n, m, c = image.shape
    Dx, Dy = build_Dx_Dy(image)
    Dx = Dx.tocsr()
    Dy = Dy.tocsr()

    list_img = np.zeros((n, m, c, 3))

    list_img[:,:,:,0] = WLS_RGB(image, Dx, Dy, alpha=1.2, eps=1e-4, lamb=1)

    for i in range(1, 3):
        img_intermediaire = list_img[:,:,:,i-1].copy().astype(np.float32)
        list_img[:,:,:,i] = WLS_RGB(img_intermediaire, Dx, Dy, alpha=1.2, eps=1e-4, lamb=1)

    u = image
    d1 = list_img[:,:,:,0]
    d2 = list_img[:,:,:,1]
    d3 = list_img[:,:,:,2]

    kc = 0.75
    km = 0.75
    kf = 0.75

    base = d3
    coarse = d3 + kc*(d2-d3) 
    medium = coarse + km*(d1-d2)
    fine = medium + kf*(u-d1)

    return base, coarse, medium, fine

def dessiner_contours_noirs(image, chemin_sortie="image_contours.jpg"):

    # 2. Convertir en niveaux de gris (nécessaire pour la détection de contours)
    image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Appliquer un flou gaussien pour réduire le bruit et améliorer la détection
    flou = cv2.GaussianBlur(image_gris, (5, 5), 0)

    # 4. Détecter les contours avec l'algorithme de Canny
    # Vous pouvez ajuster les seuils (100 et 200) selon vos besoins
    contours_canny = cv2.Canny(flou, 30, 80)

    # 5. Trouver les contours vectoriels à partir de l'image Canny
    contours, _ = cv2.findContours(contours_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Dessiner les contours en noir sur l'image d'origine
    # Le paramètre (0, 0, 0) correspond à la couleur noire en BGR
    # Le dernier paramètre (ici 2) est l'épaisseur du trait en pixels
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 0, 0), thickness=2)

    # 7. Sauvegarder et afficher le résultat
    cv2.imwrite(chemin_sortie, image_contours)

    return image_contours

def recombine_multiscale(image_orig, base1, base2, base3, lf, lm, lc):

    """
    Recombine les couches de détails avec des poids spécifiques.
    - image_orig - base1 : Détails très fins (textures)
    - base1 - base2      : Détails moyens (traits)
    - base2 - base3      : Détails grossiers (ombres globales)
    """
    res = base3 + lc*(base2 - base3) + lm*(base1 - base2) + lf*(image_orig - base1)
    return np.clip(res, 0.0, 1.0) # On intègre la sécurité pour rester dans [0, 1]

