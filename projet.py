import numpy as np
import cv2
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# img_test = np.array([
#     [0.1, 0.1,0.1,0.9, 0.9, 0.9, 0.9],
#     [0.1, 0.1,0.1,0.9, 0.9, 0.9, 0.9],
#     [0.1, 0.1,0.1,0.9, 0.9, 0.9, 0.9],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1]
# ])


img_test = cv2.imread('photo_test_epd.png') #ATTENTION : opencv est en couleur BGR
img_gris = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)


# On passe d'une échelle [0, 255] (entiers) à [0.0, 1.0] (décimaux)
img = img_gris.astype(np.float32) / 255.0
# variables
nb_l, nb_c = img.shape
print('nbcol=',nb_c,'nblin=',nb_l)
nb_p = nb_l * nb_c 

alpha = 1.5 
eps = 1e-4 
lamb = 0.5 # Paramètre de lissage lambda


# Construction de Dx tq : 'u(p+1) - u(p)' derivee horizontale (p+1 = (x+1, y))
dx_main = -np.ones(nb_p)
dx_plus1 = np.ones(nb_p)

# gestion des bords : On met donc la différence à 0 pour la dernière colonne de chaque ligne.
dx_main[nb_c-1::nb_c] = 0
dx_plus1[nb_c-1::nb_c] = 0


Dx = diags([dx_main, dx_plus1], [0, 1], shape=(nb_p, nb_p))   #diags gere directement ou placer la "double" diagonale avec "[0, 1]"

# Construction de Dy tq : 'u(p+nb_c) - u(p)' derivee horizontale (p+nb_c = (x, y+1))
dy_main = -np.ones(nb_p)
dy_plus_nbc = np.ones(nb_p)

####################
#dmd explication pour gerer Dy

#gestionn des bords : on mets le dernier pixel de chaque colonne a 0
dy_main[-nb_c:] = 0 
Dy = diags([dy_main, dy_plus_nbc], [0, nb_c], shape=(nb_p, nb_p))

#passage en logluminescence
L = np.log(img + 1e-4)
L_flat = L.flatten()

grad_x = Dx @ L_flat
grad_y = Dy @ L_flat

#Calculs des coefs ax et ay
ax = 1 / (np.abs(grad_x)**alpha + eps)
ay = 1 / (np.abs(grad_y)**alpha + eps)

#Construction des matrices de poids
Ax = diags([ax], [0], shape=(nb_p, nb_p))
Ay = diags([ay], [0], shape=(nb_p, nb_p))

# Matrice Lg
Lg = Dx.T @ Ax @ Dx + Dy.T @ Ay @ Dy

# Résolution du système linéaire : (I + lambda * Lg) * u = g
I = eye(nb_p) # Matrice identité creuse
A_sys = I + lamb * Lg
g = img.flatten()

#A_sys en format csr (condensed sparse row -> format des matrices creuses ) pour spsolve
A_sys = A_sys.tocsr()
u_flat = spsolve(A_sys, g)
img_lisse = u_flat.reshape((nb_l, nb_c))

# Affichage en couleur
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='jet', vmin=0, vmax=1)
plt.title("Image originale")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_lisse, cmap='jet', vmin=0, vmax=1)
plt.title(f"Couche de base (lambda={lamb})")
plt.axis('off')

plt.show()

#########
#cf gemini : pourquoi le resultat n'est pas fou + mettre en gris dans un premier temps