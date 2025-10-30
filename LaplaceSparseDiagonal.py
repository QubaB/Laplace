# Laplaceova rovnice - reseni pomoci sparse diagonal
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

d=10.0  # velikost oblasti dxd
Nd=100  # pocet deleni ve smeru x a y
N=Nd*Nd  # pocet neznamych
dx=d/(Nd-1)  # krok site ve smeru x
dy=dx        # krok site ve smeru y
R1=1.0  # vnitřní poloměr
R2=d/2.0  # vnejší poloměr

main_diag = np.zeros(N)   # hlavní diagonála
diag_m1 = np.zeros(N - 1)  # dolní poddiagonála -1
diag_mNd = np.zeros(N - Nd)  # dolní poddiagonála -Nd
diag_p1 = np.zeros(N - 1)  # horní poddiagonála +1
diag_pNd = np.zeros(N - Nd)  # horní poddiagonála +Nd


B=np.zeros(N)    # vektor prave strany


def nastav_rovnici_i(i):
    """
        Nastavi koeficienty pro i-tou rovnici
    """
    #A[i,i]=-2.0*(1.0/dx**2+1.0/dy**2)
    main_diag[i]=-2.0*(1.0/dx**2+1.0/dy**2)
    if (i>0):
        #A[i,i-1]=1.0/dx**2
        diag_m1[i-1]=1.0/dx**2
    if (i>Nd):
        #A[i,i-Nd]=1.0/dy**2
        diag_mNd[i-Nd]=1.0/dy**2
    if (i<N-1):
        #A[i,i+1]=1.0/dx**2
        diag_p1[i]=1.0/dy**2
    if (i<N-Nd):
        #A[i,i+Nd]=1.0/dy**2
        diag_pNd[i]=1.0/dy**2

def nastav_dirichlet_i(i,Phi_i):
    """
        Nastavi radek v i-te rovnici na Dirichletovu okrajovou podminku
    """
    #A[i,:]=0.0  # vynuluj koeficienty
    if (i>0):
        diag_m1[i-1]=0.0
    if (i>Nd):
        diag_mNd[i-Nd]=0.0
    if (i<N-1):
        diag_p1[i]=0.0
    if (i<N-Nd):
        diag_pNd[i]=0.0
    #A[i,i]=1.0  # nastav koeficient na diagonale
    main_diag[i]=1.0
    B[i]=Phi_i  # nastav pozadovany potencial do prave strany


def nastav_okrajovou_podminku(ix,iy):
    """
        Nastavi hodnotu okrajove podminky pro bod ix,iy
    """
    i=(iy)*Nd+ix  # z indexu ix, iy urci poradove cislo bodu
    if (ix==0):
        # leva hranice
        nastav_dirichlet_i(i,0.0)
        return
    if (ix==Nd-1):
        # prava hranice
        nastav_dirichlet_i(i,1.0)
        return
    if (iy==0):
        # dolni hranice
        nastav_dirichlet_i(i,0.0)
        return
    if (iy==Nd-1):
        # horni hranice
        nastav_dirichlet_i(i,1.0)
        return
    
def nastav_vnitrni_valec(ix,iy):
    """
        Nastavi hodnotu okrajove podminky pro bod ix,iy
    """
    x=ix*dx  # poloha bodu
    y=iy*dy
    r=np.sqrt((x-d/2)**2+(y-d/2)**2)
    if (r<=R1):
        i=(iy)*Nd+ix  # z indexu ix, iy urci poradove cislo bodu
        nastav_dirichlet_i(i,0.0)
    return

def nastav_vnejsi_valec(ix,iy):
    """
        Nastavi hodnotu okrajove podminky pro bod ix,iy
    """
    x=ix*dx  # poloha bodu
    y=iy*dy
    r=np.sqrt((x-d/2)**2+(y-d/2)**2)
    if (r>=R2):
        i=(iy)*Nd+ix  # z indexu ix, iy urci poradove cislo bodu
        nastav_dirichlet_i(i,1.0)
    return


# nastav koeficienty v matici A
for i in range(N):
    nastav_rovnici_i(i)



# nastav Dirichletovu podminku na vnitrnim valci
for iy in range(Nd):
    for ix in range(Nd):
        nastav_vnitrni_valec(ix,iy)

# nastav Dirichletovu podminku na vnejsim valci
for iy in range(Nd):
    for ix in range(Nd):
        nastav_vnejsi_valec(ix,iy)


diagonals = [diag_mNd, diag_m1, main_diag, diag_p1, diag_pNd]
offsets = [-Nd, -1, 0, 1, Nd]
A = diags(diagonals, offsets, format='csr')


# vyres soustavu rovnic
Phi = spsolve(A, B)

# potrebujeme preformatovat Phi z vektoru do matice NdxNd
Phi_xy=Phi.reshape([Nd,Nd])


# vykresleni reseni
plt.figure(figsize=(6, 6))
plt.imshow(Phi_xy, aspect='auto', cmap='rainbow')
plt.colorbar()
plt.title('Potencial')
plt.show()

# nebo

x = np.arange(0, d+dx/2, dx)  # hodnoty x souradnic sitovych bodu
y = np.arange(0, d+dy/2, dy)
X, Y = np.meshgrid(x, y)  # souradnice X a Y vsech bodu

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
surface=ax.plot_surface(X, Y, Phi_xy,cmap='viridis')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('Phi [V]')
plt.savefig('plot.png')
plt.show()

print("konec")



