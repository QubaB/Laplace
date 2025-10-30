# deskový kondenzátor v 1D (s dieletrikem)
import numpy as np
import matplotlib.pyplot as plt
d = 5e-3  # mm  vzdálenost desek
N = 11 # pocet bodů sítě

dx = d/(N-1)

A = np.zeros((N,N))
B = np.zeros(N)

epsilonr=np.ones(N)
#epsilonr[5:]=2.0

# Dirichletova podmínka na levém okraji
A[0,0]=1.0
B[0]=0.0  
# Dirichletova podmínka na pravém okraji
U=1.0
A[N-1,N-1]=1.0
B[N-1]=U

for i in range(1,N-1):
    A[i,i]=-(epsilonr[i-1]+epsilonr[i+1])
    A[i,i-1]=epsilonr[i-1]
    A[i,i+1]=epsilonr[i+1]

Phi=np.linalg.solve(A,B)

Energie=0.0
epsilon=8.85e-12
S=1  # fiktivni plocha elektrod pro urceni objemu
for i in range(N-1):
    E=(Phi[i+1]-Phi[i])/dx  # intenzita elektrickeho pole
    wel=1/2*epsilonr[i]*epsilon*E**2
    Energie=Energie+wel*dx*S

print(f"Energie kondenzatoru je {Energie} J")
C=2*Energie/U**2
print(f"Kapacita kondenzatoru je {C} F")

Cvzorec=epsilon*S/d
print(f"Kapacita kondenzatoru ma byt {Cvzorec} F")
plt.plot(Phi)
plt.show()
print("konec")

