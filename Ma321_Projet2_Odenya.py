######################################################################################
############### Ma321 - Projet1 - Groupe : Grar/Guyon/Khayat/Odenya/Tu ###############
###################################################################################### 


### Importations

import numpy as np
import matplotlib.pyplot as plt

p = np.loadtxt("dataP.dat")
q = np.loadtxt("dataQ.dat")

### Constantes

x0 = np.array([-9, -7])
tol = 1e-6
rho = 1e-3

m = p.size

X = np.ones((m,2))
X[:,1] = p

A = X.T@X

b = X.T@q


### Gradient à pas fixe

def GradientPasFixe(A,b,x0,rho,tol):
    xit = []
    xit.append(x0)
    i = 1
    iMax = 5*10**4
    r = 1
    x = x0
    while (np.linalg.norm(r)>tol and i<iMax):
        r = A@x-b
        d = -r
        x = x + rho*d
        i += 1
        xit.append(x)
    return(xit, x, i)

Res1 = GradientPasFixe(A, b, x0, rho, tol)


### Expérience Pas fixe ###

c_etoile = np.linalg.solve(A, b)

X = Res1[0]
n = Res1[2]
k = []
Rap = np.zeros(n-1)
for i in range(n-1):
    Rap[i] = np.linalg.norm(X[i+1] - c_etoile)/np.linalg.norm(X[i]-c_etoile)
    k.append(i)

plt.plot(k, Rap)
plt.xlim(0, 20)
plt.title('Test numérique pour le pas fixe')
plt.grid()
plt.show()


### Gradient à pas optimal

def GradientPasOptimal(A,b,x0,tol):
    xit = []
    xit.append(x0)
    i = 1
    iMax = 5*10**4
    r = 1
    x = x0
    while np.linalg.norm(r)>tol and i<iMax:
        r = A@x-b
        d = -r
        rho = ((r.T)@r)/((r.T)@A@r)
        x = x + rho*d
        i += 1
        xit.append(x)
    return(x,i)

Res2 = GradientPasOptimal(A, b, x0, tol)


### Construction d'une matrice dont on maitrise le conditionnement

a = 2*np.random.rand(2,2)-1 + 2*np.random.rand(2,2)-1
b = np.transpose(a)
c = a + b ## Nous construisons ainsi une matrice symétrique avec des coefficients aléatoires

C = 10 ## Nous fixons ici la valeur de conditionnement désirée

[u, s, v] = np.linalg.svd(c) ## Nous décomposons la matrice symétrique c créée en valeurs singulières afin de passer dans la base des vecteurs propres

s = s[0]*( 1-((C-1)/C)*(s[0]-s)/(s[0]-s[-1])) ## Nous avons créer un vecteur ayant pour coefficients les valeurs propres de c dans la base des vecteurs propres 

s = np.diag(s)  ### On transforme le vecteur s en matrice s avec les coefficients du vecteur sur la diagonale


B = u@s@np.transpose(v) ## On finit par remplir s qui est une matrice creuse avec les valeurs singulières trouvées précédemment

Co = np.linalg.cond(B) # On peut vérifier que le conditionnement de la matrice construite vaut C

## Expérience pas optimal

### mais comme nous l'expliquons dans le rapport, nous ne nous servirons pas de cette matrice créée
### notre étude portera sur l'expresion du taux en fonction de l'expression du conditionnement trouvée théoriquement

Cond = np.linspace(1, 300, 200)

tau = np.zeros(Cond.size)
for i in range(Cond.size):
    tau[i] = ((Cond[i] - 1)/(Cond[i] + 1))**2
    
    
plt.plot(Cond, tau)
plt.xlabel('cond(A)')
plt.ylabel('tau**2')
plt.title('Test numérique pour le pas optimal')
plt.grid()
plt.show()




############### METHODES A DIRECTIONS DE DESCENTE : ETUDE THEORIQUE


cp = np.linalg.eig(A)

Lambda1, v1 = cp[0][0],cp[1][0]/np.linalg.norm(cp[1][0])
Lambda2, v2 = cp[0][1],cp[1][1]/np.linalg.norm(cp[1][1])

Alpha = np.arange(0,2/Lambda2,1e-6) # On discrétise alpha entre 0 et 2/Lambda2

def tau1(alpha):
    return 1 - alpha*Lambda1

def tau2(alpha):
    return 1 - alpha*Lambda2

Tau1 = abs(tau1(Alpha))
Tau2 = abs(tau2(Alpha))

Tau = np.zeros(Alpha.size)
for i in range(Alpha.size):
    Tau[i] = max(Tau1[i], Tau2[i]) # On stocke dans une table le maximum calculé à chaque itération pour Tau1 et Tau2

plt.plot(Alpha, Tau1, 'x', color = 'red', label = '$tau_1$(alpha)')
plt.plot(Alpha, Tau2, color = 'blue', label = 'tau2(alpha)')
plt.plot(Alpha, Tau, color = 'green', label = 'tau(alpha)')
plt.axvline(x=2/Lambda2, color='yellow', linestyle = '--', label = ' x = 2/ $lambda_2$')
plt.axvline(x=2/(Lambda2 + Lambda1), color='gray', linestyle = '--', label = ' x = 2/($lambda_1$ + $lambda_2$)')
plt.title('Courbe représentative de $tau_1$, $tau_2$ et tau')
# plt.xlim(0.0014,0.00144)
# plt.ylim(0.96,1)   #Pour faire un zoom sur le point d'intersection
plt.legend()
plt.grid()
plt.show()

