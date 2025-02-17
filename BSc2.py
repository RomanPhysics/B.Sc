#Roman Sultanov
#Imports
import jax.numpy as np
from jax import grad, jit
import vegas
import time

@jit
def PDF(theta_Xi, phi_Xi, theta_L, phi_L, theta_p, phi_p, theta_BL, phi_BL, theta_Bp, phi_Bp, aPsi, Dphi, Pe, aXi, aBXi, aL, aBL, pXi, pBXi):
    bPsi=np.sin(np.arccos(-aPsi))*np.sin(Dphi)
    yPsi=np.sin(np.arccos(-aPsi))*np.cos(Dphi)
    C = (3/(3+aPsi))*np.array([[1 + aPsi * np.cos(theta_Xi) ** 2, yPsi * Pe * np.sin(theta_Xi), bPsi * np.sin(theta_Xi) * np.cos(theta_Xi), (1 + aPsi) * Pe * np.cos(theta_Xi)],\
                               [yPsi * Pe * np.sin(theta_Xi), np.sin(theta_Xi) ** 2, 0, yPsi * np.sin(theta_Xi) * np.cos(theta_Xi)],\
                               [-bPsi * np.sin(theta_Xi) * np.cos(theta_Xi), 0, aPsi * np.sin(theta_Xi) ** 2, -bPsi*Pe*np.sin(theta_Xi)],\
                               [-(1+aPsi)*Pe*np.cos(theta_Xi), -yPsi*np.sin(theta_Xi)*np.cos(theta_Xi), -bPsi*Pe*np.sin(theta_Xi), -aPsi-np.cos(theta_Xi)**2]])

    bXi= np.sqrt(1-aXi*aXi)*np.sin(pXi)
    yXi= np.sqrt(1-aXi*aXi)*np.cos(pXi)
    A_Xi = np.array([[1, 0, 0, aXi],\
                     [aXi*np.sin(theta_L)*np.cos(phi_L), yXi*np.cos(theta_L)*np.cos(phi_L)-bXi*np.sin(phi_L), -bXi*np.cos(theta_L)*np.cos(phi_L)-yXi*np.sin(phi_L), np.sin(theta_L)*np.cos(phi_L)],\
                     [aXi*np.sin(theta_L)*np.sin(phi_L), bXi*np.cos(phi_L)+yXi*np.cos(theta_L)*np.sin(phi_L), yXi*np.cos(phi_L)-bXi*np.cos(theta_L)*np.sin(phi_L), np.sin(theta_L)*np.sin(phi_L)],\
                     [aXi*np.cos(theta_L), -yXi*np.sin(theta_L), bXi*np.sin(theta_L), np.cos(theta_L)]])

    bBXi= np.sqrt(1-aBXi*aBXi)*np.sin(pBXi)
    yBXi= np.sqrt(1-aBXi*aBXi)*np.cos(pBXi)
    A_BXi = np.array([[1, 0, 0, aBXi],\
                      [aBXi*np.sin(theta_BL)*np.cos(phi_BL), yBXi*np.cos(theta_BL)*np.cos(phi_BL)-bBXi*np.sin(phi_BL), -bBXi*np.cos(theta_BL)*np.cos(phi_BL)-yBXi*np.sin(phi_BL), np.sin(theta_BL)*np.cos(phi_BL)],\
                      [aBXi*np.sin(theta_BL)*np.sin(phi_BL), bBXi*np.cos(phi_BL)+yBXi*np.cos(theta_BL)*np.sin(phi_BL), yBXi*np.cos(phi_BL)-bBXi*np.cos(theta_BL)*np.sin(phi_BL), np.sin(theta_BL)*np.sin(phi_BL)],\
                      [aBXi*np.cos(theta_BL), -yBXi*np.sin(theta_BL), bBXi*np.sin(theta_BL), np.cos(theta_BL)]])


    A_L = np.array([[1],\
                    [aL*np.sin(theta_p)*np.cos(phi_p)],\
                    [aL*np.sin(theta_p)*np.sin(phi_p)],\
                    [aL*np.cos(theta_p)]])

    A_BL = np.array([[1],\
                    [aBL*np.sin(theta_Bp)*np.cos(phi_Bp)],\
                    [aBL*np.sin(theta_Bp)*np.sin(phi_Bp)],\
                    [aBL*np.cos(theta_Bp)]])

    SUM=0
    for mu in range(4):
        for nu in range(4):
            SUM += C[mu][nu] * (A_Xi[mu][0]*A_L[0] + A_Xi[mu][1]*A_L[1] + A_Xi[mu][2]*A_L[2] + A_Xi[mu][3]*A_L[3])\
                         * (A_BXi[nu][0]*A_BL[0] + A_BXi[nu][1]*A_BL[1] + A_BXi[nu][2]*A_BL[2] + A_BXi[nu][3]*A_BL[3])

    return (1/(4*np.pi)**5)*SUM[0]






def fishercomponent(i, j):
    par = np.array([0.586, 1.213, 0.5, -0.376, 0.371, 0.757, -0.763, 0.011, -0.021])
    dPdomega = [grad(PDF, argnums=10), grad(PDF, argnums=11), grad(PDF, argnums=12), grad(PDF, argnums=13),
                grad(PDF, argnums=14), grad(PDF, argnums=15), grad(PDF, argnums=16), grad(PDF, argnums=17),
                grad(PDF, argnums=18)]

    @jit
    def integrand(x):
        return (1 / PDF(*x, *par)) \
            * dPdomega[i](*x, *par) \
            * dPdomega[j](*x, *par) \
            * np.sin(x[0]) * np.sin(x[2]) * np.sin(x[4]) * np.sin(x[6]) * np.sin(x[8])

    integ = vegas.Integrator([[0, np.pi], [-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]])
    integ(integrand, nitn=10, neval=100000)
    intval=integ(integrand, nitn=15, neval=100000)

    return intval.mean

# Define i and j
i, j = 0, 0

# Get the properly compiled function
funct = fishercomponent(i, j)

print(funct)
