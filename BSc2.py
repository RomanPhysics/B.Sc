#Roman Sultanov
#Imports
import numpy as np
from numba import njit, prange
import vegas
import time

@njit(fastmath=True)
def PDF(theta_Xi, phi_Xi, theta_L, phi_L, theta_p, phi_p, theta_BL, phi_BL, theta_Bp, phi_Bp, aPsi, Dphi, Pe, aXi, aBXi, aL, aBL, pXi, pBXi):
    
    sqrtaPsi = np.sqrt(1 - aPsi*aPsi)
    bPsi = sqrtaPsi * np.sin(Dphi)
    yPsi = sqrtaPsi * np.cos(Dphi)

    C = (3 / (3 + aPsi)) * np.array([
        [1 + aPsi * np.cos(theta_Xi)**2, yPsi * Pe * np.sin(theta_Xi), bPsi * np.sin(theta_Xi) * np.cos(theta_Xi), (1 + aPsi) * Pe * np.cos(theta_Xi)],
        [yPsi * Pe * np.sin(theta_Xi), np.sin(theta_Xi)**2, 0, yPsi * np.sin(theta_Xi) * np.cos(theta_Xi)],
        [-bPsi * np.sin(theta_Xi) * np.cos(theta_Xi), 0, aPsi * np.sin(theta_Xi)**2, -bPsi * Pe * np.sin(theta_Xi)],
        [-(1 + aPsi) * Pe * np.cos(theta_Xi), -yPsi * np.sin(theta_Xi) * np.cos(theta_Xi), -bPsi * Pe * np.sin(theta_Xi), -aPsi - np.cos(theta_Xi)**2]
    ])

    sqrtaXi = np.sqrt(1 - aXi*aXi)
    bXi = sqrtaXi * np.sin(pXi)
    yXi = sqrtaXi * np.cos(pXi)
    A_Xi = np.array([
        [1, 0, 0, aXi],
        [aXi*np.sin(theta_L)*np.cos(phi_L), yXi*np.cos(theta_L)*np.cos(phi_L)-bXi*np.sin(phi_L), -bXi*np.cos(theta_L)*np.cos(phi_L)-yXi*np.sin(phi_L), np.sin(theta_L)*np.cos(phi_L)],
        [aXi*np.sin(theta_L)*np.sin(phi_L), bXi*np.cos(phi_L)+yXi*np.cos(theta_L)*np.sin(phi_L), yXi*np.cos(phi_L)-bXi*np.cos(theta_L)*np.sin(phi_L), np.sin(theta_L)*np.sin(phi_L)],
        [aXi*np.cos(theta_L), -yXi*np.sin(theta_L), bXi*np.sin(theta_L), np.cos(theta_L)]
    ])

    sqrtBaXi = np.sqrt(1 - aBXi*aBXi)
    bBXi = sqrtBaXi * np.sin(pBXi)
    yBXi = sqrtBaXi * np.cos(pBXi)
    A_BXi = np.array([
        [1, 0, 0, aBXi],
        [aBXi*np.sin(theta_BL)*np.cos(phi_BL), yBXi*np.cos(theta_BL)*np.cos(phi_BL)-bBXi*np.sin(phi_BL), -bBXi*np.cos(theta_BL)*np.cos(phi_BL)-yBXi*np.sin(phi_BL), np.sin(theta_BL)*np.cos(phi_BL)],
        [aBXi*np.sin(theta_BL)*np.sin(phi_BL), bBXi*np.cos(phi_BL)+yBXi*np.cos(theta_BL)*np.sin(phi_BL), yBXi*np.cos(phi_BL)-bBXi*np.cos(theta_BL)*np.sin(phi_BL), np.sin(theta_BL)*np.sin(phi_BL)],
        [aBXi*np.cos(theta_BL), -yBXi*np.sin(theta_BL), bBXi*np.sin(theta_BL), np.cos(theta_BL)]
    ])

    A_L = np.array([1, aL*np.sin(theta_p)*np.cos(phi_p), aL*np.sin(theta_p)*np.sin(phi_p), aL*np.cos(theta_p)])
    A_BL = np.array([1, aBL*np.sin(theta_Bp)*np.cos(phi_Bp), aBL*np.sin(theta_Bp)*np.sin(phi_Bp), aBL*np.cos(theta_Bp)])

    dot_XiL = np.dot(A_Xi, A_L)
    dot_BXiBL = np.dot(A_BXi, A_BL)
    
    SUM = np.sum(C * np.outer(dot_XiL, dot_BXiBL))
    
    return (1.0 / ((4 * np.pi)**5)) * SUM

@njit(fastmath=True)
def dPdomega(theta_Xi, phi_Xi, theta_L, phi_L, theta_p, phi_p, theta_BL, phi_BL, theta_Bp, phi_Bp, aPsi, Dphi, Pe, aXi, aBXi, aL, aBL, pXi, pBXi, i):
    h = 1e-6
    x = np.array([theta_Xi, phi_Xi, theta_L, phi_L, theta_p, phi_p, theta_BL, phi_BL, theta_Bp, phi_Bp, aPsi, Dphi, Pe, aXi, aBXi, aL, aBL, pXi, pBXi])

    x[i] += h
    f_forward = PDF(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],
                    x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18])

    x[i] -= 2*h    
    f_backward = PDF(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],
                    x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18])

    dPDF = (f_forward - f_backward) / (2 * h)

    return dPDF

def fishermatrix():
    I = np.zeros((9, 9))
    global par
    
    bounds = [[0, np.pi], [-np.pi, np.pi]] * 5
    integ = vegas.Integrator(bounds)

    for i in prange(10, 19):
        for j in prange(i, 19):

            @njit(fastmath=True)
            def integrand(x):
                return (1 / PDF(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8])) \
                    * dPdomega(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], i) \
                    * dPdomega(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], j) \
                    * np.sin(x[0]) * np.sin(x[2]) * np.sin(x[4]) * np.sin(x[6]) * np.sin(x[8])

            integ(integrand, nitn=5, neval=1000000)
            intval=integ(integrand, nitn=10, neval=1000000)
            print(f"Fisher element I[{i+1},{j+1}] = {intval.mean}")
            I[i][j] = intval.mean
            I[j][i] = intval.mean

    return I

#               aPsi,   Dphi,  Pe,   aXi,   aBXi,  aL,     aBL,   pXi,    pBXi
par = np.array([0.586, 1.213, 0.5, -0.376, 0.371, 0.757, -0.763, 0.011, -0.021])

start_time = time.time()
COV = np.linalg.inv(fishermatrix())
end_time = time.time()
totalseconds = end_time - start_time
print(f"Time taken: {int(totalseconds//60)} m, {totalseconds % 60:.2f} s.")
print("Covariance matrix:")
print(COV)
