#Roman Sultanov
#Imports
import numpy as np
from sympy import *
import vegas
import timeit
start = timeit.default_timer()



#Symbolic representation

#Given just after Eq. (50)
#Phase Space
theta_Xi, phi_Xi, theta_L, phi_L, theta_p, phi_p, theta_BL, phi_BL, theta_Bp, phi_Bp\
    = symbols('theta_Xi phi_Xi theta_L phi_L theta_p phi_p theta_BL phi_BL theta_Bp phi_Bp')

#Global Parameters
aPsi, Dphi, Pe, aXi, aBXi, aL, aBL, pXi, pBXi = symbols('aPsi Dphi Pe aXi aBXi aL aBL pXi pBXi')


#C matrix Eq. (37) and (38)
bPsi=sin(acos(-aPsi))*sin(Dphi)
yPsi=sin(acos(-aPsi))*cos(Dphi)

C = (3/(3+aPsi))*np.array([[1 + aPsi * cos(theta_Xi) ** 2, yPsi * Pe * sin(theta_Xi), bPsi * sin(theta_Xi) * cos(theta_Xi), (1 + aPsi) * Pe * cos(theta_Xi)],\
                           [yPsi * Pe * sin(theta_Xi), sin(theta_Xi) ** 2, 0, yPsi * sin(theta_Xi) * cos(theta_Xi)],\
                           [-bPsi * sin(theta_Xi) * cos(theta_Xi), 0, aPsi * sin(theta_Xi) ** 2, -bPsi*Pe*sin(theta_Xi)],\
                           [-(1+aPsi)*Pe*cos(theta_Xi), -yPsi*sin(theta_Xi)*cos(theta_Xi), -bPsi*Pe*sin(theta_Xi), -aPsi-cos(theta_Xi)**2]])


#A_Xi matrix Eq. (49) and (8) but for cascade
bXi= sqrt(1-aXi*aXi)*sin(pXi)
yXi= sqrt(1-aXi*aXi)*cos(pXi)

A_Xi = np.array([[1, 0, 0, aXi],\
                 [aXi*sin(theta_L)*cos(phi_L), yXi*cos(theta_L)*cos(phi_L)-bXi*sin(phi_L), -bXi*cos(theta_L)*cos(phi_L)-yXi*sin(phi_L), sin(theta_L)*cos(phi_L)],\
                 [aXi*sin(theta_L)*sin(phi_L), bXi*cos(phi_L)+yXi*cos(theta_L)*sin(phi_L), yXi*cos(phi_L)-bXi*cos(theta_L)*sin(phi_L), sin(theta_L)*sin(phi_L)],\
                 [aXi*cos(theta_L), -yXi*sin(theta_L), bXi*sin(theta_L), cos(theta_L)]])


#A_BXi matrix Eq. (49) and (8) but for anti-cascade
bBXi= sqrt(1-aBXi*aBXi)*sin(pBXi)
yBXi= sqrt(1-aBXi*aBXi)*cos(pBXi)

A_BXi = np.array([[1, 0, 0, aBXi],\
                 [aBXi*sin(theta_BL)*cos(phi_BL), yBXi*cos(theta_BL)*cos(phi_BL)-bBXi*sin(phi_BL), -bBXi*cos(theta_BL)*cos(phi_BL)-yBXi*sin(phi_BL), sin(theta_BL)*cos(phi_BL)],\
                 [aBXi*sin(theta_BL)*sin(phi_BL), bBXi*cos(phi_BL)+yBXi*cos(theta_BL)*sin(phi_BL), yBXi*cos(phi_BL)-bBXi*cos(theta_BL)*sin(phi_BL), sin(theta_BL)*sin(phi_BL)],\
                 [aBXi*cos(theta_BL), -yBXi*sin(theta_BL), bBXi*sin(theta_BL), cos(theta_BL)]])


#A_L matrix Eq. (49) but for lambda, but is reduced to a_[mu', 0] as seen in Eq. (50)
A_L = np.array([[1],\
                 [aL*sin(theta_p)*cos(phi_p)],\
                 [aL*sin(theta_p)*sin(phi_p)],\
                 [aL*cos(theta_p)]])


#A_BL matrix Eq. (49) but for anti-lambda, but is reduced to a_[nu', 0] as seen in Eq. (50)
A_BL = np.array([[1],\
                 [aBL*sin(theta_Bp)*cos(phi_Bp)],\
                 [aBL*sin(theta_Bp)*sin(phi_Bp)],\
                 [aBL*cos(theta_Bp)]])


#Joint angular distribution Eq. (50)
SUM=0
for mu in [0,1,2,3]:
    for nu in [0,1,2,3]:
        SUM += C[mu][nu] * (A_Xi[mu][0]*A_L[0] + A_Xi[mu][1]*A_L[1] + A_Xi[mu][2]*A_L[2] + A_Xi[mu][3]*A_L[3])\
                         * (A_BXi[nu][0]*A_BL[0] + A_BXi[nu][1]*A_BL[1] + A_BXi[nu][2]*A_BL[2] + A_BXi[nu][3]*A_BL[3])
P = (1/(4*pi)**5)*SUM[0]




#Fisher Information Matrix Eq. (52)
#Zero-matrix
I = np.zeros((9, 9))

#the different differentiations of P, as seen in Eq. (52) dP/domega_k
dPdomega = np.array([diff(P, aPsi), diff(P, Dphi), diff(P, Pe), diff(P, aXi), diff(P, aBXi), diff(P, aL), diff(P, aBL), diff(P, pXi), diff(P, pBXi)])

#Up to this point, the program barely takes a second to compute. Doing the integration and filling in the matrix takes
#Up all of the time. Running integ below at nitn=10 and neval=1000 takes about 269 seconds. nitn=20 and neval=10000 can
#take up to an hour.



#Iterate over the upper triangle of the matrix, the lower half is automatically filled
for i in range(9):
    for j in range(9):
        if j < i:
            continue

        #the integrand of Eq. (52), jacobian included
        INT=(1/P)*dPdomega[i]*dPdomega[j]*sin(theta_Xi)*sin(theta_L)*sin(theta_p)*sin(theta_BL)*sin(theta_Bp)
        #Global parameter values
        INT=INT.subs({aPsi:0.586, Dphi: 1.213, aXi: -0.376, aL: 0.757, pXi: 0.011, aBXi: 0.376, aBL: -0.757, pBXi: -0.011, Pe: 0.5})

        #Turns the symbolic expression into a function that can be evaluated
        IINT = lambdify([theta_Xi, phi_Xi, theta_L, phi_L, theta_p, phi_p, theta_BL, phi_BL, theta_Bp, phi_Bp], INT)
        def f(x):
            global IINT
            return IINT(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])

        #Vegas integration
        integ = vegas.Integrator([[0, pi], [0, 2*pi], [0, pi], [0, 2*pi], [0, pi], [0, 2*pi], [0, pi], [0, 2*pi], [0, pi], [0, 2*pi]])
        VAL = integ(f, nitn=10, neval=100).mean

        #Filling up the matrix
        I[i][j] = VAL
        I[j][i] = VAL



#Inverse of Information matrix
COV=np.linalg.inv(I)

#Returns list of standard deviation
s=np.zeros(9)
for i in range(9):
    s[i]=round(float(format(sqrt(COV[i][i]), '.2f')), 2)
print('s(aPsi), s(Dphi), s(Pe), s(aXi), s(aBXi), s(aL), s(aBL), s(pXi), s(pBXi)')
print(s)


stop = timeit.default_timer()
print('Run time:', stop - start, 's')