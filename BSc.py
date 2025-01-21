#Roman Sultanov
#Imports
import numpy as np
from iminuit import Minuit
from numba import njit
np.set_printoptions(precision=19)
np.random.seed(6)

#-----------------------------------------------------------------------------------------------------------------------
#Probabiliy Distribution Function
@njit
def PDF(costL, pL, costp, pp, Px, Py, Pz, aXi, aL, pXi):
    Lx = np.sin(np.arccos(costL))*np.cos(pL)
    Ly = np.sin(np.arccos(costL))*np.sin(pL)
    Lz = costL
    px = np.sin(np.arccos(costp))*np.cos(pp)
    py = np.sin(np.arccos(costp))*np.sin(pp)
    pz = costp
    bXi = np.sqrt(1-aXi*aXi)*np.sin(pXi)
    yXi = np.sqrt(1-aXi*aXi)*np.cos(pXi)
    PL1 = ((aXi+Px*Lx+Py*Ly+Pz*Lz)*Lx + bXi*(Py*Lz-Pz*Ly) + yXi*(Ly*(Px*Ly-Py*Lx)-Lz*(Pz*Lx-Px*Lz)))/(1+aXi*(Px*Lx+Py*Ly+Pz*Lz))
    PL2 = ((aXi+Px*Lx+Py*Ly+Pz*Lz)*Ly + bXi*(Pz*Lx-Px*Lz) + yXi*(Lz*(Py*Lz-Pz*Ly)-Lx*(Px*Ly-Py*Lx)))/(1+aXi*(Px*Lx+Py*Ly+Pz*Lz))
    PL3 = ((aXi+Px*Lx+Py*Ly+Pz*Lz)*Lz + bXi*(Px*Ly-Py*Lx) + yXi*(Lx*(Pz*Lx-Px*Lz)-Ly*(Py*Lz-Pz*Ly)))/(1+aXi*(Px*Lx+Py*Ly+Pz*Lz))
    z1 = Lx; z2 = Ly; z3 = Lz
    x1 = (Py*Lz-Pz*Ly)/np.sqrt((Py*Lz-Pz*Ly)**2 + (Pz*Lx-Px*Lz)**2 + (Px*Ly-Py*Lx)**2)
    x2 = (Pz*Lx-Px*Lz)/np.sqrt((Py*Lz-Pz*Ly)**2 + (Pz*Lx-Px*Lz)**2 + (Px*Ly-Py*Lx)**2)
    x3 = (Px*Ly-Py*Lx)/np.sqrt((Py*Lz-Pz*Ly)**2 + (Pz*Lx-Px*Lz)**2 + (Px*Ly-Py*Lx)**2)
    y1 = z2*x3-z3*x2; y2 = z3*x1-z1*x3; y3 = z1*x2-z2*x1
    PLz = PL1*z1 + PL2*z2 + PL3*z3
    PLx = PL1*x1 + PL2*x2 + PL3*x3
    PLy = PL1*y1 + PL2*y2 + PL3*y3
    pdf = (1+aXi*(Px*Lx + Py*Ly + Pz*Lz))*(1+aL*(PLx*px + PLy*py + PLz*pz))/(4*np.pi)**2
    return pdf

#Monte-Carlo Generation
@njit
def MonteCarlo(n, Px, Py, Pz, aXi, aL, pXi):
    data=[]
    while len(data) < n:
        costL = np.random.uniform(-1, 1)
        pL = np.random.uniform(-np.pi, np.pi)
        costp = np.random.uniform(-1, 1)
        pp = np.random.uniform(-np.pi, np.pi)
        pdf = PDF(costL, pL, costp, pp, Px, Py, Pz, aXi, aL, pXi)
        y = np.random.uniform(0, 0.0159)  #from 0 to maxvalue of PDF
        if pdf>0.0159:
            print(pdf)
            print([costL, pL, costp, pp])
        if y<pdf:
            data.append([costL, pL, costp, pp])
    return np.array(data)

#Number of events
N=10000000

#Parameter values
Px = 0.0
Py = 0.0
Pz = 0.90     # Xi      #anti-Xi
aXi=-0.376    #-0.376   # 0.371
aL = 0.757    # 0.757   #-0.763
pXi= 0.011    # 0.011   #-0.021

#Generate data
angles = MonteCarlo(N, Px, Py, Pz, aXi, aL, pXi)

#Negative Log-Likelihood Function
@njit(fastmath=True, nogil=True)
def NLL(Px, Py, Pz, aXi, aL, pXi):
    logL = np.sum(np.log(PDF(angles[:,0], angles[:,1], angles[:,2], angles[:,3], Px, Py, Pz, aXi, aL, pXi)))
    return -logL

#-----------------------------------------------------------------------------------------------------------------------

#Results
print('Polarization Magnitude: ' + str(np.sqrt(Px**2 + Py**2 + Pz**2)))
m = Minuit(NLL, Px=Px, Py=Py, Pz=Pz, aXi=aXi, aL=aL, pXi=pXi)
m.errordef = Minuit.LIKELIHOOD
m.strategy = 2
m.limits["Px"] = (-0.1, 0.1)
m.limits["Py"] = (-0.1, 0.1)
m.limits["Pz"] = (-1.0, 1.0)
m.limits["aXi"] = (-1.0, 1.0)
m.limits["aL"]  = (-1.0, 1.0)
m.limits["pXi"] = (-np.pi, np.pi)
m.migrad()                  #Migrad minimization
m.hesse()                   #Hesse algorithm
print(m.values)
print(m.errors)
print(m.covariance)
