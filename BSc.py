#Roman Sultanov
#Imports
import random
from math import *
import numpy as np
from iminuit import Minuit


#---------------------------------------------------------------------------------------------------------------------
#Probabiliy Distribution Function
def PDF(costL, pL, costp, pp, Px, Py, Pz, aXi, aL, pXi):
    Lx = np.sin(np.arccos(np.array(costL)))*np.cos(np.array(pL))
    Ly = np.sin(np.arccos(np.array(costL)))*np.sin(np.array(pL))
    Lz = np.array(costL)
    px = np.sin(np.arccos(np.array(costp)))*np.cos(np.array(pp))
    py = np.sin(np.arccos(np.array(costp)))*np.sin(np.array(pp))
    pz = np.array(costp)
    bXi= sqrt(1-aXi*aXi)*sin(pXi)
    yXi= sqrt(1-aXi*aXi)*cos(pXi)
    PL1= ((aXi+Px*Lx+Py*Ly+Pz*Lz)*Lx + bXi*(Py*Lz-Pz*Ly) + yXi*(Ly*(Px*Ly-Py*Lx)-Lz*(Pz*Lx-Px*Lz)))/(1+aXi*(Px*Lx+Py*Ly+Pz*Lz))
    PL2= ((aXi+Px*Lx+Py*Ly+Pz*Lz)*Ly + bXi*(Pz*Lx-Px*Lz) + yXi*(Lz*(Py*Lz-Pz*Ly)-Lx*(Px*Ly-Py*Lx)))/(1+aXi*(Px*Lx+Py*Ly+Pz*Lz))
    PL3= ((aXi+Px*Lx+Py*Ly+Pz*Lz)*Lz + bXi*(Px*Ly-Py*Lx) + yXi*(Lx*(Pz*Lx-Px*Lz)-Ly*(Py*Lz-Pz*Ly)))/(1+aXi*(Px*Lx+Py*Ly+Pz*Lz))
    z1 = Lx; z2 = Ly; z3 = Lz
    x1 = (Py*Lz-Pz*Ly)/np.sqrt((Py*Lz-Pz*Ly)**2 + (Pz*Lx-Px*Lz)**2 + (Px*Ly-Py*Lx)**2)
    x2 = (Pz*Lx-Px*Lz)/np.sqrt((Py*Lz-Pz*Ly)**2 + (Pz*Lx-Px*Lz)**2 + (Px*Ly-Py*Lx)**2)
    x3 = (Px*Ly-Py*Lx)/np.sqrt((Py*Lz-Pz*Ly)**2 + (Pz*Lx-Px*Lz)**2 + (Px*Ly-Py*Lx)**2)
    y1 = z2*x3-z3*x2; y2 = z3*x1-z1*x3; y3 = z1*x2-z2*x1
    PLz = PL1*z1 + PL2*z2 + PL3*z3
    PLx = PL1*x1 + PL2*x2 + PL3*x3
    PLy = PL1*y1 + PL2*y2 + PL3*y3
    pdf=(1+aXi*(Px*Lx + Py*Ly + Pz*Lz))*(1+aL*(PLx*px + PLy*py + PLz*pz))/(4*pi)**2
    return pdf

#Monte-Carlo Simulation
def generate(n, Px, Py, Pz, aXi, aL, pXi):
    VcostL=[]
    VpL   =[]
    Vcostp=[]
    Vpp   =[]
    i=0
    while i < n:
        costL=random.uniform(-1,1)
        pL=random.uniform(0,2*pi)
        costp=random.uniform(-1,1)
        pp=random.uniform(0,2*pi)
        pdf=PDF(costL, pL, costp, pp, Px, Py, Pz, aXi, aL, pXi)
        y=random.uniform(0, 0.0159)  #from 0 to maxvalue of PDF
        if y<pdf:
            VcostL.append(costL)
            VpL.append(pL)
            Vcostp.append(costp)
            Vpp.append(pp)
            i=i+1
    return VcostL, VpL, Vcostp, Vpp

#Negative Log-Likelihood Function
def f(Px, Py, Pz, aXi, aL, pXi):
    logL = np.sum(np.log(PDF(angle[0], angle[1], angle[2], angle[3], Px, Py, Pz, aXi, aL, pXi)))
    return -logL

#---------------------------------------------------------------------------------------------------------------------
# #Number of events
N=10000000

#Parameter values
PPx = 0.0
PPy = 0.0
PPz = 0.98      # Xi      #anti-Xi
PaXi=-0.376    #-0.376   # 0.371
PaL = 0.757    # 0.757   #-0.763
PpXi= 0.011    # 0.011   #-0.021

#Sample Generation and Minimization Preperation
print('Polarization Magnitude: ' + str(sqrt(PPx**2 + PPy**2 + PPz**2)))
angle=generate(N, Px=PPx, Py=PPy, Pz=0.98, aXi=PaXi, aL=PaL, pXi=PpXi)
m = Minuit(f, Px=0.01, Py=0.01, Pz=0.01, aXi=0.01, aL=0.5, pXi=0.01)
m.errordef = Minuit.LIKELIHOOD
m.limits["Px"] = (-0.1, 0.1)
m.limits["Py"] = (-0.1, 0.1)
m.limits["Pz"] = (-0.98, 0.98)
#Results
m.migrad()                  #Migrad minimization
m.hesse()                   #Hesse algorithm
print(m.values)             #Print parameter values
print(m.errors)             #Print parameter parabolic errors