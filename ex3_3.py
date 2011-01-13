#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
#                                                                       #
# An implementation of the 1-D finite element method for the exercise   #
# 3.3 of Jin p.66.                                                      #
#                                                                       #
# Kyungwon Chun (kwchun@gist.ac.kr)                                     #
#                                                                       #
#########################################################################


from cmath import exp, sqrt
from math import pi, cos, sin
from numpy import inf
from ex3_2 import Fem1d


class dielectricSlabEz(Fem1d):
    def __init__(self, number_of_element, domain_length, wavelength, angle):
        Fem1d.__init__(self, number_of_element, domain_length)
        self.wavelength = wavelength
        self.theta = angle
        self.k0 = 2 * pi / self.wavelength
        
        # boundary condition parameters
        self.p = 0
        self.gamma = 1j * self.k0 * cos(self.theta)
        
        expnt = 1j * self.k0 * self.L * cos(self.theta)
        self.q = 2j * self.k0 * cos(self.theta) * exp(expnt)
        
    def eps_r(self, i):
        """Return relative permittivity of the ith element.
        
        """
        if i > self.M or i < 1:
            return 1.
        else:
            return 4. - .1j
    
    def mu_r(self, i):
        """Return relative permeability of the ith element.

        """
        if i > self.M or i < 1:
            return 1.
        else:
            return 2.
    
    def alpha_func(self, i):
        return 1 / self.mu_r(i)

    def beta_func(self, i):
        tmp = self.eps_r(i) - sin(self.theta)**2 / self.mu_r(i)
        tmp *= -self.k0**2
        return tmp
    
    def setBoundaryConditions(self):
        # Apply the boundary condition of the third kind at x=0.
        gamma = -1j * self.k0 * cos(self.theta)
        self.K[0,0] -= gamma

        q = 0
        self.b[0] -= q

        # Apply the boundary condition of the third kind at x=L.
        gamma = 1j * self.k0 * cos(self.theta)
        self.K[-1,-1] += gamma
        
        expnt = 1j * self.k0 * self.L * cos(self.theta)
        q = 2j * self.k0 * cos(self.theta) * exp(expnt)
        self.b[-1] += q

    def mySolve(self):
        self.generateMaterialProp()
#        self.printMaterialProp()
        self.setK()
        self.setB()
        self.setBoundaryConditions()
        self.solve()
#        print "phi =", self.phi


class analyticEz(dielectricSlabEz):
    """Expand the dielectricSlabEz class to be solved in an analytic way.

    """
    def __init__(self, number_of_element, domain_length, wavelength, angle):
        dielectricSlabEz.__init__(self, 
                                  number_of_element, domain_length, 
                                  wavelength, angle)
        self.R = empty(self.N + 1, complex)
        self.A = empty(self.N + 1, complex)
        self.flagR = False
        
    def kx(self, m):
        return self.k0 * \
            sqrt(self.mu_r(m) * self.eps_r(m) - sin(self.theta)**2)
            
    def eta(self, m):
        numer = self.mu_r(m) * self.kx(m+1) - self.mu_r(m+1) * self.kx(m)
        denom = self.mu_r(m) * self.kx(m+1) + self.mu_r(m+1) * self.kx(m)
        return numer / denom
    
    def getR(self):
        self.R[0] = 0
        self.R[1] = self.eta(0) * exp(2j * self.kx(1) * self.l)
        for m in xrange(1, self.N):
            exp1 = exp(2j * self.kx(m) * (m+1) * self.l)
            exp2 = exp(2j * self.kx(m+1) * (m+1) * self.l)
            numer = self.eta(m) + self.R[m] / exp1
            denom = 1 + self.eta(m) * self.R[m] / exp1
            self.R[m+1] = numer / denom * exp2
        self.flagR = True
        return self.R[-1]

    def getT(self):
        if self.flagR is False:
            print "getR method should be called before getT."
            return 

        self.A[-1] = 1.
        for m in xrange(self.N, 0, -1):
            exp1 = exp(1j * m * self.l * self.kx(m-1))
            exp2 = exp(1j * m * self.l * self.kx(m))
            numer = self.A[m] * (exp2 + self.R[m] / exp2)
            denom = exp1 + self.R[m-1] / exp1
            self.A[m-1] = numer / denom
        return self.A[0]
    

class analyticHz(analyticEz):
    def eta(self, m):
        numer = self.eps_r(m) * self.kx(m+1) - self.eps_r(m+1) * self.kx(m)
        denom = self.eps_r(m) * self.kx(m+1) + self.eps_r(m+1) * self.kx(m)
        return numer / denom


class dielectricSlabHz(dielectricSlabEz):
    def alpha_func(self, i):
        return 1 / self.eps_r(i)
    
    def beta_func(self, i):
        tmp = self.mu_r(i) - sin(self.theta)**2 / self.eps_r(i)
        tmp *= -self.k0**2
        return tmp


if __name__ == '__main__':
    from pylab import *

    plotPoints = 24
    dTheta = pi / 2 / (plotPoints - 1)
    
    ezRefR = []
    ezRefT = []
    for i in xrange(plotPoints):
        refEz = analyticEz(50, 5*1, 1, i*dTheta)
        ezRefR.append(abs(refEz.getR()))
        ezRefT.append(abs(refEz.getT()))

    ez100R = []
    ez100T = []
    for i in xrange(plotPoints):
        slabEz = dielectricSlabEz(100, 5*1, 1, i*dTheta)
        slabEz.mySolve()
        expnt = 1j * slabEz.k0 * slabEz.L * cos(slabEz.theta)
        numer = slabEz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        ez100R.append(abs(numer / denom))
        ez100T.append(abs(slabEz.phi[0]))
        
    hzRefR = []
    hzRefT = []
    for i in xrange(plotPoints):
        refHz = analyticHz(50, 5*1, 1, i*dTheta)
        hzRefR.append(abs(refHz.getR()))
        hzRefT.append(abs(refHz.getT()))

    hz100R = []
    hz100T = []
    for i in xrange(plotPoints):
        slabHz = dielectricSlabHz(100, 5*1, 1, i*dTheta)
        slabHz.mySolve()
        expnt = 1j * slabHz.k0 * slabHz.L * cos(slabHz.theta)
        numer = slabHz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        hz100R.append(abs(numer / denom))
        hz100T.append(abs(slabHz.phi[0]))

    # Plot the results.
    degree = []
    for i in xrange(plotPoints):
        degree.append(i * 180 * dTheta / pi)
        
    plot(degree, ezRefR, label="R(Ez) with an analytical method")
    plot(degree, ez100R, label="R(Ez) with a FEM")
    plot(degree, ezRefT, label="T(Ez) with an analytical method")
    plot(degree, ez100T, label="T(Ez) with a FEM")
    plot(degree, hzRefR, label="R(Hz) with an analytical method")
    plot(degree, hz100R, label="R(Hz) with a FEM")
    plot(degree, hzRefT, label="T(Hz) with an analytical method")
    plot(degree, hz100T, label="T(Hz) with a FEM")
    xlabel(r"$\theta$ (degrees)")
    ylabel("Reflection & Transmission  Coefficient")
    title("Plane-wave Reflection by A Dielectric Slab")
    grid(True)
    legend()
    show()
    
abel="T(Ez) with an analytical method")
    plot(degree, ez100T, label="T(Ez) with a FEM")
    plot(degree, hzRefR, label="R(Hz) with an analytical method")
    plot(degree, hz100R, label="R(Hz) with a FEM")
    plot(degree, hzRefT, label="T(Hz) with an a
alytical method")
 