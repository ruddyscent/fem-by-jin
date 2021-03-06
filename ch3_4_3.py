#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
#                                                                       #
# An implementation of the 1-D finite element method for the plane-wave #
# reflectoin by a metal-backed dielectric slab presented at Jin pp. 61- #
# 66.                                                                   #
#                                                                       #
# Kyungwon Chun (kwchun@gist.ac.kr)                                     #
#                                                                       #
#########################################################################


from cmath import exp, sqrt
from math import pi, cos, sin
from ex3_2 import Fem1d


class dielectricSlabEz(Fem1d):
    def __init__(self, number_of_element, domain_length, wavelength, angle):
        Fem1d.__init__(self, number_of_element, domain_length)
        self.wavelength = wavelength
        self.theta = angle
        self.k0 = 2 * pi / self.wavelength
        
    def eps_r(self, i):
        """Return relative permittivity of the ith element.
        
        i: element number

        """
        if i > self.M:
            return 1.0
        else:
            return 4 + (2 - .1j) * (1 - i * self.l / self.L)**2
    
    def mu_r(self, i):
        """Return relative permeability of the ith element.
        
        i: element number

        """
        if i > self.M:
            return 1.0
        else:
            return 2 - .1j
    
    def alpha_func(self, i):
        return 1 / self.mu_r(i)

    def beta_func(self, i):
        tmp = -(self.eps_r(i) - sin(self.theta)**2 / self.mu_r(i)) * self.k0**2
        return tmp
    
    def setBoundaryConditions(self):
        # Apply the Dirchlet boundary condition at x=0.
        self.K[0,0] = 1
        self.K[0,1:] = 0
        
        p = 0
        self.b[0] = p
        
        # Apply the boundary condition of the third kind at x=L.
        gamma = 1j * self.k0 * cos(self.theta)
        self.K[-1,-1] += gamma

        expnt = 1j * self.k0 * self.L * cos(self.theta)
        q = 2j * self.k0 * cos(self.theta) * exp(expnt)
        self.b[-1] += q

    def mySolve(self):
        self.generateMaterialProp()
#       self.printMaterialProp()
        self.setK()
        self.setB()
        self.setBoundaryConditions()
        self.solve()
#       print "phi =", self.phi


class analyticEz(dielectricSlabEz):
    """Expand the dielectricSlabEz class to be solved in an analytic way.
    
    """
    R = -1
    
    def kx(self, m):
        return self.k0 * \
            sqrt(self.mu_r(m) * self.eps_r(m) - sin(self.theta)**2)

    def eta(self, m):
        numer = self.mu_r(m) * self.kx(m+1) - self.mu_r(m+1) * self.kx(m)
        denom = self.mu_r(m) * self.kx(m+1) + self.mu_r(m+1) * self.kx(m)
        return numer / denom

    def getR(self):
        for m in xrange(1, self.N):
            exp1 = exp(-2j * self.kx(m) * (m+1) * self.l)
            exp2 = exp(2j * self.kx(m+1) * (m+1) * self.l)
            numer = self.eta(m) + self.R * exp1
            denom = 1 + self.eta(m) * self.R * exp1
            self.R = numer / denom * exp2
        return self.R
    
    
class analyticHz(analyticEz):
    R = 1
    
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
    
    ezRef = []
    for i in xrange(plotPoints):
        refEz = analyticEz(50, 5*1, 1, i*dTheta)
        ezRef.append(abs(refEz.getR()))
        
    ez50 = []
    for i in xrange(plotPoints):
        slabEz = dielectricSlabEz(50, 5*1, 1, i*dTheta)
        slabEz.mySolve()
        expnt = 1j * slabEz.k0 * slabEz.L * cos(slabEz.theta)
        numer = slabEz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        ez50.append(abs(numer / denom))
        
    ez100 = []
    for i in xrange(plotPoints):
        slabEz = dielectricSlabEz(100, 5*1, 1, i*dTheta)
        slabEz.mySolve()
        expnt = 1j * slabEz.k0 * slabEz.L * cos(slabEz.theta)
        numer = slabEz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        ez100.append(abs(numer / denom))
        
    hzRef = []
    for i in xrange(plotPoints):
        refHz = analyticHz(50, 5*1, 1, i*dTheta)
        hzRef.append(abs(refHz.getR()))

    hz50 = []
    for i in xrange(plotPoints):
        slabHz = dielectricSlabHz(50, 5*1, 1, i*dTheta)
        slabHz.mySolve()
        expnt = 1j * slabHz.k0 * slabHz.L * cos(slabHz.theta)
        numer = slabHz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        hz50.append(abs(numer / denom))
        
    hz100 = []
    for i in xrange(plotPoints):
        slabHz = dielectricSlabHz(100, 5*1, 1, i*dTheta)
        slabHz.mySolve()
        expnt = 1j * slabHz.k0 * slabHz.L * cos(slabHz.theta)
        numer = slabHz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        hz100.append(abs(numer / denom))
        
    # Plot the results.
        degree = []
        for i in xrange(plotPoints):
            degree.append(i * 180 * dTheta / pi)
            
    plot(degree, ezRef, label="Ez with an analytical method")
    plot(degree, ez50, label="Ez with 50 cells")
    plot(degree, ez100, label="Ez with 100 cells")
    plot(degree, hzRef, label="Hz with an analytical method")
    plot(degree, hz50, label="Hz with 50 cells")
    plot(degree, hz100, label="Hz with 100 cells")
    xlabel(r"$\theta$ (degrees)")
    ylabel("Reflection Coefficient")
    title("Plane-wave Reflection by A Metal-backed Dielectric Slab")
    grid(True)
    legend()
    show()
        
    plot(degree, ezRef, label="Ez with an analytical met