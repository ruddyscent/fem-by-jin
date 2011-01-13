#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
#                                                                       #
# An implementation of the 1-D finite element method for the exercise   #
# 3.4 of Jin p.80.                                                      #
#                                                                       #
# Kyungwon Chun (kwchun@gist.ac.kr)                                     #
#                                                                       #
#########################################################################


from cmath import exp, sqrt
from math import pi, cos, sin
from ex3_2 import Fem1d
from ch3_4_3 import analyticEz, analyticHz


class Fem1dQuadratic(Fem1d):
    def __init__(self, number_of_element, domain_length):
        Fem1d.__init__(self, number_of_element, domain_length)
        self.N = 2 * self.M + 1
        
        self.K = zeros((self.N, self.N), complex)
        self.b = zeros(self.N, complex)

    def setK(self):
        for i in xrange(self.M):
            alpha = self.alpha_func(i)
            l = self.l
            beta = self.beta_func(i)
            
            # Set just the upper triangluar elements of K.
            self.K[2*i+2, 2*i+2] = 7 * alpha / (3 * l) + 2 * l * beta / 15
            self.K[2*i, 2*i] += self.K[2*i+2, 2*i+2]
            self.K[2*i+1, 2*i+1] = 16 * alpha / (3 * l) + 8 * l * beta / 15
            self.K[2*i+1, 2*i+2] = -8 * alpha / (3 * l) + l * beta / 15 
            self.K[2*i, 2*i+1] = self.K[2*i+1, 2*i+2]
            self.K[2*i, 2*i+2] = alpha / (3 * l) - l * beta / 30

        self.K = self.K + self.K.T - diag(self.K.diagonal())
        
    def setB(self):
        for i in xrange(self.M):
            l = self.l
            f = self.f_func(i)
            
            self.b[2*i+2] = f * l / 6
            self.b[2*i] += self.b[2*i+2]
            self.b[2*i+1] = 2 * f * l / 3


class Fem1dCubic(Fem1d):
    def __init__(self, number_of_element, domain_length):
        Fem1d.__init__(self, number_of_element, domain_length)
        self.N = 3 * self.M + 1
        
        self.K = zeros((self.N, self.N), complex)
        self.b = zeros(self.N, complex)

    def setK(self):
        for i in xrange(self.M):
            alpha = self.alpha_func(i)
            l = self.l
            beta = self.beta_func(i)
            
            # Set just the upper triangluar elements of K.
            self.K[3*i+3, 3*i+3] = 37 * alpha / (10 * l) + 8 * l * beta / 105
            self.K[3*i, 3*i] += self.K[3*i+3, 3*i+3]
            self.K[3*i+2, 3*i+2] = 54 * alpha / (5 * l) + 27 * l * beta / 70
            self.K[3*i+1, 3*i+1] = self.K[3*i+2, 3*i+2]
            self.K[3*i+2, 3*i+3] = -189 * alpha / (40 * l) + 33 * l * beta / 560
            self.K[3*i, 3*i+1] = self.K[3*i+2, 3*i+3]
            self.K[3*i+1, 3*i+3] = 27 * alpha / (20 * l) - 3 * l * beta / 140
            self.K[3*i, 3*i+2] = self.K[3*i+1, 3*i+3]
            self.K[3*i+1, 3*i+2] = -297 * alpha / (40 * l) - 27 * l * beta / 560
            self.K[3*i, 3*i+3] = -13 * alpha / (40 * l) + 19 * l * beta / 1680
            
        self.K = self.K + self.K.T - diag(self.K.diagonal())
        
    def setB(self):
        for i in xrange(self.M):
            l = self.l
            f = self.f_func(i)
            
            self.b[3*i+3] = f * l / 8
            self.b[3*i] += self.b[3*i+3]
            self.b[3*i+2] = 3 * f * l / 8
            self.b[3*i+1] = self.b[3*i+2]


class SlabEzQuadratic(Fem1dQuadratic):
    def __init__(self, number_of_element, domain_length, wavelength, angle):
        Fem1dQuadratic.__init__(self, number_of_element, domain_length)
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
        self.setK()
        self.setB()
        self.setBoundaryConditions()
        self.solve()


class SlabHzQuadratic(SlabEzQuadratic):
    def alpha_func(self, i):
        return 1 / self.eps_r(i)
    
    def beta_func(self, i):
        tmp = -(self.mu_r(i) - sin(self.theta)**2 / self.eps_r(i)) * self.k0**2
        return tmp


class SlabEzCubic(Fem1dCubic):
    def __init__(self, number_of_element, domain_length, wavelength, angle):
        Fem1dCubic.__init__(self, number_of_element, domain_length)
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
        self.setK()
        self.setB()
        self.setBoundaryConditions()
        self.solve()


class SlabHzCubic(SlabEzCubic):
    def alpha_func(self, i):
        return 1 / self.eps_r(i)
    
    def beta_func(self, i):
        tmp = -(self.mu_r(i) - sin(self.theta)**2 / self.eps_r(i)) * self.k0**2
        return tmp


if __name__ == '__main__':
    from pylab import *
    
    plot_points = 24
    number_of_element = 35
    dTheta = pi / 2 / (plot_points - 1)
    
    ezRef = []
    for i in xrange(plot_points):
        refEz = analyticEz(number_of_element, 5*1, 1, i*dTheta)
        ezRef.append(abs(refEz.getR()))
        
    ezQuad = []
    for i in xrange(plot_points):
        slabEz = SlabEzQuadratic(number_of_element, 5*1, 1, i*dTheta)
        slabEz.mySolve()
        expnt = 1j * slabEz.k0 * slabEz.L * cos(slabEz.theta)
        numer = slabEz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        ezQuad.append(abs(numer / denom))
        
    ezCubic = []
    for i in xrange(plot_points):
        slabEz = SlabEzCubic(number_of_element, 5*1, 1, i*dTheta)
        slabEz.mySolve()
        expnt = 1j * slabEz.k0 * slabEz.L * cos(slabEz.theta)
        numer = slabEz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        ezCubic.append(abs(numer / denom))

    hzRef = []
    for i in xrange(plot_points):
        refHz = analyticHz(number_of_element, 5*1, 1, i*dTheta)
        hzRef.append(abs(refHz.getR()))
 
    hzQuad = []
    for i in xrange(plot_points):
        slabHz = SlabHzQuadratic(number_of_element, 5*1, 1, i*dTheta)
        slabHz.mySolve()
        expnt = 1j * slabHz.k0 * slabHz.L * cos(slabHz.theta)
        numer = slabHz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        hzQuad.append(abs(numer / denom))
        
    hzCubic = []
    for i in xrange(plot_points):
        slabHz = SlabHzCubic(number_of_element, 5*1, 1, i*dTheta)
        slabHz.mySolve()
        expnt = 1j * slabHz.k0 * slabHz.L * cos(slabHz.theta)
        numer = slabHz.phi[-1] - exp(expnt)
        denom = exp(-expnt)
        hzCubic.append(abs(numer / denom))

    # Plot the results.
        degree = []
        for i in xrange(plot_points):
            degree.append(i * 180 * dTheta / pi)
            
    plot(degree, ezRef, label="Ez with an analytical method")
    plot(degree, ezQuad, label="Ez with %d cells(quadratic)" % number_of_element)
    plot(degree, ezCubic, label="Ez with %d cells(cubic)" % number_of_element)
    plot(degree, hzRef, label="Hz with an analytical method")
    plot(degree, hzQuad, label="Hz with %d cells(quadratic)" % number_of_element)
    plot(degree, hzCubic, label="Hz with %d cells(cubic)" % number_of_element)
    xlabel(r"$\theta$ (degrees)")
    ylabel("Reflection Coefficient")
    title("Plane-wave Reflection by A Metal-backed Dielectric Slab")
    grid(True)
    legend()
    show()
 number_of_element)
    plot(degree, ezCubic, label="Ez with %d cells(cubic)" % number_of_element)
    plot(degree, hzRef, label="Hz with an analytical method")
    plot