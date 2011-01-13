#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
#                                                                       #
# An implementation of the 1-D finite element method for the exercise   #
# 3.2 of Jin p.60.                                                      #
#                                                                       #
# Kyungwon Chun (kwchun@gist.ac.kr)                                     #
#                                                                       #
#########################################################################


from numpy import zeros, diag
from scipy.linalg import solve


class Fem1d:
    def __init__(self, number_of_element, domain_length):
        self.M = number_of_element
        self.L = float(domain_length)
        self.l = self.L / self.M  # node distance
        self.N = self.M + 1  # number of nodes
        self.phi = None
        
        # boundary conditon parameters
        self.p = 0j
        self.gamma = 0j
        self.q = 0j
        
        self.alpha = zeros(self.M, complex)
        self.beta = zeros(self.M, complex)
        self.f = zeros(self.M, complex)
        
        self.K = zeros((self.N, self.N), complex)
        self.b = zeros(self.N, complex)
        
    def alpha_func(self, i):
        """
        i: element number

        """
        return 0j

    def beta_func(self, i):
        """
        i: element number

        """
        return 0j

    # a forcing function
    def f_func(self, i):
        """
        i: element number

        """
        return 0j
    
    def generateMaterialProp(self):
        for i in xrange(self.M):
            self.alpha[i] = self.alpha_func(i)
            self.beta[i] = self.beta_func(i)
            self.f[i] = self.f_func(i)
            
    def printMaterialProp(self):
        print "M =", self.M
        print "alpha =", self.alpha
        print "beta =", self.beta
        print "f =", self.f
        
    def setK(self):
        # diagonal components
        self.K[0,0] = self.alpha[0] / self.l + self.beta[0] * self.l / 3
        for i in xrange(1, self.N-1):
            self.K[i,i] = self.alpha[i-1]/self.l + self.beta[i-1]*self.l/3\
                + self.alpha[i]/self.l + self.beta[i]*self.l/3
        self.K[-1,-1] = self.alpha[-1]/self.l + self.beta[-1]*self.l/3    
        
        # off-diagonal components
        for i in xrange(0, self.N-1):
            self.K[i,i+1] = self.beta[i]*self.l/6 - self.alpha[i]/self.l
            self.K[i+1,i] = self.K[i,i+1]
        
    def setB(self):
        self.b[0] = self.f[0] * self.l / 2
        self.b[1:-1] = (self.f[:-1] + self.f[1:]) * self.l / 2
        self.b[-1] = self.f[-1] * self.l / 2
        
    def setBoundaryCondition(self):
        pass
    
    def solve(self):
        self.phi = solve(self.K, self.b)


if __name__ == '__main__':
    import math
    import cmath
    
    class dielectricSlab(Fem1d):
        def __init__(self, number_of_element, domain_length, wavelength, incidentAngle):
            Fem1d.__init__(self, number_of_element, domain_length)
            self.wavelength = wavelength
            self.theta = incidentAngle
            self.k0 = 2 * math.pi / self.wavelength
            
            # boundary condition parameters
            self.p = 0
            self.gamma = 1j * self.k0 * math.cos(self.theta)
            
            expnt = 1j * self.k0 * self.L * math.cos(self.theta)
            self.q = 2j * self.k0 * math.cos(self.theta) * cmath.exp(expnt)
            
        def eps_r(self, i):
            """Return relative permittivity of the ith element.
        
            """
            return 4 + (2 - .1j) * (1 - i * self.l / self.L)**2
        
        def mu_r(self, i):
            """Return relative permeability of the ith element.

            """
            return 2 - .1j
        
        def alpha_func(self, i):
            return 1 / self.mu_r(i)

        def beta_func(self, i):
            tmp = -(self.eps_r(i) - math.sin(self.theta)**2 / self.mu_r(i)) * self.k0**2
            return tmp
        
        def setBoundaryConditions(self):
            # Dirchlet boundary condition
            self.K[0,0] = 1
            self.K[0,1:] = 0
            self.b[0] = self.p
            
            # Neumann boundary condition
            self.K[-1,-1] += self.gamma
            self.b[-1] += self.q
            
        def mySolve(self):
            self.generateMaterialProp()
            self.printMaterialProp()
            self.setK()
            self.setB()
            self.setBoundaryConditions()
            self.solve()
            print "phi =", self.phi
            
    slab = dielectricSlab(3, 5, 1, 5)
    slab.mySolve()
self.K[0,1:] = 0
            self.b[0] = self.p
            
            # Neumann boundary condition
            self.K[-1,-1] += self.gamma
            self.b[-1] += self.q
            
        def mySolve(self):
            self.generateMaterialProp()
            self.printMaterialProp()
            self.setK()
            s