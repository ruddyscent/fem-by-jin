#########################################################################
# An implementation of the 1-D finite element method for the exercise   #
# 3.2 of Jin p.60.                                                      #
#                                                                       #                                                                    #
#########################################################################

from numpy.core import zeros, arange
from LinearAlgebra import solve_linear_equations
from scipy.linalg import solve

class fem1d:
    def __init__(self, numberOfElement, domainLength):
        self.M = numberOfElement
        self.L = domainLength
        self.l = float(self.L)/self.M  # node distance
        self.N = self.M + 1  # number of nodes
        self.phi = None
        
        # boundary conditon parameters
        self.p = complex(0, 0)
        self.gamma = complex(0, 0)
        self.q = complex(0, 0)
        
        self.alpha = zeros(self.M, complex)
        self.beta = zeros(self.M, complex)
        self.f = zeros(self.M, complex)
        
        self.K = zeros((self.M+1, self.M+1), complex)
        self.b = zeros(self.M+1, complex)
        
    def alpha_func(self, i):
        return complex(0, 0)

    def beta_func(self, i):
        return complex(0, 0)

    # a forcing function
    def f_func(self, i):
        return complex(0, 0)
    
    def generateMaterialProp(self):
        for i in arange(self.M):
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
        self.K[0][0] = self.alpha[0]/self.l + self.beta[0]*self.l/3
        for i in arange(1, self.N-1):
            self.K[i][i] = self.alpha[i-1]/self.l + self.beta[i-1]*self.l/3 \
                + self.alpha[i]/self.l + self.beta[i]*self.l/3
        self.K[self.N-1][self.N-1] = self.alpha[self.M-1]/self.l + self.beta[self.M-1]*self.l/3    
        
        # off-diagonal components
        for i in arange(0, self.N-1):
            self.K[i][i+1] = self.beta[i]*self.l/6 - self.alpha[i]/self.l
            self.K[i+1][i] = self.K[i][i+1]
        
    def setB(self):
        self.b[0] = self.f[0]*self.l/2
        for i in arange(1, self.N-1):
            self.b[i] = (self.f[i-1]+self.f[i])*self.l/2
        self.b[self.N-1] = self.f[self.M-1]*self.l/2
        
    def setBoundaryCondition(self):
        pass
    
    def solve(self):
        self.phi = solve(self.K, self.b)
#        self.phi = solve_linear_equations(self.K, self.b) 
        
if __name__ == '__main__':
    import math
    import cmath
    
    class dielectricSlab(fem1d):
        def __init__(self, numberOfElement, domainLength, wavelength, incidentAngle):
            fem1d.__init__(self, numberOfElement, domainLength)
            self.wavelength = wavelength
            self.theta = incidentAngle
            self.k0 = 2*math.pi/wavelength
            
            # boundary condition parameters
            self.p = 0
            self.gamma = complex(0, self.k0*math.cos(self.theta))
            
            exponent = complex(0, self.k0*self.L*math.cos(self.theta))
            self.q = complex(0, 2*self.k0*math.cos(self.theta)*cmath.exp(exponent))
            
        def epsilon_r(self, i):
            return 4 + (2-complex(0, .1))*(1-i*self.l/self.L)**2
        
        def mu_r(self, i):
            return complex(2, -.1)
        
        def alpha_func(self, i):
            return 1 / self.mu_r(i)

        def beta_func(self, i):
            tmp = self.epsilon_r(i) - math.sin(self.theta)**2/self.mu_r(i)
            tmp *= -self.k0**2
            return tmp
        
        def setBoundaryConditions(self):
            # Dirchlet boundary condition
            self.K[0][0] = 1
            for i in arange(1, self.N):
                self.K[0][i] = 0
            self.b[0] = self.p
            
            # Neumann boundary condition
            self.K[-1][-1] += self.gamma
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