#########################################################################
# An implementation of the 1-D finite element method for the plane-wave #
# reflectoin by a metal-backed dielectric slab presented at Jin pp. 61- #
# 66.                                                                   #                                                                    #
#########################################################################

from cmath import pi, cos, sin, exp, sqrt
from numpy.core import arange

from exercise3_2 import fem1d

class dielectricSlabEz(fem1d):
    def __init__(self, numberOfElement, domainLength, wavelength, incidentAngle):
        fem1d.__init__(self, numberOfElement, domainLength)
        self.wavelength = wavelength
        self.theta = incidentAngle
        self.k0 = 2*pi/wavelength
        
        # boundary condition parameters
        self.p = 0
        self.gamma = complex(0, self.k0*cos(self.theta))
        
        exponent = complex(0, self.k0*self.L*cos(self.theta))
        self.q = complex(0, 2*self.k0*cos(self.theta)*exp(exponent))
        
    def epsilon_r(self, i):
        return 4 + complex(2, -.1)*(1-i*self.l/self.L)**2
    
    def mu_r(self, i):
        return complex(2, -.1)
    
    def alpha_func(self, i):
        return 1 / self.mu_r(i)

    def beta_func(self, i):
        tmp = self.epsilon_r(i) - sin(self.theta)**2/self.mu_r(i)
        tmp *= -self.k0**2
        return tmp
    
    def setBoundaryConditions(self):
        # Dirchlet boundary condition
        self.K[0][0] = 1
        for i in range(1, self.N):
            self.K[0][i] = 0
        self.b[0] = self.p
        
        # Neumann boundary condition
        self.K[-1][-1] += self.gamma
        self.b[-1] += self.q
        
    def mySolve(self):
        self.generateMaterialProp()
#        self.printMaterialProp()
        self.setK()
        self.setB()
        self.setBoundaryConditions()
        self.solve()
#        print "phi =", self.phi

class analyticEz(dielectricSlabEz):
    R = -1
    
    def k(self, m):
        return self.k0 * \
            sqrt(self.mu_r(m)*self.epsilon_r(m) - sin(self.theta)**2)
            
    def eta(self, m):
        numerator = self.mu_r(m)*self.k(m+1)**2 - self.mu_r(m+1)*self.k(m)**2
        denominator = self.mu_r(m)*self.k(m+1)**2 + self.mu_r(m+1)*self.k(m)**2
        return numerator/denominator
    
    def getR(self):
        for m in arange(self.M):
            exponent1 = complex(0, -2*self.k(m)*(m+1)*self.l)
            exponent2 = complex(0, 2*self.k(m+1)*(m+1)*self.l)
            numerator = self.eta(m) + self.R*exp(exponent1)
            denominator = 1 + self.eta(m)*self.R*exp(exponent1)
            self.R = numerator/denominator*exp(exponent2)
        return self.R

class analyticHz(analyticEz):
    R = 1
    
    def eta(self, m):
        numerator = self.epsilon_r(m)*self.k(m+1)**2 - self.epsilon_r(m+1)*self.k(m)**2
        denominator = self.epsilon_r(m)*self.k(m+1)**2 + self.epsilon_r(m+1)*self.k(m)**2
        return numerator/denominator
          
class dielectricSlabHz(dielectricSlabEz):
    def alpha_func(self, i):
        return 1 / self.epsilon_r(i)
    
    def beta_func(self, i):
        tmp = self.mu_r(i) - sin(self.theta)**2/self.epsilon_r(i)
        tmp *= -self.k0**2
        return tmp

if __name__ == '__main__':
    from pylab import *

    plotPoints = 16
    dTheta = pi/2/(plotPoints-1)
    
    ezRef = []
    for i in range(plotPoints):
        refEz = analyticEz(50, 5*1, 1, i*dTheta)
        ezRef.append(abs(refEz.getR()))

    ez50 = []
    for i in range(plotPoints):
        slabEz = dielectricSlabEz(50, 5*1, 1, i*dTheta)
        slabEz.mySolve()
        exponent = complex(0, slabEz.k0*slabEz.L*cos(slabEz.theta))
        numerator = slabEz.phi[-1] - exp(exponent)
        denominator = exp(-exponent)
        ez50.append(abs(numerator/denominator))

    ez100 = []
    for i in range(plotPoints):
        slabEz = dielectricSlabEz(100, 5*1, 1, i*dTheta)
        slabEz.mySolve()
        exponent = complex(0, slabEz.k0*slabEz.L*cos(slabEz.theta))
        numerator = slabEz.phi[-1] - exp(exponent)
        denominator = exp(-exponent)
        ez100.append(abs(numerator/denominator))
    
    hzRef = []
    for i in range(plotPoints):
        refHz = analyticHz(50, 5*1, 1, i*dTheta)
        hzRef.append(abs(refHz.getR()))

    hz50 = []
    for i in range(plotPoints):
        slabHz = dielectricSlabHz(50, 5*1, 1, i*dTheta)
        slabHz.mySolve()
        exponent = complex(0, slabHz.k0*slabHz.L*cos(slabHz.theta))
        numerator = slabHz.phi[-1] - exp(exponent)
        denominator = exp(-exponent)
        hz50.append(abs(numerator/denominator))

    hz100 = []
    for i in range(plotPoints):
        slabHz = dielectricSlabHz(100, 5*1, 1, i*dTheta)
        slabHz.mySolve()
        exponent = complex(0, slabHz.k0*slabHz.L*cos(slabHz.theta))
        numerator = slabHz.phi[-1] - exp(exponent)
        denominator = exp(-exponent)
        hz100.append(abs(numerator/denominator))

    # Plot the results.
    degree = []
    for i in range(plotPoints):
        degree.append(i * 180*dTheta/pi)
        
    plot(degree, ezRef, label='Ez with an analytical method')
    plot(degree, ez50, label='Ez with 50 elements')
    plot(degree, ez100, label='Ez with 100 elements')
    plot(degree, hzRef, label='Hz with an analytical method')
    plot(degree, hz50, label='Hz with 50 elements')
    plot(degree, hz100, label='Hz with 100 elements')
    xlabel('degrees')
    ylabel('Reflection coefficient')
    title('Plane-wave reflection by a metal-backed dielectric slab')
    grid(True)
    legend()
    show()
    