#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
#                                                                       #
# A python implementation of the Fortran program shown in section 4.3.5 #
# of Jin pp.109-112 and for the exercise 4.7 of Jin p.112.              #
#                                                                       #
# Kyungwon Chun (kwchun@gist.ac.kr)                                     #
#                                                                       #
#########################################################################


from math import sqrt
from numpy import ndindex, zeros, empty
from scipy.linalg import solve
from pprint import pprint


class Fem2d(object):
    def __init__(self, nn, ne, n1, n2):
        self.nn = nn # total number of nodes
        self.x = empty(self.nn, float)
        self.y = empty(self.nn, float)

        self.ne = ne # total number of elements
        self.alpha_x = empty(self.ne, float)
        self.alpha_y = empty(self.ne, float)
        self.beta = empty(self.ne, float)
        self.f = empty(self.ne, float)
        self.n = empty((self.ne, 3), int)

        self.n1 = n1 # number of nodes on Gamma_1
        self.p = empty(self.n1, float)
        self.nd = empty(self.n1, int)

        self.n2 = n2 # number of segments on Gamma_2
        self.gamma = empty(self.n2, float)
        self.q = empty(self.n2, float)
        self.ns = empty((self.n2, 2), int)
        
        # Initialize the matrix [K] and {b}.
        self.k = zeros((self.nn, self.nn), float)
        self.b = zeros(self.nn, float)

        self.phi = empty(self.nn, float)

    def show_input(self):
        print "x:", self.x
        print "y:", self.y
        print "alpha_x:", self.alpha_x
        print "alpha_y:", self.alpha_y
        print "beta:", self.beta
        print "f:", self.f
        print "n:", self.n
        print "p:", self.p
        print "nd:", self.nd
        print "gamma:", self.gamma
        print "q:", self.q
        print "ns:", self.ns
        
    def set_system(self):
        de = empty(3, float)
        ce = empty(3, float)

        be = empty(3, float)
        ke = empty((3, 3), float)

        # Start to assemble all area elements in omega.
        for e in xrange(self.ne):
            # Calculate b^e_i and c^e_i (i=0,1,2)
            i = self.n[e,0]
            j = self.n[e,1]
            m = self.n[e,2]
            de[0] = self.y[j] - self.y[m]
            de[1] = self.y[m] - self.y[i]
            de[2] = self.y[i] - self.y[j]
            ce[0] = self.x[m] - self.x[j]
            ce[1] = self.x[i] - self.x[m]
            ce[2] = self.x[j] - self.x[i]
            
            # Calculate delta^e
            deltae = 0.5 * (de[0] * ce[1] - de[1] * ce[0])
            
            # Generate the elemental matrix [K^e], {b^e}.
            for i in range(3):
                for j in range(3):
                    if i == j:
                        del_ij = 1
                    else:
                        del_ij = 0
                    ke[i,j] = \
                        (self.alpha_x[e] * de[i] * de[j] + 
                         self.alpha_y[e] * ce[i] * ce[j]) / (4 * deltae) +\
                         self.beta[e] * (1 + del_ij) * deltae / 12
                be[i] = deltae * self.f[e] / 3

            # Add [K^e] and {b^e} to [K] and {b}, respectively..
            for i in range(3):
                for j in range(3):
                    self.k[self.n[e,i], self.n[e,j]] += ke[i,j]
                self.b[self.n[e,i]] += be[i]

    def set_boundary(self):
        # Start to assemble all line segments on Gamma_2.
        ks = empty((2, 2), float)
        bs = empty(2, float)
        for s in xrange(self.n2):
            # Calculate the length of each segment.
            i = self.ns[s,0]
            j = self.ns[s,1]
            ls = sqrt((self.x[i] - self.x[j])**2 + 
                      (self.y[i] - self.y[j])**2)

            # Compute [k^s].
            ks[0,0] = self.gamma[s] * ls / 3
            ks[0,1] = self.gamma[s] * ls / 6
            ks[1,0] = ks[0,1]
            ks[1,1] = ks[0,0]

            # Compute {b^s}.
            bs[0] = 0.5 * self.q[s] * ls
            bs[1] = 0.5 * self.q[s] * ls

            # Add [K^s] and {b^s} to [K] and {b}, respectively.
            for i in range(2):
                for j in range(2):
                    self.k[self.ns[s,i], self.ns[s,j]] += ks[i,j]
                self.b[self.ns[s,i]] += bs[i]
        
        # Impose the Dirichlet boundary condition
        for i in xrange(self.n1):
            self.b[self.nd[i]] = self.p[i]
            self.k[self.nd[i], self.nd[i]] = 1
            for j in xrange(self.nn):
                if j == self.nd[i]: continue
                self.b[j] -= self.k[self.nd[i],j] * self.p[i]
                self.k[self.nd[i], j] = 0
                self.k[j, self.nd[i]] = 0

    def show_system(self):
        print "k:", self.k
        print "b:", self.b

    def solve(self):
        self.phi = solve(self.k, self.b)
        return self.phi


if __name__ == '__main__':
    """This example uses the system of exercise 4.5.

    """
    from numpy import array, ones, linspace
    from matplotlib.mlab import griddata
    import matplotlib.pyplot as plt

    nn = 9
    ne = 8
    n1 = 6
    n2 = 4

    ex4_5 = Fem2d(nn, ne, n1, n2)
    
    ex4_5.x = array([1.5, 0.0, 0.0, 1.5, 0.0, 1.5, 3.0, 3.0, 3.0])
    ex4_5.y = array([2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    
    ex4_5.n = array([[1, 2, 0],
                     [3, 0, 2],
                     [2, 4, 3],
                     [5, 3, 4],
                     [6, 7, 5],
                     [3, 5, 7],
                     [7, 8, 3],
                     [0, 3, 8]])
    ex4_5.alpha_x = ones(ne, float)
    ex4_5.alpha_y = ones(ne, float)
    ex4_5.beta = zeros(ne, float)
    ex4_5.f = ones(ne, float)
    
    ex4_5.nd = array([8, 0, 1, 4, 5, 6])
    ex4_5.p = zeros(n1, float)
    ex4_5.p[0], ex4_5.p[1], ex4_5.p[2] = 1, 1, 1

    ex4_5.ns = array([[1, 2],
                      [2, 4],
                      [8, 7],
                      [7, 6]])
    ex4_5.gamma = ones(n2, float)
    ex4_5.q = ones(n2, float)

    ex4_5.show_input()

    ex4_5.set_system()
    ex4_5.set_boundary()

    ex4_5.show_system()
    
    ex4_5.solve()
    print "phi:", ex4_5.solve()
    
    # define grid.
    xi = linspace(0,3,300)
    yi = linspace(0,2,200)
    zi = griddata(ex4_5.x,ex4_5.y,ex4_5.phi,xi,yi,interp='linear')
    ct = plt.contourf(xi,yi,zi,100,cmap=plt.cm.jet)
    plt.colorbar(ct)
    plt.xlim(0,3)
    plt.ylim(0,2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
