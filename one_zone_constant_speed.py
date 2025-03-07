import numpy as np
import numpy.linalg as la
from PDEModel1DWithExactSoln import PDEModel1DWithExactSoln

class OneZoneConSpeed(PDEModel1DWithExactSoln):
    """
    Exact solution of variable speed problem. Boundary conditions are inflow
    Q_{in}cos(Omega t) and terminal resistance R.
    """

    def __init__(self):
        super().__init__(2)
        """Initialize constants."""
        self.c0 = 1.0
        self.gamma = 1.25 * np.log(2)
        self.a = 0.1
        self.b = 0.9
        self.L = 1.0
        self.A_r = 0.5
        self.rho = 1.0
        self.R = 1.5
        self.Q_in = 1.0
        self.Omega = 8.0 * np.pi
        self.get_coefficients()

    # ----------------------------------------------------------------------
    # Methods of PDEModel1D 

    def speed(self):
      """Constant speed throughout."""

      return self.c0
      
    def jacobian(self, x):
        """Compute the Jacobian at a point x"""
        c = self.speed()

        return np.array([[0, 2 * (c ** 2)], [0.5, 0]])
    
    def jacEigensystem(self, x):
        c = self.speed()
        V = np.array([[2 * c, -2 * c], [1, 1]])
        lam = np.array([c, -c])
        return (lam, V)

    def jacEigenvalues(self, x):
        c = self.speed()
        lam = np.array([c, -c])

        return lam

    def applyLeftBC(self, x, t, dx, dt, u):
        
        # Get current solution at points indexed 0 and 1
        uAt0 = u[:,0] 
        uAt1 = u[:,1]

        # Get the eigenvalues and eigenvectors 
        lam, V = self.jacEigensystem(x)

        # Solve the equation V*w = u for the Riemann variables at x0 and x1
        wAt0 = la.solve(V, uAt0)
        wAt1 = la.solve(V, uAt1) 

        w2At0 = wAt0[1]
        w2At1 = wAt1[1]
        
        # With inflow BCs, the first Riemann variable is specified. The second
        # Riemann variable is advected backwards
        w1 = self.Q_in * np.cos(self.Omega * t)
        
        w2 = w2At0 - lam[1] * (dt/dx) * (w2At1 - w2At0)

        uNextAt0 = np.matmul(V, np.array([w1,w2]))

        return uNextAt0
    

    def applyRightBC(self, x, t, dx, dt, u):
        """Setup the right boundary conditons"""

        c = self.speed()
        R = self.R

        # Get u at the final and penultimate point
        u_pen = u[:, -2]
        u_ult = u[:, -1]

        # Get the eigenvalues and eigenvectors 
        lam, V = self.jacEigensystem(x)

        # Solve the equation V*w = u for the Riemann variables at x0 and x1
        w_pen = la.solve(V, u_pen)
        w_ult = la.solve(V, u_ult) 

        w1_pen = w_pen[0]
        w1_ult = w_ult[0]

        # Advect w1 left to right
        w1 = w1_ult - lam[0] * (dt / dx) * (w1_ult - w1_pen)

        # Solve the equation p = R q for w2
        w2 = ((2 * c - R) / (2 * c + R)) * w1
        #print('w2=',w2)

        wNext = np.array([w1, w2])

        uNext = np.matmul(V, wNext)

        return uNext


    # --------------------------------------------------------------------
    # Method for PDEModel1DWithExactSoln
    def exact_solution(self, X, t):
        """
        Exact solution p(x, t) and q(x, t) of our PDE system
        p_t + (rho * c^2)/A_r * q_x = 0
        q_t + (A_r / rho) * p_x = 0
        """
        i = 1.0j

        d2 = np.exp(i * self.Omega * t)
        d3 = (self.A_r) / (self.rho * self.Omega)

        P = np.copy(X)
        Q = np.copy(X)
        for ix, x in enumerate(X):
            
            (psi1, psi2, dpsi1, dpsi2) = self.one_zone(x)
            A = self.C[0]
            B = self.C[1]

            P[ix] = 2.0 * np.real((A * psi1 + B * psi2) * d2) 
            Q[ix] = 2.0 * np.real(i * d3 * (A * dpsi1 + B * dpsi2) * d2)
   
        # stack P,Q into U
        U = np.vstack((P, Q))
        
        return U
        


    # ------------------------------------------------------------------------
    # Internal methods specific to this model. The user code should never
    # need to call these.

    def one_zone(self, x):
        """Generate the one_zone basis functions."""
        i = 1.0j

        psi1 = np.exp((i * self.Omega * x) / self.speed())
        psi2 = np.exp((-i * self.Omega * x)/ self.speed())
                      
        dpsi1 = (i * self.Omega / self.speed()) * np.exp((i * self.Omega * x) / self.speed())
        dpsi2 = (-i * self.Omega / self.speed()) * np.exp((-i * self.Omega * x) / self.speed())

        return (psi1, psi2, dpsi1, dpsi2)
    
    def left_BC(self):
        """Inflow on the left boundary."""

        in_flow = np.complex128(0.5 * self.speed() * self.rho / self.A_r)

        left = np.array([in_flow, 0.0])

        return left
    
    def right_BC(self):
        """Terminal resistance on the right boundary."""

        i = 1.0j

        d = (i * self.A_r * self.R) / (self.rho * self.Omega)

        psi1 = np.exp((i * self.Omega * self.L) / self.speed())
        psi2 = np.exp((-i * self.Omega * self.L)/ self.speed())
                      
        dpsi1 = (i * self.Omega / self.speed()) * np.exp((i * self.Omega * self.L) / self.speed())
        dpsi2 = (-i * self.Omega / self.speed()) * np.exp((-i * self.Omega * self.L) / self.speed())

        sr1 = psi1 - d * dpsi1
        sr2 = psi2 - d * dpsi2

        S_R = np.array([sr1, sr2])

        return S_R
    
    def get_coefficients(self):
        """Solves for the coefficients vector C."""

        M = np.array([[0, 1], self.right_BC()])
        self.C = np.linalg.solve(M, self.left_BC())
