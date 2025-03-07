import numpy as np
import numpy.linalg as la
from PDEModel1DWithExactSoln import PDEModel1DWithExactSoln

class StateEquation(PDEModel1DWithExactSoln):
    """
    Exact solution of variable speed problem. Boundary conditions are inflow
    Q_{in}(t) = cos(t) and no reflection at x = L.
    """

    def __init__(self):
        super().__init__(1)
        """Initialize constants."""
        self.c0 = 1.0
        self.alpha = 0.25

    # ----------------------------------------------------------------------
    # Methods of PDEModel1D 

    def speed(self, x):
      """Variable speed."""

      return self.c0 * np.exp(self.alpha * x)
      
    def f(self, x):

        return (1 / (self.alpha * self.c0)) * (1 - np.exp(-self.alpha * x))
    
    def applyLeftBC(self, x, t, dx, dt, u):
        """Inflow at the left boundary"""

        left = np.cos(8 * np.pi * t)

        return left
    

    def applyRightBC(self, x, t, dx, dt, u):
        """Setup the right boundary conditons"""

        # Get u at the final penultimate point
        u_pen = u[:, -2]
        u_ult = u[:, -1]

        # Advance w2 right to left
        right = u_ult - self.speed(x) * (dt / dx) * (u_ult - u_pen)

        return right


    # --------------------------------------------------------------------
    # Function for PDEModel1DWithExactSoln
    def exact_solution(self, X, t):
        """
        Exact solution u(x, t) of our PDE 
        u_t + c(x) * u_x = 0
        """
        U = np.zeros_like(X)

        for ix, x in enumerate(X):
            
            U[ix] = np.cos(8 * np.pi * (t - self.f(x)))
        
        return U