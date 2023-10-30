"""Computes n-dimensional Gaussian integrals.

See https://en.wikipedia.org/wiki/Gaussian_integral for the definition of the
Gaussian integral. We follow the method of solution for 2 dimensions given in
https://en.wikipedia.org/wiki/Gaussian_integral#By_polar_coordinates, but
generalize to n dimensions.

An intermediate product of this module is a pair of functions that compute the
Jacobian matrix and determinant for the Cartesian-to-spherical change of
coordinates.

See https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates for the
definition of the spherical coordinate system.

Usage:
  import gaussian_integral as g

  i = g.GaussianIntegral(1)
  i.doit()  # Returns sqrt(pi).
  i.jacobian()  # Returns Matrix([
                # [cos(phi0), -r*sin(phi0)],
                # [sin(phi0),  r*cos(phi0)]]).
  i.jacobian_det()  # Returns r.

  i = g.GaussianIntegral(1)
  i.doit()  # Returns sqrt**(3/2).

  i = g.GaussianIntegral(3)
  i.doit()  # Returns pi**2.
"""

from sympy import *


class GaussianIntegral:
  """A Gaussian integral in n dimensions.

  Attributes:
      n: The number of dimensions.
      r: The symbol for radial dimension.
      phis: The symbols for the angular dimensions.

  Raises:
      ValueError: If the provided number of dimensions n is less than 1.
  """

  def __init__(self, n: int):
    if n < 1:
      raise ValueError('n must be at least 1.')
    self.n = n
    self.r = Symbol('r')
    if n == 1:
      self.phis = (symbols('phi0'),)
    else:
      self.phis = symbols('phi0:' + str(n-1))

  """Expresses Cartesian coordinates in terms of spherical coordinates.

  Returns:
      A 1 x n matrix of Cartesian coordinates expressed in terms of spherical
      coordinates. If n == 1, returns a 1 x 2 matrix.
  """
  def _cartesian_to_spherical(self) -> Matrix:
    x_coords = [self.r * cos(self.phis[0]), self.r * sin(self.phis[0])]
    for i in range(1, self.n - 1):
      old = x_coords[i]
      x_coords[i] = old * cos(self.phis[i])
      x_coords.append(old * sin(self.phis[i]))
    return Matrix(x_coords).T

  """Compute the Jacobian of the Cartesian-to-spherical change of coordinates.

  Returns:
      An n x n Jacobian matrix. If n == 1, returns a 2 x 2 matrix.
  """
  def jacobian(self) -> Matrix:
    syms = list(self.phis)
    syms.insert(0, self.r)  # Prepend r.
    x_coords = self._cartesian_to_spherical()
    return x_coords.jacobian(syms)

  """Computes the determinant of the change of coordinates Jacobian.

  Returns:
      The simplified determinant of the Jacobian.
  """
  def jacobian_det(self):
    return self.jacobian().det().simplify()

  """Evaluates the Gaussian integral.

  Returns:
      The value of the Gaussian integral.
  """
  def doit(self):
    limits_of_int = [(self.r, 0, oo), (self.phis[-1], 0, 2 * pi)]
    for phi in self.phis[:-1]:
      limits_of_int.append((phi, 0, pi))
    int = integrate(exp(-self.r**2) * self.jacobian_det(), *limits_of_int)
    if self.n == 1:
      return sqrt(int)
    return int