#!/usr/bin/env python

from __future__ import division

from numpy import *
from scipy import *
from matplotlib.pyplot import *


class Geometry(object):
    """
    The Geometry class contains information on the domain, geometry,
    regions, and materials in the problem.
    
    Attributes:
      width         = width of the problem domain
      height        = height of the problem domain
      regions       = list of <Region>s in the geometry
      regionsByName = dictionary with region names as keys and regions as values

    Methods:
      addRegion = add a region to the geometry
      endpoint  = for given r,phi find the points of intersection with boundary
      __init__  = initialize the geometry
      __iter__  = add ability to iterate over all stacks in geometry
    """

    def __init__(self, width, height, z):
        self.width = width
        self.height = height
        self.z_top = z
        self.stacks = []
        self.stacksByName = {}

    def addStack(self, stack):
        """Add a <Stack> object to the geometry, storing it in the
        stacks list and stacksByName dictionary.
        """

        self.stacks.append(stack)
        self.stacksByName[stack.name] = stack

    def endpoint(self,r,phi):
        """ For a given x,y coordinate and angle, determine the
        coordinates of the outgoing boundary point
        """

        w = self.width
        h = self.height
        x = r.x
        y = r.y
        m = tan(phi)

        # Determine points of intersection with boundaries
        points = [(0, y - x*m), (w, y + (w-x)*m),
                  (x - y/m, 0), (x + (h-y)/m, h)]

        # Get rid of trivial pair (x, y)
        points.remove((x,y))

        # Determine which point is on boundary
        for x,y in points:
            if x >= 0 and x <= w and y >= 0 and y <= h:
                return Vector2D(x,y)

    def __iter__(self):
        for stack in self.stacks:
            yield stack


class AxialStack(object):
    """
    Set of regions stacked on top of one another.

    Attributes:
      region = list of <Region>s
      cell   = <SurfaceNode> or <OperatorNode> representing a closed area

    Methods:
      __iter__ = iterate over regions in stack
    """

    def __init__(self, name):
        self.name = name
        self.regions = []
        self.cell = None

    def __iter__(self):
        for r in self.regions:
            yield r

    def __len__(self):
        return len(self.regions)

    def __repr__(self):
        return "<AxialStack: {0}>".format(self.name)


class Region(object):
    """
    Represents a single flat source region.

    Attributes:
      name        = name of the region
      volume      = actual volume of the region
      volume_calc = calculated volume based on segments within region
      flux        = scalar flux in region
      source      = source that each track picks up in this regino
      segments    = list of segments within region
      cell        = <SurfaceNode> or <OperatorNode> representing a closed volume
    """
    
    def __init__(self, name):
        self.name = name
        self.volume = 1.
        self.volume_calc = 0.
        self.weight = 0.
        self.flux = 0.
        self.source = 0.
        self.segments = []
        self.cell = None

    def __repr__(self):
        return "<Region: {0}>".format(self.name)


class Surface2D(object):
    """
    Defines a quadratic surface of the form:
    F(x,y) = Ax^2 + By^2 + Cxy + Dx + Ey + F

    Attributes:
      _A through _F = coefficients defined above

    Methods:
      sense         = evalute equation of the surface at a point
      positiveSense = boolean indicating if point is on positive side of surface
      distance      = for given positon/direction, find distance to surface
      intersectLine = find points of intersection of a line with surface
    """

    def __init__(self):
        self._A = 0
        self._B = 0
        self._C = 0
        self._D = 0
        self._E = 0
        self._F = 0

    def sense(self,r):
        """Evaluate F(x,y) at a given point"""

        x = r.x
        y = r.y
        return (self._A*x**2 + self._B*y**2 + self._C*x*y + 
                self._D*x + self._E*y + self._F)

    def distance(self, r, omega):
        """Solves the equation F(r0 + s*omega) = 0 for the point
        s. This is the point of intersection of the surface F(x,y) and
        the line r = r0 + s*Omega.
        """

        x = r.x      # x coordinate
        y = r.y      # y coordinate
        u = omega.x  # Cos of angle
        v = omega.y  # Sin of angle

    def positiveSense(self, r):
        """Determine if location r is 'above' or 'below' the surface
        by determining the sense"""

        return self.sense(r) > 0

    def intersectLine(self, r, phi):
        """ Determine the points of intersection of the surface with a
        line drawn defined from point r with angle phi. Effectively
        solves the equation F(x, y0+m*(x-x0)) = 0 for x.
        """

        # Reference point and angle for line
        x0 = r.x
        y0 = r.y
        m = tan(phi)

        # Center of circle
        A = self._A
        B = self._B
        C = self._C
        D = self._D
        E = self._E
        F = self._F

        # Put F(x,y) = 0 in form of ax^2 + bx + c = 0
        q = y0 - m*x0
        a = A + B*m**2 + C*m
        b = 2*B*m*q + C*q + D + E*m
        c = B*q**2 + E*q + F
            
        # Determine intersection
        quad = b**2 - 4*a*c
        if quad < 0:
            return []
        elif quad == 0:
            x = -b/(2*a)
            y = m*(x - x0) + y0
            return [Vector2D(x,y)]
        elif quad > 0:
            x1 = (-b + sqrt(quad))/(2*a)
            y1 = m*(x1 - x0) + y0
            x2 = (-b - sqrt(quad))/(2*a)
            y2 = m*(x2 - x0) + y0
            d1 = abs(y1-y0)
            d2 = abs(y2-y0)
            if d1 < d2:
                return [Vector2D(x1,y1), Vector2D(x2,y2)]
            else:
                return [Vector2D(x2,y2), Vector2D(x1,y1)]


class Plane(Surface2D):
    """
    Defines a plane of the form:
    F(x,y) = Ax + By + C = 0
    """

    def __init__(self, A, B, C):
        super(Plane, self).__init__()
        self._C = A
        self._D = B
        self._F = C

    def __repr__(self):
        return "<Plane: {0}x + {1}y + {2} = 0>".format(
            self._C, self._D, self._F)

    def draw(self):
        raise NotImplementedError
        

class Circle(Surface2D):
    """
    Defines a circle of the form:
    F(x,y) = (x-x0)^2 + (y-y0)^2 - R^2 = 0

    Takes as arguments 'c', a 2D vector specifying the center of the
    circle and a float R specifying the radius.
    """

    def __init__(self, x, y, R):
        self._A = 1
        self._B = 1
        self._C = 0
        self._D = -2*x
        self._E = -2*y
        self._F = x**2 + y**2 - R**2

    @property
    def x(self):
        return -0.5 * self._D

    @property
    def y(self):
        return -0.5 * self._E

    @property
    def R(self):
        return sqrt(self.x**2 + self.y**2 - self._F)

    def area(self):
        return pi*self.R**2

    def __repr__(self):
        return "<Circle: ({0},{1}) with R={2}>".format(
            self.x, self.y, self.R)

    def draw(self):
        cir = matplotlib.pyplot.Circle(
            (self.x, self.y), radius = self.R, fc = 'none', lw = 2)
        gca().add_patch(cir)

    def intersectLine(self, r, phi):
        """ Determine the points of intersection of the circle with a
        line drawn defined from point r with angle phi. Effectively
        solves the equation F(x, y0+m*(x-x0)) = 0 for x.
        """

        # Reference point and angle for line
        x0 = r.x
        y0 = r.y
        m = tan(phi)

        # Center of circle
        xc = self.x
        yc = self.y
        R  = self.R

        # Put F(x,y) = G(x,y) in form of ax^2 + 2kx + c = 0
        a = 1 + m**2
        k = -xc - m**2*x0 + m*(y0 - yc)
        c = xc**2 + (m*x0)**2 - 2*m*x0*(y0 - yc) + (y0 - yc)**2 - R**2
            
        # Determine intersection
        quad = k**2 - a*c
        if quad < 0:
            return []
        elif quad == 0:
            x = -k/a
            y = m*(x - x0) + y0
            return [Vector2D(x,y)]
        elif quad > 0:
            x1 = (-k + sqrt(quad))/a
            y1 = m*(x1 - x0) + y0
            x2 = (-k - sqrt(quad))/a
            y2 = m*(x2 - x0) + y0
            d1 = abs(y1-y0)
            d2 = abs(y2-y0)
            if d1 < d2:
                return [Vector2D(x1,y1), Vector2D(x2,y2)]
            else:
                return [Vector2D(x2,y2), Vector2D(x1,y1)]

class Box(object):
    
    def __init__(self, x_min, y_min, x_max, y_max):
        self.r_min = Vector2D(x_min, y_min)
        self.r_max = Vector2D(x_max, y_max)

    def intersectLine(self,r,phi):
        """ For a given x,y coordinate and angle, determine the
        coordinates of the outgoing boundary point
        """

        x_min = self.r_min.x
        y_min = self.r_min.y
        x_max = self.r_max.x
        y_max = self.r_max.y
        x = r.x
        y = r.y
        m = tan(phi)

        # Determine points of intersection with boundaries
        points = [(x_min, y + (x_min - x)*m),
                  (x_max, y + (x_max - x)*m),
                  (x + (y_min - y)/m, y_min),
                  (x + (y_max - y)/m, y_max)]

        # Get rid of trivial pair (x, y)
        points.remove((x,y))

        # Determine which point is on boundary
        for x,y in points:
            if x >= x_min and x <= x_max and y >= y_min and y <= ymax:
                return Vector2D(x,y)

    def positiveSense(self, location):
        if (location.x >= self.r_min.x and location.x <= self.r_max.x and
            location.y >= self.r_min.y and location.y <= self.r_max.y):
            return False
        else:
            return True

    def area(self):
        width = self.r_max.x - self.r_min.x
        height = self.r_max.y - self.r_min.y
        return width * height

    def __repr__(self):
        return "<Box: ({0.x},{0.y}) --> ({1.x},{1.y})>".format(
            self.r_min, self.r_max)


class BoxNode(object):

    def __init__(self, box=None, positiveSense=None):
        self.surface = box
        self.positiveSense = positiveSense

    def contains(self, location):
        return self.surface.positiveSense(location) == self.positiveSense

    def distance(self, r, phi):
        """ For a given x,y coordinate and angle, determine the
        coordinates of the outgoing boundary point
        """

        surf = self.surface
        x_min = surf.r_min.x
        y_min = surf.r_min.y
        x_max = surf.r_max.x
        y_max = surf.r_max.y
        x = r.x
        y = r.y
        m = tan(phi)

        # Determine points of intersection with boundaries
        points = [(x_max, y + (x_max - x)*m),
                  (x_min, y + (x_min - x)*m),
                  (x + (y_max - y)/m, y_max),
                  (x + (y_min - y)/m, y_min)]

        # Remove trivial pair (x,y)
        for x_p, y_p in points:
            if abs(x_p - x) < 1e-7 and abs(y_p - y) < 1e-7:
                points.remove((x_p,y_p))
                break

        # Determine which point is on boundary
        for x,y in points:
            if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
                return r.distance(Vector2D(x,y))
        
    def __repr__(self):
        return "<SurfaceNode: {0}{1}>".format(
            "+" if self.positiveSense else "-", self.surface)

    def __iter__(self):
        yield self
    


class Vector2D(object):
    """
    Represents two-dimensional coordinates such as position,
    direction, etc

    Attributes:
      x = abscissa
      y = ordinate

    Methods:
      __init__ = initialize object
      __repr__ = text representation of point
      distance = determine distance to a given point
      closeTo  = determine if this point is close to a given point
    """

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "<Vector2D: ({0},{1})>".format(self.x,self.y)

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def distance(self, vec):
        return sqrt((vec.x - self.x)**2 + (vec.y - self.y)**2)

    def closeTo(self, vec, eps = 1e-5):
        return self.distance(vec) < eps

class SurfaceNode(object):

    def __init__(self, surface=None, positiveSense=None):
        self.surface = surface
        self.positiveSense = positiveSense

    def contains(self, location):
        return self.surface.positiveSense(location) == self.positiveSense

    def __repr__(self):
        return "<SurfaceNode: {0}{1}>".format(
            "+" if self.positiveSense else "-", self.surface)

    def __iter__(self):
        yield self

class OperatorNode(object):

    def __init__(self, leftNode=None, rightNode=None, operator=None):
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.operator = operator

    def __iter__(self):
        # Loop through surfaces in left node
        if type(self.leftNode) is OperatorNode:
            for surfaceNode in self.leftNode:
                yield surfaceNode
        else:
            yield self.leftNode

        # Loop through surfaces in right node
        if type(self.rightNode) is OperatorNode:
            for surfaceNode in self.rightNode:
                yield surfaceNode
        else:
            yield self.rightNode

    def contains(self, location):
        return self.operator.evaluate(self.leftNode, self.rightNode, location)
            
class Union(object):

    def evaluate(self, left, right, location):
        return left.contains(location) or right.contains(location)
        
class Intersection(object):

    def evaluate(self, left, right, location):
        return left.contains(location) and right.contains(location)
