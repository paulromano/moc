#!/usr/bin/env python

"""
3D Method of Characteristics Code
Written by Paul Romano for MIT 22.212

The workhorse of this MOC code is the MOC class itself. All the
representation of the geometry is handled external to this module in
the geometry.py file.
"""


from __future__ import division, print_function

from numpy import *
from scipy import *
from matplotlib.pyplot import *
from random import random
from mpl_toolkits.mplot3d import Axes3D

import geometry as geom

class MOC(object):
    """
    Main method of characteristics class which can solve fixed source
    and eigenvalue problems.

    Attributes:
      nangle    = number of azimuthal angles in (0,2*pi)
      delta     = desired track spacing
      boundary  = boundary condition (e.g. 'reflective')
      geometry  = associated <Geometry> object
      polarquad = associated <Quadrature> object
      tracks    = list of tracks indexed by angle

    Physics Methods:
      makeTracks     = generate cyclic tracks over a rectangular region
      makeReflective = create references from track to track for reflective BCs
      makePeriodic   = create references from track to track for periodic BCs
      makeSegments   = divide tracks into segments based on geometry
      solve          = solve a fixed source problem
      solveKeff      = solve an eigenvalue problem

    Other Methods:
      __init__       = initialize an <MOC> object
      __iter__       = ability to loop over all tracks
      setup          = generate tracks, BCs, and segments all at once
      execute        = run a method and display OK if it didn't fail
      drawSegments   = use matplotlib to draw all segments
      findRegion     = find a region by name
    """

    def __init__(self, geometry = None):
        """Create a <MOC> object with the given geometry and set
        default parameters for number of azimuthal angles, track spacing, 
        """

        self.nangle = 16
        self.delta = 0.02
        self.boundary = 'vacuum'
        self.geometry = geometry

        self.polarquad = Quadrature()
        # self.polarquad.TabuchiYamamoto(3)
        self.polarquad.GaussLegendre(10)

    def makeTracks(self):
        """Generate tracks based on the specified number of azimuthal
        angle, subdivisions, and track spacing.
        """

        # Geometry preliminaries
        w = self.geometry.width            # Width of 2D domain
        h = self.geometry.height           # Height of 2D domain
        self.div = 2                       # Azimuthal angle subdivision, e.g. div=2 is (0,pi)
        nangle = self.nangle//self.div     # Num of azimuthal angles followed

        self.angles = [Angle() for i in range(nangle)]

        for i, a in enumerate(self.angles):
            p = 2*pi/self.nangle*(0.5+i)              # Desired angles
            a.nx = int(abs(w/self.delta*sin(p)) + 1)  # Num of intersections along x-axis
            a.ny = int(abs(h/self.delta*cos(p)) + 1)  # Num of intersections along y-axis
            a.phi = arctan(h*a.nx/(w*a.ny))           # Actual angle
            if p > pi/2:
                a.phi = pi - a.phi         # Fix angles in (pi/2, pi)

            a.xprime = w/a.nx              # Spacing between points along x-axis
            a.yprime = h/a.ny              # Spacing between points along y-axis
            a.delta = a.xprime*sin(a.phi)  # Actual track spacing

        # Determine azimuthal weight
        for i, a in enumerate(self.angles):
            if i+1 < nangle:
                x1 = 0.5*(self.angles[i+1].phi - a.phi)
            else:
                x1 = 2*pi/self.div - a.phi
            if i > 0:
                x2 = 0.5*(a.phi - self.angles[i-1].phi)
            else:
                x2 = a.phi
            a.weight = (x1 + x2)/(2*pi)*a.delta**2*self.div
                
        # Determine coordinates
        self.tracks = []
        for a in self.angles:
            xin = zeros(a.nx + a.ny)
            yin = zeros(a.nx + a.ny)
            
            xin[:a.nx] = a.xprime*(0.5 + arange(a.nx))
            yin[:a.nx] = 0
            yin[a.nx:] = a.yprime*(0.5 + arange(a.ny))
            if sin(a.phi) > 0 and cos(a.phi) > 0:
                xin[a.nx:] = 0
            elif sin(a.phi) > 0 and cos(a.phi) < 0:
                xin[a.nx:] = w
            self.tracks.append([])
            for x,y in zip(xin,yin):
                r_in = geom.Vector2D(x,y)
                r_out = self.geometry.endpoint(r_in,a.phi)
                newTrack = Track2D(r_in, r_out, a)
                self.tracks[-1].append(newTrack)


    def makeSegments(self):
        """Generate a segment for each part of each track that goes
        through a different flat source region. This method is not
        completely general right now and takes advantage of the fact
        that I know I only have two regions
        """

        g = self.geometry

        for stack in self.geometry:
            for r in stack:
                r.volume_calc = 0
                r.segments = []

        for track in self:
            r = track.r_in
            phi = track.angle.phi
            if phi > pi/2:
                sgn = -1
            else:
                sgn = 1

            while not r.closeTo(track.r_out):
                # Figure out what axial stack we're in
                foundNode = False
                for stack in g:
                    node = stack.cell
                    r_test = r + geom.Vector2D(sgn*1e-6,1e-6)
                    if node.contains(r_test):
                        foundNode = True
                        break
                if not foundNode:
                    raise
                
                d = node.distance(r,phi)
                r_new = r + geom.Vector2D(d*cos(phi),d*sin(phi))
                s = Segment(stack, r, r_new)
                track.segments.append(s)
                r = r_new

    def solve(self):
        """Solve the backwards characteristic form of the transport
        equation over the domain with a fixed source (assumed to be
        flat in each region). We start by assuming that the fluxes on
        all the boundaries are zero and then iterate until they
        converge."""

        sintheta = self.polarquad.sin
        costheta = self.polarquad.cos
        wgtp = self.polarquad.weight
        z_array = self.geometry.z_bottom

        # Initialize volume flux to zero
        for stack in self.geometry:
            for r in stack:
                r.flux = 0
                r.volume_calc = 0

        # Loop over each track
        for track in self:
            #print(track)

            lengths = cumsum([s.length for s in track])

            # Loop over polar angles
            for j in range(len(self.polarquad)):

                weight = 4*pi*track.angle.weight*wgtp[j]*sintheta[j]/2

                lprime = lengths[-1] - track.angle.delta/costheta[j]/2
                z = 0

                # Loop over axially-stacked rays
                while z < self.geometry.z_top:

                    reg_list = []

                    k_start = min(where(lengths > lprime)[0])
                    flux = 0

                    # Loop over each segment in track
                    for i, s in enumerate(track.segments[k_start:]):
                        if i == 0:
                            length = min(lengths[k_start] - lprime, s.length)
                        else:
                            length = s.length
                        z_new = z + length*costheta[j]/sintheta[j]

                        i_lowz = max(where(z >= z_array)[0])
                        i_highz = max(where(z_new >= z_array)[0])

                        for i in range(i_lowz, i_highz + 1):
                            if i+1 == len(z_array):
                                length = (min(self.geometry.z_top,z_new) - z)/costheta[j]
                            else:
                                length = (min(z_array[i+1],z_new) - z)/costheta[j]

                            r = s.stack.regions[i]
                            sigma = r.sigma_t
                            factor = 1.0 - exp(-sigma*length/sintheta[j])
                            delta = (flux - r.source/sigma)*factor
                            r.flux += delta*weight
                            r.volume_calc += length*track.angle.weight*wgtp[j]
                            flux -= delta

                            # Store region pointer and attenuation
                            # factor for backwards track
                            reg_list.append((r, factor))

                            z += length*costheta[j]
                        
                    lprime -= track.angle.delta/costheta[j]
                    if lprime > 0:
                        z = 0
                    else:
                        z = -lprime*costheta[j]/sintheta[j]

                    # Now traverse track in reverse
                    flux = 0
                    for r, factor in reg_list[::-1]:
                        delta = (flux - r.source/r.sigma_t)*factor
                        r.flux += delta*weight
                        flux -= delta
                    

        # Add in source and normalize to volume
        for stack in self.geometry:
            for r in stack:
                r.flux = 4*pi*r.source/r.sigma_t + r.flux/(r.sigma_t*r.volume)

    def solveSource(self, max_outer = 1000):
        """Using the fixed source kernel from above, iterate on the
        scattering sources until they converge (outer iterations)."""

        # Populate regions list
        regions = []
        for stack in self.geometry:
            for r in stack:
                regions.append(r)
        rSource = self.geometry.stacks[0].regions[0]
        rMiddle = self.geometry.stacks[4].regions[1]
        # rSource = rMiddle

        # Initial guess
        S_old = {}
        S = {}
        for r in regions:
            S_old[r] = 0
        flux_old = 0

        # Loop over outer iterations
        for i_outer in range(max_outer):
            # Inner iteration
            for r in regions:
                r.source = S_old[r]/(4*pi)
            rSource.source += 1/(4*pi)
            self.solve()

            # Outer iteration
            for r in regions:
                S[r] = r.sigma_s*r.flux

            flux = rMiddle.flux
            print("i = {0}, flux = {1}".format(i_outer, flux))
            if abs(flux - flux_old) < 1e-7:
                break

            # Update source
            flux_old = flux
            for r in regions:
                S_old[r] = S[r]

    def setup(self, output=True):
        """Generate tracks, segments, and link tracks together for
        treating boundary conditions"""

        # Generate tracks, BCs, segments
        if output:
            self.execute(self.makeTracks, 'Generating tracks...')
            self.execute(self.makeSegments, 'Generating segments...')
            #self.execute(self.makeLengths, 'Normalizing track lengths...')
        else:
            self.makeTracks()
            self.makeSegments()
            #self.makeLengths()

    def execute(self, func, output):
        """Run a subroutine with the specified output and indicate
        whether the subroutine ran to completion (OK) or failed
        (FAILED).
        """

        print(output, end='')
        try:
            func()
        except:
            print('\x1b[40G[ \x1b[31mFAILED\x1b[0m ]')
            raise
        print('\x1b[40G[   \x1b[32mOK\x1b[0m   ]')

    def drawSegments(self):
        """Draw all the segments over the domain. If the segment is in
        the fuel, it is represented by a black line whereas segments
        in the moderator are represented by blue lines.
        """

        colordict = {}
        for track in self:
            for s in track:
                try:
                    p = s.plot()
                    p[0].set_color(colordict[s.stack])
                except:
                    colordict[s.stack] = [random(), random(), random()]
                    p[0].set_color(colordict[s.stack])
        show()

    def draw3D(self):
        colordict = {}
        fig = figure()
        ax = Axes3D(fig)

        sintheta = self.polarquad.sin
        costheta = self.polarquad.cos
        z_array = self.geometry.z_bottom

        # Loop over each track
        for track in self:
            print(track)

            lengths = cumsum([s.length for s in track])

            # Loop over polar angles
            for j in range(len(self.polarquad)):
                
                lprime = lengths[-1] - track.angle.delta/costheta[j]/2
                z = 0

                while z < self.geometry.z_top:

                    k_start = min(where(lengths > lprime)[0])

                    x = track.r_in.x
                    y = track.r_in.y
                    phi = track.angle.phi
                    if lprime > 0:
                        r_start = (x + lprime*cos(phi), y + lprime*sin(phi), z)
                    else:
                        r_start = (x,y,z)
                    
                    # Loop over each segment in track
                    for i, s in enumerate(track.segments[k_start:]):
                        if i == 0:
                            length = min(lengths[k_start] - lprime, s.length)
                        else:
                            length = s.length
                        z_new = z + length*costheta[j]/sintheta[j]

                        i_lowz = max(where(z >= z_array)[0])
                        i_highz = max(where(z_new >= z_array)[0])
                        stack = s.stack

                        for i in range(i_lowz, i_highz + 1):
                            if i+1 == len(z_array):
                                length = (min(self.geometry.z_top,z_new) - z)/costheta[j]
                            else:
                                length = (min(z_array[i+1],z_new) - z)/costheta[j]
                            r_end = (r_start[0] + length*cos(phi)*sintheta[j],
                                     r_start[1] + length*sin(phi)*sintheta[j],
                                     r_start[2] + length*costheta[j])
                            try:
                                p = ax.plot(array([r_start[0], r_end[0]]),
                                            array([r_start[1], r_end[1]]),
                                            array([r_start[2], r_end[2]]))
                                p[0].set_color(colordict[s.stack,i])
                            except:
                                colordict[s.stack,i] = [random(), random(), random()]
                                p[0].set_color(colordict[s.stack,i])
                            r_start = r_end
                            z += length*costheta[j]
                        
                    lprime -= track.angle.delta/costheta[j]
                    if lprime > 0:
                        z = 0
                    else:
                        z = -lprime*costheta[j]/sintheta[j]
                            

    def __iter__(self):
        for i in range(len(self.angles)):
            for track in self.tracks[i]:
                yield track

    def writeFlux(self):
        for stack in self.geometry:
            for r in stack:
                print("{0}: {1}".format(r, r.flux))

    def eval1(self):
        rC = self.geometry.stacks[4].regions[1]

        phiV = 0
        V = 0
        for stack in self.geometry:
            for r in stack:
                if r is not rC:
                    phiV += r.flux*r.volume
                    V += r.volume
        phi1 = phiV/V
        print("Flux in 1a = {0}".format(phi1))
        print("Flux in 1b = {0}".format(rC.flux))
                


class Track2D(object):
    """
    A single characteristic track drawn through the geometry. Each
    track is made up of segments in each flat source region.
    
    Attributes:
      r_in      = <Vector2D> object representing the starting coordinates
      r_out     = <Vector2D> object representing the ending coordinates
      track_in  = <Track2D> object that connects at end of this track
      track_out = <Track2D> object that connects at beginning of this track
      segments  = a list of all the segments in this track
      phi       = azimuthal angle
      weight    = weight for this track

    Methods:
      plot     = draw the track based on r_in and r_out
      __iter__ = enable ability to iterate over segments in a track
      __repr__ = text representation of track
    """
    
    def __init__(self, r_in, r_out, angle):
        """Create a 2D track within problem domain"""

        self.r_in      = r_in  # Incoming coordinates
        self.r_out     = r_out # Outgoing coordinates
        self.angle     = angle # Angle
        self.track_in  = None  # Incoming track for boundary conditions
        self.track_out = None  # Outgoing track for boundary conditions
        self.segments  = []    # Segments within track

    def plot(self, linespec = 'k-'):
        """Draw the track with optionally specified linespec"""

        plot([self.r_in.x, self.r_out.x],
             [self.r_in.y, self.r_out.y],
             linespec)

    def __iter__(self):
        for segment in self.segments:
            yield segment

    def __repr__(self):
        return "<Track2D: ({0.x},{0.y}) --> ({1.x},{1.y}), phi = {2}>".format(
            self.r_in, self.r_out, self.angle.phi)

    def __len__(self):
        return len(self.segments)

        
class Segment(object):
    """
    A single segment over one flat source region. Each track has a
    list of these segments.

    Attributes:
      length = length of the track in centimeters
      region = reference to the flat source <Region> object
      r0     = <Vector2D> object of starting coordinates
      r1     = <Vector2D> object of ending coordinates

    Methods:
      plot     = draw the segment
      __init__ = initialize a segment
      __repr__ = text representation of segment
    """

    def __init__(self, stack, r0, r1):
        self.stack = stack
        self.r0 = r0
        self.r1 = r1
        self.length = self.r0.distance(self.r1)

    def plot(self, linespec = 'k-'):
        """Draw the segment with optionally specified linespec"""

        return plot([self.r0.x, self.r1.x],
                    [self.r0.y, self.r1.y],
                    linespec)

    def __repr__(self):
        return "<Segment: {0}>".format(self.length)


class Angle(object):
    """
    One set of azimuthal angles
    """

    def __init__(self):
        self.phi = None
        self.delta = None
        self.nx = None
        self.ny = None
        self.weight = None

class Quadrature(object):
    """
    Polar quadrature used for method of characteristics. Right now the
    only options are to use the Leonard optimized quadrature of the
    Tabuchi-Yamamoto quadrature.

    Attributes:
      sin    = sin of the polar angle (measured from z-axis)
      weight = corresponding polar weight

    Methods:
      TabuchiYamamoto = set quadrature to a TY quadrature 
      Leonard         = set quadrature to Leonard optimized quadrature
      __init__        = initialize quadrature
      __iter__        = enable ability to iterate over pairs of (sin, weight)
    """

    def __init__(self):
        self.sin = []
        self.weight = []

    def __iter__(self):
        for i in range(len(self.sin)):
            yield (self.sin[i], self.weight[i])

    def __len__(self):
        return len(self.sin)

    def TabuchiYamamoto(self, n):
        """
        Set the polar angles and weights according to the
        Tabuchi-Yamamoto polar quadrature derived in A. Yamamoto et
        al., "Derivation of Optimum Polar Angle Quadrature Set for the
        Method of Characteristics Based on Approximation Error for the
        Bickley Function," J. Nucl. Sci. Tech., 44 (2), pp. 129-136
        (2007).
        """

        if n == 1:
            self.sin = array([0.798184])
            self.weight = array([1.0])
        elif n == 2:
            self.sin = array([0.363900, 0.899900])
            self.weight = array([0.212854, 0.787146])
        elif n == 3:
            self.sin = array([0.166648, 0.537707, 0.932954])
            self.weight = array([0.046233, 0.283619, 0.670148])
        self.cos = sqrt(1 - self.sin**2)


    def Leonard(self, n):
        """
        Set the polar angles and weights according to the Leonord
        Optimized polar quadrature from A. Leonard and C. McDaniel,
        "Optimal Polar Angles and Weight for the Characteristics
        Method," Trans. Am. Nucl. Soc., 73, pp. 172-173 (1995) as well
        as R. Le Tellier and A. Hebert, "Anisotropy and Particle
        Conservation for Trajectory-Based Deterministic Methods,"
        Nucl. Sci. Eng., 158, pp. 28-39 (2008).
        """

        if n == 2:
            self.sin = array([0.273658, 0.865714])
            self.weight = array([0.139473, 0.860527])
        elif n == 3:
            self.sin = array([0.099812, 0.395534, 0.891439])
            self.weight = array([0.017620, 0.188561, 0.793819])
        self.cos = sqrt(1 - self.sin**2)

    def GaussLegendre(self, n):
        """
        Set the polar angles and weights according to a Gauss-Legendre
        quadrature with the polar cosine mapped onto [0,1)
        """

        if n == 2:
            self.sin = array([0,0])
            self.weight = array([0,0])
        elif n == 3:
            self.sin = array([0,0,0])
            self.weight = array([0,0,0])
        elif n == 5:
            self.sin = array([0.302687295, 0.638966388, 0.866025404,
                              0.973009432, 0.998899116])
            self.weight = array([0.118463443, 0.239314335, 0.284444444,
                                 0.118463443, 0.239314335])
        elif n == 10:
            self.sin = array([0.999914887721197, 0.997721417153943,
                              0.987069118033507, 0.959030659130099,
                              0.904928879685587, 0.818548678031661,
                              0.697383976746143, 0.543043161706711,
                              0.361088160369026, 0.161007000373654])
            self.weight = array([0.0333356721543441, 0.0747256745752903, 
                                 0.109543181257991, 0.134633359654998,
                                 0.147762112357376, 0.147762112357376,
                                 0.134633359654998, 0.109543181257991,
                                 0.0747256745752903, 0.0333356721543441])
        self.cos = sqrt(1 - self.sin**2)


################################################################################
######################### Subroutines for NEA Benchmark ########################
################################################################################


def makeGeometry():
    gamma = 0.5
    L = 1.0

    box = geom.Box
    box11 = box(0,0,                    (1-gamma)/2,(1-gamma)/2)
    box12 = box(0,(1-gamma)/2,          (1-gamma)/2,(1+gamma)/2)
    box13 = box(0,(1+gamma)/2,          (1-gamma)/2,1)
    box21 = box((1-gamma)/2,0,          (1+gamma)/2,(1-gamma)/2)
    box22 = box((1-gamma)/2,(1-gamma)/2,(1+gamma)/2,(1+gamma)/2)
    box23 = box((1-gamma)/2,(1+gamma)/2,(1+gamma)/2,1)
    box31 = box((1+gamma)/2,0,          1,(1-gamma)/2)
    box32 = box((1+gamma)/2,(1-gamma)/2,1,(1+gamma)/2)
    box33 = box((1+gamma)/2,(1+gamma)/2,1,1)

    boxNode11 = geom.BoxNode(box11, False)
    boxNode12 = geom.BoxNode(box12, False)
    boxNode13 = geom.BoxNode(box13, False)
    boxNode21 = geom.BoxNode(box21, False)
    boxNode22 = geom.BoxNode(box22, False)
    boxNode23 = geom.BoxNode(box23, False)
    boxNode31 = geom.BoxNode(box31, False)
    boxNode32 = geom.BoxNode(box32, False)
    boxNode33 = geom.BoxNode(box33, False)

    stack11 = geom.AxialStack('stack11')
    stack11.cell = boxNode11
    stack11.regions = [makeReg1('111'),makeReg1('112'),makeReg1('113')]
    stack12 = geom.AxialStack('stack12')
    stack12.cell = boxNode12
    stack12.regions = [makeReg1('121'),makeReg1('122'),makeReg1('123')]
    stack13 = geom.AxialStack('stack13')
    stack13.cell = boxNode13
    stack13.regions = [makeReg1('131'),makeReg1('132'),makeReg1('133')]
    stack21 = geom.AxialStack('stack21')
    stack21.cell = boxNode21
    stack21.regions = [makeReg1('211'),makeReg1('212'),makeReg1('213')]
    stack22 = geom.AxialStack('stack22')
    stack22.cell = boxNode22
    stack22.regions = [makeReg1('221'),makeReg2('222'),makeReg1('223')]
    stack22.regions[1].source = 1/(4*pi)
    stack23 = geom.AxialStack('stack23')
    stack23.cell = boxNode23
    stack23.regions = [makeReg1('231'),makeReg1('232'),makeReg1('233')]
    stack31 = geom.AxialStack('stack31')
    stack31.cell = boxNode31
    stack31.regions = [makeReg1('311'),makeReg1('312'),makeReg1('313')]
    stack32 = geom.AxialStack('stack32')
    stack32.cell = boxNode32
    stack32.regions = [makeReg1('321'),makeReg1('322'),makeReg1('323')]
    stack33 = geom.AxialStack('stack33')
    stack33.cell = boxNode33
    stack33.regions = [makeReg1('331'),makeReg1('332'),makeReg1('333')]

    g = geom.Geometry(1,1,L)
    g.addStack(stack11)
    g.addStack(stack12)
    g.addStack(stack13)
    g.addStack(stack21)
    g.addStack(stack22)
    g.addStack(stack23)
    g.addStack(stack31)
    g.addStack(stack32)
    g.addStack(stack33)

    g.z_bottom = array([0, L*(1-gamma)/2, L*(1+gamma)/2])

    # Calculate true volume of each region
    for stack in g:
        area = stack.cell.surface.area()
        stack.regions[0].volume = area*L*(1-gamma)/2
        stack.regions[1].volume = area*L*gamma
        stack.regions[2].volume = area*L*(1-gamma)/2
        
    return g

def makeReg1(name):
    r = geom.Region(name)
    c = 0.5
    r.sigma_t = 0.1
    r.sigma_s = r.sigma_t*c
    return r

def makeReg2(name):
    r = geom.Region(name)
    c = 0.5
    r.sigma_t = 0.1
    r.sigma_s = r.sigma_t*c
    return r
