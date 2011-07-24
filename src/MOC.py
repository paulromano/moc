#!/usr/bin/env python

"""
Method of Characteristics Code
Written by Paul Romano for MIT 22.212

The workhorse of this MOC code is the MOC class itself. All the
representation of the geometry is handled external to this module in
the geometry.py file.
"""


from __future__ import division, print_function

from numpy import *
from scipy import *
from matplotlib.pyplot import *

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
        self.delta = 0.05
        self.boundary = 'periodic'
        self.geometry = geometry

        self.polarquad = Quadrature()
        self.polarquad.TabuchiYamamoto(3)

    def makeTracks(self):
        """Generate tracks based on the specified number of azimuthal
        angle, subdivisions, and track spacing.
        """

        # Geometry preliminaries
        w = self.geometry.width            # Width of 2D domain
        h = self.geometry.height           # Height of 2D domain
        if self.boundary == 'reflective':  # Azimuthal angle subdivision, e.g. div=2 is (0,pi)
            self.div = 2                        
        elif self.boundary == 'periodic' or self.boundary == 'white':
            self.div = 4
        nangle = self.nangle//self.div     # Num of azimuthal angles followed

        #Determine azimuthal angles
        intv = vectorize(int)
        p = 2*pi/self.nangle*(0.5+arange(nangle))      # Desired angles
        self.nx = intv(abs(w/self.delta*sin(p)) + 1)   # Num of intersections along x-axis
        self.ny = intv(abs(h/self.delta*cos(p)) + 1)   # Num of intersections along y-axis
        self.nt = self.nx + self.ny                    # Total num of tracks for each angle
        self.phi = arctan(h*self.nx/(w*self.ny))       # Actual angle
        for i, angle in enumerate(self.phi):
            if p[i] > pi/2:
                self.phi[i] = pi - self.phi[i]         # Fix angles in (pi/2, pi)

        # Determine track spacing
        xprime = w/self.nx                             # Spacing between points along x-axis
        yprime = h/self.ny                             # Spacing between points along y-axis
        self.dels = xprime*sin(self.phi)               # Actual track spacing

        # Determine azimuthal weights
        temp1 = concatenate((self.phi,[2*pi/self.div]+self.phi[0]))
        temp2 = concatenate((-self.phi[0:1],self.phi))
        x1 = 0.5*(temp1[1:]-temp1[:-1])
        x2 = 0.5*(temp2[1:]-temp2[:-1])
        self.wgta = (x1 + x2)/(2*pi)*self.dels*self.div

        # Determine coordinates
        self.tracks = []
        for i in range(nangle):
            xin = zeros(self.nt[i])
            yin = zeros(self.nt[i])
            
            phi = self.phi[i]
            xin[:self.nx[i]] = xprime[i]*(0.5 + arange(self.nx[i]))
            yin[:self.nx[i]] = 0
            yin[self.nx[i]:] = yprime[i]*(0.5 + arange(self.ny[i]))
            if sin(phi) > 0 and cos(phi) > 0:
                xin[self.nx[i]:] = 0
            elif sin(phi) > 0 and cos(phi) < 0:
                xin[self.nx[i]:] = w
            self.tracks.append([])
            for x,y in zip(xin,yin):
                r_in = geom.Vector2D(x,y)
                r_out = self.geometry.endpoint(r_in,phi)
                newTrack = Track2D(r_in, r_out, phi)
                newTrack.weight = self.wgta[i]          # Could make index on track to get rid of this
                self.tracks[i].append(newTrack)

    def makeReflective(self):
        """Determine the outgoing and incoming tracks for each track
        for the purpose of treating reflective boundary
        conditions. Also, we need to determine whether the flux going
        out of one track will go into the beginning or the end of the
        outgoing track.

        track            <---- current track
        track.track_out  <---- outgoing track
        track.track_in   <---- incoming track
        track.refl_in    <---- outgoing flux (when tracking in reverse) 
                               at start (0)/end (1) of incoming track
        track.refl_out   <---- outgoing flux at start/end of outgoing track
        """

        nangle = self.nangle//2

        # Determine reflected tracks -- we only need to loop over half
        # the angles because we are going to set pointers for
        # connecting tracks as well
        for i in range(nangle//2):
            nx = self.nx[i]
            ny = self.ny[i]
            nt = self.nt[i]
            tracks_refl = self.tracks[-(i+1)]

            for j,track in enumerate(self.tracks[i]):
                # More points along y-axis
                if nx <= ny:
                    if j < nx:
                        # Bottom to right side
                        track.track_in          = tracks_refl[j]
                        track.track_in.track_in = track
                        track.refl_in           = 0
                        track.track_in.refl_in  = 0

                        track.track_out          = tracks_refl[2*nx-1-j]
                        track.track_out.track_in = track
                        track.refl_out           = 0
                        track.track_out.refl_in  = 1
                    elif j < ny:
                        # Left side to right side
                        track.track_in           = tracks_refl[j-nx]
                        track.track_in.track_out = track
                        track.refl_in            = 1
                        track.track_in.refl_out  = 0

                        track.track_out          = tracks_refl[j+nx]
                        track.track_out.track_in = track
                        track.refl_out           = 0
                        track.track_out.refl_in  = 1
                    else:
                        # Left side to top side
                        track.track_in           = tracks_refl[j-nx]
                        track.track_in.track_out = track
                        track.refl_in            = 1
                        track.track_in.refl_out  = 0

                        track.track_out           = tracks_refl[-(nx-(nt-j)+1)]
                        track.track_out.track_out = track
                        track.refl_out            = 1
                        track.track_out.refl_out  = 1

                # More points along x-axis
                else:
                    if j < nx - ny:
                        # Bottom to top
                        track.track_in          = tracks_refl[j]
                        track.track_in.track_in = track
                        track.refl_in           = 0
                        track.track_in.refl_in  = 0

                        track.track_out           = tracks_refl[nt-(nx-ny)+j]
                        track.track_out.track_out = track
                        track.refl_out            = 1
                        track.track_out.refl_out  = 1
                    elif j < nx:
                        # Bottom to right side
                        track.track_in          = tracks_refl[j]
                        track.track_in.track_in = track
                        track.refl_in           = 0
                        track.track_in.refl_in  = 0

                        track.track_out          = tracks_refl[nx+(nx-j)-1]
                        track.track_out.track_in = track
                        track.refl_out           = 0
                        track.track_out.refl_in  = 1
                    else:
                        # Left side to top
                        track.track_in           = tracks_refl[j-nx]
                        track.track_in.track_out = track
                        track.refl_in            = 1
                        track.track_in.refl_out  = 0

                        track.track_out           = tracks_refl[ny+(nt-j)-1]
                        track.track_out.track_out = track
                        track.refl_out            = 1
                        track.track_out.refl_out  = 1

    def makePeriodic(self):
        """Determine the outgoing track for each track for the purpose
        of treating periodic (or white) boundary conditions.

        track            <---- current track
        track.track_out  <---- outgoing track
        """

        nangle = self.nangle//4

        # Determine reflected tracks
        for i in range(nangle):
            nx = self.nx[i]
            ny = self.ny[i]
            nt = self.nt[i]
            tracks = self.tracks[i]

            for j,track in enumerate(tracks):
                # Always pass flux to beginning of next track
                track.refl_out = 0
                
                # More points along y-axis
                if nx <= ny:
                    if j < nx:
                        # Bottom to right side
                        track.track_out = tracks[2*nx-1-j]
                    elif j < ny:
                        # Left side to right side
                        track.track_out = tracks[j+nx]
                    else:
                        # Left side to top side
                        track.track_out = tracks[nt-1-j]

                # More points along x-axis
                else:
                    if j < nx - ny:
                        # Bottom to top
                        track.track_out = tracks[ny+j]
                    elif j < nx:
                        # Bottom to right side
                        track.track_out = tracks[nx+(nx-j)-1]
                    else:
                        # Left side to top
                        track.track_out = tracks[nt-1-j]

    def makeSegments(self):
        """Generate a segment for each part of each track that goes
        through a different flat source region. This method is not
        completely general right now and takes advantage of the fact
        that I know I only have two regions
        """

        fuel = self.findRegion('fuel')
        water = self.findRegion('water')
        circle = fuel.cell.surface

        for r in self.geometry:
            r.volume_calc = 0
            r.segments = []

        nangle = self.nangle//self.div
        for i in range(nangle):
            phi = self.phi[i]
            delta = self.dels[i]
            for track in self.tracks[i]:
                phi = track.phi
                r_in = track.r_in
                r_out = track.r_out
                intersect = circle.intersectLine(r_in,phi)
                intersect.insert(0, r_in)
                intersect.append(r_out)
                self.pairs = zip(intersect[:-1], intersect[1:])
                for r0, r1 in self.pairs:
                    if (r0 != r_in and r1 != r_out):
                        s = Segment(fuel, r0, r1)
                    else:
                        s = Segment(water, r0, r1)
                    track.segments.append(s)
                    s.region.segments.append(s)

                    # Calculate volume based on tracks
                    s.region.volume_calc += s.length*track.weight

        # Correct segment lengths based on tracking error
        for r in self.geometry:
            r.factor = r.volume / r.volume_calc
            for s in r.segments:
                s.length *= r.factor

    def solve(self):
        """Solve the backwards characteristic form of the transport
        equation over the domain with a fixed source (assumed to be
        flat in each region). We start by assuming that the fluxes on
        all the boundaries are zero and then iterate until they
        converge."""

        fuel = self.findRegion('fuel')
        water = self.findRegion('water')

        # Precompute exponential factor and set incoming fluxes to zero
        for track in self:
            track.flux_in = zeros((2,3))
            for s in track:
                sigma = s.region.sigma_t
                sintheta = self.polarquad.sin
                s.factor = 1.0 - exp(-sigma*s.length/sintheta)

        # Iterate on incoming fluxes on domain boundary
        max_iterations = 100
        for i_boundary in range(max_iterations):
            #print("Inner iteration {0}".format(i_boundary))

            # Initialize volume flux to zero
            for r in self.geometry:
                r.flux = 0

            # Loop over polar angles
            for j, (sintheta, wgtp) in enumerate(self.polarquad):

                # Loop over each track
                for track in self:
                    # Need to divide total weight for each track by 2
                    # since we are tracking forwards and backwards
                    weight = 4*pi*track.weight*wgtp*sintheta

                    # Loop over each segment in track
                    flux = track.flux_in[0,j]
                    if i_boundary == 0:
                        flux = 0
                    for s in track:
                        r = s.region
                        delta = (flux - r.source/r.sigma_t)*s.factor[j]
                        r.flux += delta*weight
                        flux -= delta

                    # Transfer flux to outgoing track
                    track.track_out.flux_in[track.refl_out,j] = flux

                    # Now traverse this track in reverse for reflective
                    if self.boundary == "reflective":
                        flux = track.flux_in[1,j]
                        if i_boundary == 0:
                            flux = 0
                        for s in track.segments[::-1]:
                            r = s.region
                            delta = (flux - r.source/r.sigma_t)*s.factor[j]
                            r.flux += delta*weight
                            flux -= delta

                        # Transfer flux to incoming track
                        track.track_in.flux_in[track.refl_in,j] = flux

            if self.boundary == 'white':
                # Find average angular flux on boundary
                totalFlux = 0
                for track in self:
                        totalFlux += sum(track.flux_in[0,:])
                ntrack = len([t for t in self])*len(self.polarquad)
                averageFlux = totalFlux/ntrack
                        
                # Set boundary fluxes to average flux
                for track in self:
                    track.flux_in[0,:] = averageFlux

            # Add in source and normalize to volume
            for r in self.geometry:
                if self.boundary == "reflective":
                    r.flux /= 2
                r.flux = 4*pi*r.source/r.sigma_t + r.flux/(r.sigma_t*r.volume)

            # Save source in fuel for Dancoff factor
            if i_boundary == 0:
                fuel.flux0 = fuel.flux

            # Check if converged
            fluxv = array([r.flux for r in self.geometry])
            if i_boundary > 0:
                if all(abs((fluxv - fluxvold)/fluxvold) < 1e-10):
                    return
            fluxvold = fluxv

    def solveKeff(self, max_outer = 1000):
        """Using the fixed source kernel from above, iterate on the
        scattering and fission sources until they converge (outer
        iterations)."""

        # Initial guess
        S_old = {}
        S = {}
        for r in self.geometry:
            S_old[r] = 1
        k_old = 1.0

        # Loop over outer iterations
        for i_outer in range(max_outer):
            # Inner iteration
            for r in self.geometry:
                scatter = r.sigma_s*r.flux/(4*pi)
                r.source = 1/(4*pi*k_old)*S_old[r] + scatter
            self.solve()

            # Outer iteration
            for r in self.geometry:
                S[r] = r.nusigma_f*r.flux
            keff = k_old*sum(S.values())/sum(S_old.values())

            print("i = {0}, keff = {1}".format(i_outer, keff))
            if abs(keff - k_old) < 1e-7:
                break

            # Update source
            k_old = keff
            for r in self.geometry:
                S_old[r] = S[r]

    def setup(self, output=True):
        """Generate tracks, segments, and link tracks together for
        treating boundary conditions"""

        # Determine which routine to use for boundary conditions
        if self.boundary == 'reflective':
            makeBCs = self.makeReflective
        elif self.boundary == 'periodic' or self.boundary == 'white':
            makeBCs = self.makePeriodic

        # Generate tracks, BCs, segments
        if output:
            self.execute(self.makeTracks, 'Generating tracks...')
            self.execute(makeBCs, 'Generating boundary conditions...')
            self.execute(self.makeSegments, 'Generating segments...')
        else:
            self.makeTracks()
            makeBCs()
            self.makeSegments()

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

        for track in self:
            for s in track:
                if s.region == self.findRegion('fuel'):
                    s.plot()
                else:
                    s.plot('b-')

    def findRegion(self, name):
        """Find a region by its name"""

        return self.geometry.regionsByName[name]
    
    def __iter__(self):
        for i in range(len(self.phi)):
            for track in self.tracks[i]:
                yield track


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
    
    def __init__(self, r_in, r_out, phi):
        """Create a 2D track within problem domain"""

        self.r_in      = r_in  # Incoming coordinates
        self.r_out     = r_out # Outgoing coordinates
        self.track_in  = None  # Incoming track for boundary conditions
        self.track_out = None  # Outgoing track for boundary conditions
        self.segments  = []    # Segments within track

        # Could get rid of these and replace with index since the
        # weight is linked to angle
        self.phi      = phi   # Angle
        self.weight   = 0     # Weight of track

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
            self.r_in, self.r_out, self.phi)
        
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

    def __init__(self, region, r0, r1):
        self.region = region
        self.r0 = r0
        self.r1 = r1
        self.length = self.r0.distance(self.r1)

    def plot(self, linespec = 'k-'):
        """Draw the segment with optionally specified linespec"""

        plot([self.r0.x, self.r1.x],
             [self.r0.y, self.r1.y],
             linespec)

    def __repr__(self):
        return "<Segment: {0}>".format(self.length)

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


################################################################################
######################### Subroutines for HW problems ##########################
################################################################################

def Problem1Part1():
    """Create table of fuel/coolant flux ratio as a function of ray
    spacing and number of azimuthal angles for fuel xs = 1.0 and
    coolant xs = 0.1
    """

    g = makeGeometry()
    m = MOC(g)

    # Adjust fuel cross-sections
    fuel = m.findRegion('fuel')
    fuel.sigma_t = 1
    fuel.source = 1./(4*pi)

    # Adjust moderator cross-sections
    water = m.findRegion('water')
    water.sigma_t = 0.1
    water.source = 0.

    for bc in ['periodic', 'white']:
        m.boundary == bc
        for nangle in [16,32,64,128,256,512]:
            for spacing in [0.10, 0.05, 0.02, 0.01]:
                m.nangle = nangle
                m.delta = spacing
                m.setup(output=False)
                m.solve()
                print("{0} {1} {2} {3}".format(bc, nangle, spacing,
                                               fuel.flux/water.flux))

def Problem1Part2():
    """Make six tables of Dancoff factor (fuel xs = infinity) and
    coolant xs = 0, 1, and infinity as a function of ray spacing and
    number of azimuthal angles
    """

    g = makeGeometry()
    m = MOC(g)

    # Adjust fuel cross-sections
    fuel = m.findRegion('fuel')
    fuel.sigma_t = 1e5
    fuel.source = 1./(4*pi)

    # Adjust moderator cross-sections
    water = m.findRegion('water')
    water.source = 0.

    for bc in ['periodic', 'white']:
        m.boundary = bc
        for sigma_t in [1e-5, 1, 1e5]:
            water.sigma_t = sigma_t
            for nangle in [16,32,64,128,256,512]:
                m.nangle = nangle
                for spacing in [0.10, 0.05, 0.02, 0.01]:
                    m.delta = spacing
                    m.setup(output=False)
                    m.solve()
                    dancoff = (1 - (fuel.source/fuel.sigma_t*4*pi - fuel.flux)/
                               (fuel.source/fuel.sigma_t*4*pi - fuel.flux0))
                    print("{0} {1} {2} {3} {4}".format(
                            bc, sigma_t, nangle, spacing, dancoff))

def Problem2():
    """Using the code from problem 1 and the 1-group xs data of HW2,
    write an eigenvalue solver for the pin-cell geometry with
    reflective boundary conditions and compare your results to the
    ones obtained with your CPM code.
    """

    g = makeGeometry()
    m = MOC(g)
    m.boundary = 'reflective'
    m.nangle = 128

    # Adjust fuel cross-sections
    fuel = m.findRegion('fuel')
    fuel.sigma_t = 4.52648699E-01
    fuel.sigma_s = 3.83259177E-01
    fuel.nusigma_f = 9.94076580E-02

    # Adjust moderator cross-sections
    water = m.findRegion('water')
    water.sigma_t = 8.41545641E-01
    water.sigma_s = 8.37794542E-01
    water.nusigma_f = 0.

    # Solve eigenvalue problem
    m.setup(output=False)
    m.solveKeff()

def makeGeometry():
    pitch = 1.27
    circle = geom.Circle(pitch/2, pitch/2, 0.4)
    fuelCell = geom.SurfaceNode(circle, False)
    waterCell = geom.SurfaceNode(circle, True)

    fuel = geom.Region('fuel')
    fuel.cell = fuelCell
    fuel.volume = circle.area()
    
    water = geom.Region('water')
    water.cell = waterCell
    water.volume = pitch**2 - circle.area()

    g = geom.Geometry(pitch, pitch)
    g.addRegion(fuel)
    g.addRegion(water)
    return g
