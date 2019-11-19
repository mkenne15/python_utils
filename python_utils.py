import numpy as np
from astropy.stats import LombScargle
from astropy import constants as const
from astropy import units as u
import csv
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.ascii as Ascii
from os import chdir, listdir

def SDSS_flux_to_mag(flux,filt):
    '''
    Convert SDSS fluxes to magnitudes
    flux: flux of object, float
    filt: filter, string. Choices are ugriz
    '''
    SDSS_calibs = {'u': [3767, 1.4e-10], 'g': [3631, 0.9e-10], 'r': [3631, 1.2e-10], 'i': [3631, 1.8e-10], 'z': [3565, 7.4e-10]}#https://ned.ipac.caltech.edu/help/sdss/dr6/photometry.html#asinh
    f0 = SDSS_calibs[filt][0]
    b = SDSS_calibs[filt][1]
    mag = -2.5*(np.arcsinh(flux/(2*b*f0))+np.log(b))/np.log(10)
    return mag

def PS1_to_SDSS(PS1_mag,g,i,filt):
    '''
    Converst PS1 mags to SDSS mags assuming coefficents from
    https://iopscience.iop.org/article/10.3847/0004-637X/822/2/66 . Assumes
    star is main sequence star.
    PS1_mag - magnitude of obect in PS1
    g - magnitude of obect in PS1_g
    i - magnitude of obect in PS1_i
    filt - filter of PS1_mag
    '''
    filt_coeffs = {'g': [-0.01808, -0.13595, 0.01941, -0.00183], 'r': [-0.01836, -0.03577, 0.02612, -0.00558], 'i': [0.01170, -0.00400, 0.00066, -0.00058], 'z': [-0.01062, 0.07529, -0.03592, 0.00890], 'y': [0.08924, -0.20878, 0.10360, -0.02441]}#https://iopscience.iop.org/article/10.3847/0004-637X/822/2/66
    color = g-i
    coeffs = filt_coeffs[filt]
    SDSS_mag = PS1_mag - coeffs[0] - coeffs[1]*color - coeffs[2]*color**2 - coeffs[3]*color**3
    return SDSS_mag

def mass_func(a1,P):
    '''
    Calculates mass function assuming inclination of 90 degrees
    a1 should have unites of lts (semi-major axis of primary)
    P should have units of days
    '''
    return 4*np.pi**2*a1**3/(const.G*P**2)

def min_mass(a1, P, M1):
    '''
    Calculates minimum companion mass given M1 and
    assuming inclination of 90 degrees
    a1 should have unites of lts (semi-major axis of primary)
    P should have units of days
    M1 should have units of solar masses
    '''
    lts = const.c*u.s
    a1 = a1*lts
    P = P*u.day
    fm = 4*np.pi**2*a1**3/(const.G*P**2)
    min_M2 = (((M1)**2*fm)**(1/3)).to(u.solMass)
    return min_M2.value

def wave2doppler(w, w0):
    '''
    convert wavelength to velocity around w
    '''
    vel = ((w**2-w0**2) * 3e5/ (w**2+w0**2))
    return vel

def region_around_line(w, flux, cont, cut, switch="None"):
    '''cut out and normalize or continuum subtract (optional) flux around a line
    Originally from http://learn.astropy.org/rst-tutorials/UVES.html?highlight=filtertutorials,
    but I've done a lot of tweaking to make it more customisable.
    Parameters
    ----------
    w         : 1 dim np.ndarray
                array of wavelengths
    flux      : np.ndarray of shape (N, len(w))
                array of flux values for different spectra in the series
    cont      : [[min1,max1],[min2,max2]]
                Fits polynomial to region min1<w<max1 and min2<w<max2
    cut       : [min,max]
                Wavelength range to return
    switch    : None,Norm,Sub
                Returns the cut spectrum, cont normalised spectra, or cont subtracted spectra.
    '''
    #index is true in the region where we fit the polynomial
    indcont = ((w > cont[0][0]) & (w < cont[0][1])) | ((w > cont[1][0]) & (w < cont[1][1]))
    #index of the region we want to return
    indrange = (w > cut[0]) & (w < cut[1])
    # make a flux array of shape
    # (number of spectra, number of points in indrange)
    f = np.zeros((flux.shape[0], indrange.sum()))
    for i in range(flux.shape[0]):
        # fit polynomial of second order to the continuum region
        linecoeff = np.polyfit(w[indcont], flux[i, indcont],2)
        # divide the flux by the polynomial and put the result in our
        # new flux array
        if switch=="None":
            f[i,:] = flux[i,indrange]
        elif switch=="Norm":
            f[i,:] = flux[i,indrange]/np.polyval(linecoeff, w[indrange])
        elif switch=="Sub":
            f[i,:] = flux[i,indrange]-np.polyval(linecoeff, w[indrange])
    return w[indrange], f

def EW(wavelength, flux, line_range):
    """
    Calcultes EW of a line (after flux has been normalised). Esimates
    error on EW by assuming lower and upper bounds for continuum range
    have been provided.

    Parameters
    ----------
    wavelength  : Array of floats
    flux        : Array of floats
                  Normalised Flux
    line_range  : [[minwavelength1,maxwavelength1],[minwavelength2,maxwavelength2]]
                  Calculates EW between min1 and max1, and min2 and max2, and returns the average EW.
    """
    ew_upper_filter = np.where(np.logical_and(wavelength>line_range[0][0],wavelength<line_range[0][1]))
    ew_lower_filter = np.where(np.logical_and(wavelength>line_range[1][0],wavelength<line_range[1][1]))

    ew_line = []
    ew_line_e = []


    for i in range(0,len(flux)):

        ew_temp_upper = flux[i,ew_upper_filter][0] - 1.
        ew_temp_upper = ew_temp_upper[:-1] * np.diff(wavelength[ew_upper_filter])

        ew_temp_upper = ew_temp_upper.sum()

        ew_temp_lower = flux[i,ew_lower_filter][0] - 1.
        ew_temp_lower = ew_temp_lower[:-1] * np.diff(wavelength[ew_lower_filter])

        ew_temp_lower = ew_temp_lower.sum()

        ew_line.append((ew_temp_upper+ew_temp_lower)/2)
        ew_line_e.append((ew_temp_upper+ew_temp_lower)/2 - ew_temp_lower)


    return np.array(ew_line), np.array(ew_line_e)

class PowerSpec(object):
    '''
    Custom class for creating power spectra using astropy's PS function
    '''

    def __init__(self, t, f, ferr=[]):
        """
        Methods to create and plot power spectra

        Parameters
        ----------
        t : array
            array of times of observations
        f : array
            array of flux values
        ferr: array
            Optional array of flux errors
        norm : Boolean, default = True
            flag for if data should be normed before power spectra run.
            """

        self.t = t
        self.f = f
        self.ferr = ferr

        assert len(t)>0, 'Time array must have values'
        assert len(f)>0, 'Flux array must have values'


    def runspec(self, pmin = None, pmax = 40, delta_f=None, norm='standard', centre=True, oversampling=20):
        """
        Running Power Spec

        Parameters
        ----------
        pmax : float, defaut = 60
        maximum period to probe (in mins)
        """
        self.pmax = pmax/(24*60)
        fmin = 1 / self.pmax

        if pmin != None:
            self.pmin = pmin/(24*60)
            fmax = 1/self.pmin
        else:
            self.Ny_freq = (len(self.t)/(2*(self.t[-1]-self.t[0])))
            fmax = self.Ny_freq

        if delta_f == None:
            delta_f = 1/(oversampling*(self.t[-1]-self.t[0]))

        self.freq_range = np.arange(fmin, fmax, delta_f)

        if len(self.ferr) == 0:
            self.LS = LombScargle(self.t, self.f, normalization=norm, center_data=centre)
            self.PS = LombScargle(self.t, self.f, normalization=norm, center_data=centre).power(self.freq_range)
        else:
            self.LS = LombScargle(self.t, self.f, self.ferr, normalization=norm, center_data=centre)
            self.PS = LombScargle(self.t, self.f, self.ferr, normalization=norm, center_data=centre).power(self.freq_range)

        self.best_f = self.freq_range[(np.argmax(self.PS))]

def ephemeris(t, t0, P, folded=True, absolute=False):
    """
    Calculating Ephemris

    Parameters
    ----------
    t      : float, or array of floats
             Time of observations
    t0     : float
             Time of phase 0
    P      : float
             Period to fold on
    folded : Boolean, default=True
             Whether to phase-fold data or not
    """

    t_temp = np.array(t)
    t0 = t0.to(u.day).value
    P = P.to(u.day).value
    if folded:
        return (t_temp-t0)/(P) - (t_temp-t0)//(P)
    elif absolute:
        return (t_temp-t0)/(P)
    else:
        return (t_temp-t0)/(P) - (t_temp[0]-t0)//(P)

def plot_tomogram(dopmap_dir, dopmap_name, ax, colormap='magma', vmin=1, vmax=99, manual_vmax=None, log=False, r_label_angle=45, overlay_dir = "None", include_overlay = False):

	"""
    Methods to create and plot power spectra. Uses the output files from
    Enrico Kotze's DopTomag program. Adapted from a script by Colin Littlefield
    (Thanks Colin)

    Parameters
    ----------
    dopmapdir : 	str
        	    	top level dir of maps
    dopmap_name : 	str
        		name of actual dopmap (ends in 0 for normal, 1 for inverted)
    ax: 		matplotlib axis
        		axis to plot to
    cmap : 		string
        		color map to use
	vmin : 		float
			minium percentage for color map
	vmax : 		float
			max percentage for color map
	manual_vmax :	float
			maximum velocity shown in km/s
	log : 		boolean
			log the map
	r_label_angle : float
			At what angle should the radial labels be shown?
            """

	#######################################################
	#######################################################
	#######################################################



	# The tomography output files contains a number of blank
	# spaces that are read as '' characters with Python's
	# csv reader. This function eliminates those '' characters.

	# Read the file.
	chdir(dopmap_dir)
	dopmap, maxvel, proj, wl, gamma = ReadDopFile(dopmap_name)


	################################################################
	# Create a meshgrid for displaying the image.
	###############################################################

	# We start out in cartesian coordinates.
	x = np.arange(-dopmap.shape[0]/2, dopmap.shape[0]/2 + 1, 1)
	y = np.arange(-dopmap.shape[1]/2, dopmap.shape[1]/2 + 1, 1)

	x, y = np.meshgrid(x, y)

	# Convert to polar coordinates.
	theta, r = np.arctan2(y, y.T), np.hypot(y, y.T)



	###############################################################
	# Plotting the tomogram
	###############################################################

	if log:
	    mask = np.where(dopmap > 0)
	    dopmap[mask] = np.log10(dopmap[mask])

	vmax = np.percentile(dopmap, vmax)
	vmin = np.percentile(dopmap, vmin)

	ax.pcolormesh(theta, r, dopmap, cmap = colormap,
		      vmin = vmin, vmax = vmax, rasterized=True)


	##########################################
	# Indicate the wavelength
	##########################################
	#ax.annotate("%d $\mathregular{\AA}$"%wl,
	#	    xy = (0,0), xytext = (0.03, 0.85), color = 'k',
	#	    textcoords = 'figure fraction', fontsize = 26)





	##########################################
	# Set angle labels along the circumference
	##########################################

	# We want four angle labels along the
	# circumference of the polar plot at
	# 90-degree intervals.
	#
	# The bottom label will further specify
	# units (i.e., degrees and km/s).

	ax.set_thetagrids(np.arange(0, 271, 90),
		          ["0$^{\circ}$",
		           "90$^{\circ}$",
		           "180$^{\circ}$   ",
		           "\n\n270$^{\circ}$\n"+r"($v$, $\theta$) [km s$^{-1}$, deg.]"], color='k')

	ax.grid(color = 'g', alpha = 0.5, ls = ":")





	############################################
	# set radial labels
	############################################

	# Velocity labels for the radial direction.
	# Start at 500 km/s, and go to the maximum
	# velocity.
	vel = np.arange(500, maxvel, 500)

	# km/s per pixel in the Doppler map.
	vel_per_pix = maxvel/y.max()

	# Convert the velocities into pixels. It is
	# the radial distance (in pixels) that
	# corresponds to the velocities in the
	# variable "vel"
	vel2pix = vel/vel_per_pix

	# If it is an inside-out projection,
	# the labels need to be inverted.
	if proj == 1:
	    vel2pix = y.max()- vel2pix
	    vel2pix, vel = vel2pix[::-1], vel[::-1]


	# Don't place labels within the innermost
	# 2 pixels of the tomogram.
	vel = vel[vel2pix >= 2]
	vel2pix = vel2pix[vel2pix >= 2]


	# Finally, set the radial labels.
	ax.set_rgrids(vel2pix,[int(a) for a in vel], 20, fontsize=15, color = 'k')

	# Set the position angle for the radial labels.
	ax.set_rlabel_position(r_label_angle)






	#########################################################
	# Overlay velocity model.
	#########################################################

	# Optionally, we will iterate across several other output
	# files that provide velocity overlays and the binary
	# parameters used to generate them.

	if include_overlay:

		chdir(overlay_dir)

		# Filenames for (a) the velocity of the secondary's
		# Roche lobe, (b) the velocity of the ballistic stream,
		# and (3) the binary parameters.
		overlay_names = [ 'vSecondary.out', 'vStreamBal.out', 'pBinary']

	 	#Check to see if these three files are actually in the
		#directory. We will iterate across the ones that
		#are present.
		files_in_dir = [name for name in listdir() if name in overlay_names]


		for name in files_in_dir:

			# The binary-parameters file will be read differently
			# than the files with velocity overlays. We will extract
			# and plot the velocity information after converting from
			# km/s into pixel coordinates.
			if name != 'pBinary':

				F = Ascii.read(name)

		   		# Get the velocity overlay, and convert from velocity
		    		# into pixels.
				vel_overlay = 1000*F['col6']/vel_per_pix

		    		# Invert for inside-out tomograms.
				if proj == 1:
					vel_overlay = (y.max())-vel_overlay

		    		# Do not plot within the central two pixels (for clarity).
				vfilt = np.where(vel_overlay >= 2)

				ax.plot(F['col4'][vfilt], vel_overlay[vfilt],
					c = 'k', lw = 1.5, ls = '-')

			# Instead of plotting info from the binary-parameters file,
			# we want to annotate the plot with the WD mass, the inclination,
			# and the mass ratio.
			else:
				F = Ascii.read("pBinary", delimiter = '=', format = 'no_header')
				info = {'M1': "%.2f M$_{\odot}$"%(F['col2'][1]),
					'$i$': "%d$^{\circ}$"%(F['col2'][2]),
					'q': "%.3f"%(F['col2'][3])
				}

				ax.annotate("i: %s\nq: %s\nM1: %s\n$\gamma$=%d km s$^{-1}$"%
					(info['$i$'], info['q'], info['M1'], gamma),
				xy = (0,0), xytext = (-0.14, 0.0), color = 'k',
				textcoords = 'axes fraction', fontsize = 12)

	##################################################



	##################################
	# Set the maximum radial value.
	##################################
	if manual_vmax is None:
	    ax.set_rmax(dopmap.shape[0]/2)
	else:
	    ax.set_rmax(manual_vmax / vel_per_pix)

def ReadDopFile(filename, ):
	'''
	Load the tomography file, and iterate across each line
	with Python's csv reader.
	'''
	# A flag that will be triggered once the csv reader has
	# reached the part of the file with the tomography.
	dopflag = False

	# The actual Doppler map will be stored in this list.
	dopmap = []

	no_blanks = lambda a: [_ for _ in a if _ != '']

	# Read the datafile.
	with open(filename, 'r') as file:
		reader = csv.reader(file, delimiter = ' ')

		# Iterate across the rows in the file.
		for idx, row in enumerate(reader):

		    # Get the maximum velocity and the projection type
		    # from the first row.
			if idx == 0:
				# Remove the '' instances.
				row = no_blanks(row)
				# The number of pixels in one direction in the
				# tomogram.
				npix = int(row[2])
				# the maximum velocity in the tomogram (km/s)
				maxvel = float(row[0])/1000
				# What type of projection?
				# 0 = standard, 1 = inside-out
				proj = int(row[4])

		    # Get the wavelength and systemic velocity from the
		    # third row.
			if idx == 2:
				row = no_blanks(row)

				wl = np.rint(float(row[2])).astype(int)
				gamma = np.rint(float(row[1])).astype(int)/1000 # km/s

		    # As we iterate across the rows in the file, we will
		    # eventually reach the Doppler map. The order in which
		    # these next three if statements appear is inelegant,
		    # but it works. The variable 'dopflag' indicates whether
		    # the reader has reached the part of the file with the
		    # Doppler map.

		    # After the Doppler map ends, the output file contains
		    # the word "delta." Stop reading the file at that point.
			if "delta" in row:
				dopflag = False
				break

		    # Once we reach the Doppler map, begin saving the
		    # data.
			if dopflag:
				dopmap.append([float(a) for a in row if a != ''])

		    # This is the condition for ascertaining when the reader
		    # has reached the Doppler map. The next iteration will
		    # begin saving the data.
			if "dopmap" in row:
				dopflag = True
	del reader

	# Convert the Doppler map to an array, and reshape it into an
	# image.
	dopmap = np.array(dopmap[1:])
	dopmap = np.reshape(dopmap, (npix, npix))

	return dopmap, maxvel, proj, wl, gamma

def ReadXspec(filename, no_spectra, resids=False, components=False, no_components=1):
	'''
	Custom command for loading saved plotting files from Xspec. Assumes the files
    have been slightly edited.
	'''
	# A flag that will be triggered once the csv reader has
	# reached the part of the file with the tomography.

	# The actual Doppler map will be stored in this list.
	energies = []
	residuals = []

	no_blanks = lambda a: [_ for _ in a if _ != '']

	# Read the datafile.
	with open(filename, 'r') as file:
		reader = csv.reader(file, delimiter = ' ')
		energy = []
		energy_err = []
		flux = []
		flux_err = []
		model = []
		comps = [[] for i in range(no_components)]
		if components:
			# Iterate across the rows in the file.
			for idx, row in enumerate(reader):

				if "NO" in row[0]:
					energies.append([energy, energy_err, flux, flux_err, model, comps])
					energy = []
					energy_err = []
					flux = []
					flux_err = []
					model = []
					comps = [[] for i in range(no_components)]
				elif "NO" in row[-1]:
					row = no_blanks(row)
					energy.append(float(row[0]))
					energy_err.append(float(row[1]))
					flux.append(float(row[2]))
					flux_err.append(float(row[3]))
				else:
					row = no_blanks(row)
					energy.append(float(row[0]))
					energy_err.append(float(row[1]))
					flux.append(float(row[2]))
					flux_err.append(float(row[3]))
					model.append(float(row[4]))
					for jdx,k in enumerate(row[5:]):
						comps[jdx].append(float(k))
		else:
			for idx, row in enumerate(reader):
				if "NO" in row[0]:
					energies.append([energy,energy_err,flux,flux_err,model])
					energy = []
					energy_err = []
					flux = []
					flux_err = []
					model = []
				elif "NO" in row[-1]:
					row = no_blanks(row)
					energy.append(float(row[0]))
					energy_err.append(float(row[1]))
					flux.append(float(row[2]))
					flux_err.append(float(row[3]))
				else:
					row = no_blanks(row)
					energy.append(float(row[0]))
					energy_err.append(float(row[1]))
					flux.append(float(row[2]))
					flux_err.append(float(row[3]))
					model.append(float(row[4]))

	del reader

	return energies
