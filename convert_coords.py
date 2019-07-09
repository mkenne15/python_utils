#!/usr/bin/env python

import numpy as np
from optparse import OptionParser
from astropy import units as u, constants as c
from astropy.coordinates import SkyCoord

desc="Convert Coords"""
parser=OptionParser(usage=" %prog options", description=desc)
parser.add_option('-i','--input',type='string',default="HourAngle",
                  help="Input coord type, HourAngle, Degree")
parser.add_option('-c','--coords',type='string',
                  help="Coords")

(options,args) = parser.parse_args()

if options.input == "HourAngle":
	src = SkyCoord(options.coords, unit=(u.hourangle, u.deg))
	print (src.to_string('decimal'))
elif options.input == "Degrees":
	src = SkyCoord(options.coords, unit=(u.deg, u.deg))
	print (src.to_string('hmsdms',sep=':'))
