import os

os.system('ls *.csv > csv_files.txt')

#import Sphidef as s
import csv
import sys
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from math import sqrt, log10, log, exp
from scipy import stats
from pylab import *


##### insert values: #####
t_ramp 		= array([2.0, 2.0, 5.0, 5.0, 10.0, 10.0, 25.0, 25.0, 50.0, 50.0, 100.0,100.0, 250.0, 250.0, 1000.0, 1000.0])*1.0e-6
amp_DDS		= 0.12 # read of amplitude of DDS signal
amp_beat	= 0.12 

show_fits 	= True

keywords = ['DATA', 'Firmware Version','second']
##########################

""" read in all files from commando line: ls *.csv>csv_files.txt """
cs = open('csv_files.txt', 'r')
cs.seek(0)
leser = csv.reader(cs)

files = []
dummy = []

i = 0
for row in leser:
	files.append(row[0])
	i += 1

def load_arrays(datafile): 
	print "--- load --- "+datafile
	# initialize file and reader
	verts = []
	if type(datafile) == str:
		f = open(datafile, 'r')
		f.seek(0)
		reader = csv.reader(f)
		for row in reader:
			verts.append(row)
	else:
		print('no')
	i=0
	# skip all header lines until keyword 'DATA'
	while verts[i][0] not in keywords:
		i += 1
	i_key = i
	i += 2
	# 5 possible columns
	x_values = []
	y_values = []
	z_values = []
	a_values = []
	b_values = []
	while i < len(verts):
		# check where is the first column filled with data
		first_column = 0
		if (verts[i][first_column] == ''):
			while (verts[i][first_column] == ''):
				first_column += 1
		# load x and y values
		# prevent to have only one data point and empty y values
		if (verts[i][first_column+1] != ''):
			x_values.append(float(verts[i][first_column]))
		if verts[i][first_column+1] != '':
			y_values.append(float(verts[i][first_column+1]))
		# check if there is a third column filled with data
		if ((first_column+2) <= (len(verts[i])-1)):
			if (verts[i][first_column+2] != ''):
				z_values.append(float(verts[i][first_column+2]))
				# check if there is a fourth column filled with data
		if ((first_column+3) <= (len(verts[i])-1)):
			if (verts[i][first_column+2] != ''):
				a_values.append(float(verts[i][first_column+3]))
		# check if there is a fifth column filled with data
		if ((first_column+4) <= (len(verts[i])-1)):
			if (verts[i][first_column+2] != ''):
				b_values.append(float(verts[i][first_column+4]))
		i += 1
	return x_values, y_values, z_values, a_values, b_values

def show_n():
	n = 0
	for element in files:
		print "index ",n," belongs to file ",element
		n += 1


x = [0]*(len(files)+1)
y = [0]*(len(files)+1)
z = [0]*(len(files)+1)
a = [0]*(len(files)+1)
b = [0]*(len(files)+1)

def emp():
	fig = plt.figure()
	fig.patch.set_facecolor('white')
	ax	= fig.add_subplot(111)
	ax.grid(True)
	colormap = plt.cm.gist_ncar
	num_plots = 9
	return ax, fig

def empb():
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	ax	= fig.add_subplot(111, axisbg = 'black')
	for spine in ax.spines.values():
		spine.set_edgecolor('white')
	ax.tick_params(colors='white')
	ax.grid(True, color = 'white')
	colormap = plt.cm.gist_ncar
	num_plots = 9
	return ax, fig

def leg():
	leg = plt.legend(loc=0, ncol=1, shadow=True, fancybox=True, numpoints=1)
	leg.get_frame().set_alpha(0.75)
	return leg


def load_all():
	n = 0
	for element in files:
		x[n], y[n], z[n], a[n], b[n] = load_arrays(element)
		n += 1

load_all()
show_n()

from scipy import optimize
from scipy.optimize import curve_fit
import numpy as np
from numpy import dot, sin, log10, pi, array, average, exp
from pylab import *

def sign_change(y, x1):
	value = False
	if ((y[x1] >= 0) and (average(y[(x1+1):(x1+5)]) <= 0)) : value = True
	if ((y[x1] <= 0) and (average(y[(x1+1):(x1+5)]) >= 0)) : value = True
	return value


def count_osc(y, i):
	check = False
	while check == False:
		if i >= len(y)-40: check = True
		else: check = sign_change(y, i)
		i += 1
	return i

def first_max(x, y, thresh, jump):
	i = 0
	up = []
	while (i<=jump):
		up.append(0)
		i += 1
	while (y[i] < thresh):
		up.append(0)
		i += 1
	while (y[i] >= thresh):
		up.append(y[i])
		if i > (len(y) - 5):
			y[i+1] = 0
		i += 1
	first_max = list(up).index(max(up))
	jump_new = len(up)
	return first_max, jump_new

def single_fits(x, z, amp_beat):
	jump = 0
	ind = 0
	maxima = []
	while (ind < (len(x) - 100)):
		ind, jump = first_max(x, z, 1/2.*amp_beat, jump) 
		maxima.append(ind)
		plot(x[ind], z[ind], 'o', color = colormap(0.3))
	frequencies = []
	times		= []
	for ind in arange(0, len(maxima)-1):
		x_fit = array(x[maxima[ind]:maxima[ind+1]])
		z_fit = array(z[maxima[ind]:maxima[ind+1]])
		p0 = [amp_beat, (x_fit[0]-x_fit[-1]), 90.0, 0.0]
		fitme = lambda x_fit, amp, period, phase, off: amp*np.sin(1/period*2*pi*x_fit+phase)+off
		try: 
			popt, pcov 	= curve_fit(fitme, x_fit, z_fit, p0)
		except:
			popt, pcov	= p0, None
		if pcov != None:
			frequencies.append(abs(1/popt[1]))
			times.append(average(x_fit))
			plot(x_fit, fitme(x_fit, popt[0], popt[1], popt[2], popt[3]), color = 'yellow')
	return times, frequencies


for index in arange(0,len(t_ramp)):

	if show_fits == True:
		figure()
		colormap = plt.cm.gist_ncar

		DDS = subplot(211)
		DDS.plot(x[index], y[index], label = 'DDS', color = colormap(0.75))
		single_fits(x[index], y[index], amp_DDS)

		beat = subplot(212)
		beat.plot(x[index], z[index], label = 'beat signal', color = colormap(0.3))
		single_fits(x[index], z[index], amp_beat)
		plt.savefig('show_fits'+files[index]+'.png')

	times_DDS, frequencies_DDS = single_fits(x[index], y[index], amp_DDS)
	times_beat, frequencies_beat = single_fits(x[index], z[index], amp_beat)

	fig = figure()
	fig.patch.set_facecolor('white')
	fig.figsize=(100,600)

	p = subplot(311)
	colormap = plt.cm.gist_ncar
	num_plots = 10
	p.set_ylim(0, 60e3)
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
	plot(times_DDS, frequencies_DDS , 'o', label = 'DDS frequencies', color = colormap(0.75)) 
	plot(times_beat , frequencies_beat , 'o', label = 'beat sig. frequencies', color = colormap(0.3)) 
	#leg()
	subplot(311).set_title('ramp: '+str(t_ramp[index])+'sec.', fontsize = 16)
	subplot(311).set_ylabel('frequency (Hz)', fontsize = 16)
	subplot(312)
	plot(x[index], y[index], color = colormap(0.75), label = 'DDS')
	subplot(312).set_ylabel('amplitude (arb.)', fontsize = 16)
	leg()

	subplot(313)
	plot(x[index], z[index], color = colormap(0.3), label = 'beat signal')
	subplot(313).set_ylabel('amplitude (arb.)', fontsize = 16)
	subplot(313).set_xlabel('time (s)', fontsize = 16)
	leg()
	savefig('following_'+files[index]+'.png')

