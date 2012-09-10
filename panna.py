"""
access all loaded data via p.x[] for column one and y,z,a,b for the other columns

#### otherwise data can be loaded into arrays like this: ####

import panna as p
from pylab import *

#p.plot_all()
data = p.load_all()
p.show_n()
len_files = p.number_of_files()
x,y,z,a,b = [0]*len_files, [0]*len_files, [0]*len_files, [0]*len_files, [0]*len_files

for i in arange(0, len_files):
    len_columns = p.number_of_columns(i)
    for j in arange(0, len_columns):
        if j == 0: x[i] = array(data[i][:,j])
        if j == 1: y[i] = array(data[i][:,j])
        if j == 2: z[i] = array(data[i][:,j])
        if j == 3: a[i] = array(data[i][:,j])
        if j == 4: b[i] = array(data[i][:,j])
"""

import glob
from pylab import *

# access all csv files:
files = glob.glob('*.csv')
# a line which contains data, only contains this signs:
data_signs = ['\n','',' ','.',',','0','1','2','3','4','5','6','7','8','9','e','E','-','+']

# lines[i] oontains the whole content of file files[i]
def load_lines(file):
	lines = []
	f = open(file,'r')
	lines.append(f.readlines()[0:])
	#print "file ",file," found "
	#print "loaded lines ", lines[0][0:3], "..."
	f.close()
	return lines[0]

# The Spectrum Analyzer saves the Resolution Bandwidth into the header. It is to be read out here:
def get_RBW(datafile):
	read = load_lines(datafile)
	line_array = [line.split(',') for line in read]
	RBW_element = float(line_array[11][1]) # SA saves RBW there
	return RBW_element

# the SA in IQ mode saves the used capture time here:
def get_IQ_times(index):
	read = load_lines(files[index])
	line_array = [line.split(',') for line in read]
	# print line_array[6]
	IQ_time = float(line_array[6][1]) 
	return IQ_time

def number_of_files():
	return len(files)

def load_data(i):
	#print files[i]," access data via index ",i 
	lines = load_lines(files[i])
	data = []
	first_line = True
	show_error = True
	for line in lines[0:-2]:
	# if line contains anything but numbers . and , skip, otherwise append
		if all([element in data_signs for element in line]):
			line_array = line.strip('\n')
			line_array = line.strip('')
			line_array = line.split(',')
			if first_line:
				#print "The first line of file ",i," is ",str(line)
				columns = len(line_array)
				first_line = False
			new_array = [0]*columns
			i = 0
			for element in line_array:
				try:
					float(element)
					new_array[i] = float(element)
					i += 1
				except ValueError:
					new_array[i] = 0
					i += 1
					if show_error == True:
						print element," in ",line_array," not floatable"
						show_error = False
			data.append(new_array)	
	return array(data)
"""	
def load_all():
	i = 0
	data = [0]*len(files)
	for file in files:
		data[i] = load_data(i)
		i += 1
	return data
"""

def number_of_columns(i):
	data = load_data(i)
	return shape(data)[1]
	
def show_n():
	i = 0
	for file in files:
		print "access file ",file," via index ",i
		i += 1
	
def load_all():
	i = 0
	data = [0]*len(files)
	for file in files:
		data[i] = load_data(i)
		i += 1
	return data

def load_RBW():
	i = 0
	RBW = [0]*len(files)
	for file in files:
		RBW[i] = get_RBW(file)
		i+=1
	return RBW
	
def colorcycle(num_plots):
	colormap = plt.cm.gist_rainbow
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0.35, 1.0, num_plots)])

def grey_title(my_title):
	title(my_title, style='italic', color = 'white', bbox={'facecolor':'grey', 'alpha':0.5})

def plot_all():
	data = load_all()
	i = 0
	for file in files:
		figure()
		columns = shape(data[i])[1]
		print "file ",file," has ",columns," columns"
		colorcycle(columns)
		grey_title(str(files[i]))
		for j in arange(1,columns):
			print "plotting column ",j
			plot(data[i][:,0], data[i][:,j], label = "column "+str(j))
		legend()
		savefig("plot_data_"+str(files[i])+".png")
		close()
		i += 1
	
def find_nearest(array,value):
    idx=(np.abs(array-value)).argmin()
    return idx

def normalize(array):
	minimum = min(array)
	new_array = array - minimum
	maximum = max(new_array)
	result = new_array/maximum
	return result
	
def rms(data):
    data = array(data)
    data_mean = average(data)
    rms = sqrt(sum((data_mean-x)**2 for x in data)/len(data))
    return [rms]*len(data)
	
def scale_xaxis(subplot, factor):
    ticks = subplot.get_xticks()*factor
    subplot.set_xticklabels(ticks)
	
def scale_yaxis(subplot, factor):
    ticks = subplot.get_yticks()*factor
    subplot.set_yticklabels(ticks)
	
def load_column(index, column):
	all_data = load_data(index)
	print all_data[0]
	column_data = all_data[:,column]
	return column_data
	
# Gives Spectrum of Data
def frequency_analysis(time, signal):
    spectrum = np.fft.fft(signal)#*np.hanning(len(signal)))
    dt = abs(time[-1] - time[0])/len(time)
    frate = 1/dt
    freq = np.fft.fftfreq(len(signal))
    freq_in_hertz=abs(freq*frate) # from http://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python
    """ Check if the frequencies are in agreement with Nyquist sampling theorem:
    f_min_exp = 1/2*1/(time[-1]-time[0])
    f_max_exp = 1/2*1/dt
    print f_min_exp, f_max_exp, min(freq_in_hertz), max(freq_in_hertz)"""
    # Remove small high frequency part
    tol = 0.05*abs(spectrum).max()
    for i in xrange(len(spectrum)-1, 0, -1):
        if abs(spectrum[i]) > tol:
            break
    return freq_in_hertz[:i+1], spectrum[:i+1]

""" TEST OF SPECTRUM ANALYSIS
# Try out FFT with 5 Hz signal
t = linspace(-10, 100,10000)
signal = sin(2*pi*5*t)
freq, spectrum = frequency_analysis(t, signal)
figure()
plot(freq, abs(spectrum))
show()

itemindex=np.where(spectrum==max(spectrum))
print "The maximum of the spectrum occurs at a frequency of ", freq[itemindex][0]," Hz. In agreement with the given signal frequency."
"""

# Standard action: load all data in folder and access them via p.x[]...	

show_n()

data = load_all()
len_files = number_of_files()
x,y,z,a,b = [0]*len_files, [0]*len_files, [0]*len_files, [0]*len_files, [0]*len_files
for i in arange(0, len_files):
	len_columns = number_of_columns(i)
	for j in arange(0, len_columns):
		if j == 0: x[i] = array(data[i][:,j])
		if j == 1: y[i] = array(data[i][:,j])
		if j == 2: z[i] = array(data[i][:,j])
		if j == 3: a[i] = array(data[i][:,j])
		if j == 4: b[i] = array(data[i][:,j])

	 
	

