'''
2020-01-07 CambridgeNeurotech
contact: Thal Holtzman
email: info@cambridgeneurotech.com

Derive probes to be used with SpikeInterface base on Cambridgeneurotech databases
Probe library to match and add on https://gin.g-node.org/spikeinterface/probeinterface_library/src/master/cambridgeneurotech
see repos https://github.com/SpikeInterface/probeinterface

In the 'Probe Maps 2020Final.xlsx'
'''



###########################################################
##  Main library
###########################################################
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from probeinterface import Probe
from probeinterface.plotting import plot_probe
from probeinterface import generate_multi_columns_probe
from probeinterface import combine_probes
from probeinterface import write_probeinterface, read_probeinterface
import math
import os

###########################################################
## custom and adapted methods
###########################################################
workDir = r"C:\Users\Windows\Dropbox (Scripps Research)\2021-01-SpikeInterface_CambridgeNeurotech"
workDirexport = 'export'
os.makedirs(workDir+os.sep+workDirexport, exist_ok=True)
os.chdir(workDir)

# graphing parameters
mpl.rcParams['pdf.fonttype'] = 42 # to make sure it is recognize as true font in illustrator
mpl.rcParams['svg.fonttype'] = 'none'  # to make sure it is recognize as true font in illustrator

def convertProbeShape(listCoord):
	'''
	This is to convert reference point probe shape inputed in excel as string 'x y x y x y that outline the shape of one shanck
	and can be converted to an array to draw the porbe
	'''
	listCoord = [float(s) for s in listCoord.split(' ')]
	res = [[listCoord[i], listCoord[i + 1]] for i in range(len(listCoord) - 1)]
	res = res[::2]

	return res

def convertElecrodeShape(listCoord):
	'''
	This is to convert reference shift in electrodes
	'''
	listCoord = [float(s) for s in listCoord.split(' ')]
	return listCoord

def getChannelIndex(connector, probeType, channelIdxRefFile = workDir+os.sep+'Probe Maps 2020Final.xlsx'):

	# first part of the function to opne the proper connector based on connector name
	# xls = pd.ExcelFile(channelIdxRefFile)
	# for shtIdx,shtName in enumerate(xls.sheet_names):
	df = pd.read_excel(channelIdxRefFile, sheet_name=connector, header = [0,1]) # header [0,1] is used to create a mutliindex 

	# second part to get the proper channel in the
	if probeType == 'E-1' or probeType == 'E-2':
		probeType = 'E-1 & E-2'

	if probeType == 'P-1' or probeType == 'P-2':
		probeType = 'P-1 & P-2'

	if probeType == 'H3' or probeType == 'L3':
        probeType = 'H3 & L3'

    if probeType == 'H5' or probeType == 'H9':
        probeType = 'H5 & H9'

	tmpList = []
	for i in df[probeType].columns:
		if len(df[probeType].columns) == 1:
			tmpList = np.flip(df[probeType].values.astype(int).flatten())
		else:
			tmp = df[probeType][i].values
			tmp = tmp[~np.isnan(tmp)].astype(int) # get rid of nan and convert to integer
			tmp = np.flip(tmp) # this flips the value to match index that goes from tip to headstage of the probe
			tmpList = np.append(tmpList, tmp)
			tmpList = tmpList.astype(int)

	return tmpList

def plot_probeCN(probe, ax=None, electrode_colors=None,
				with_channel_index=False, first_index='auto',
				electrode_values=None, cmap='viridis',
				title=True, electrodes_kargs={}, probe_shape_kwargs={},
				xlims=None, ylims=None, zlims=None):
	"""
	plot one probe.
	switch 2d 3d depending the Probe.ndim
	
	"""
	import matplotlib.pyplot as plt
	if probe.ndim == 2:
		from matplotlib.collections import PolyCollection
	elif probe.ndim == 3:
		from mpl_toolkits.mplot3d.art3d import Poly3DCollection

	if ax is None:
		if probe.ndim == 2:
			fig, ax = plt.subplots()
			ax.set_aspect('equal')
		else:
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1, projection='3d')

	if first_index == 'auto':
		if 'first_index' in probe.annotations:
			first_index = probe.annotations['first_index']
		elif probe.annotations.get('manufacturer', None) == 'neuronexus':
			# neuronexus is one based indexing
			first_index = 1
		else:
			first_index = 0
	assert first_index in (0, 1)

	_probe_shape_kwargs = dict(facecolor='#6f6f6e', edgecolor='k', lw=0.5, alpha=0.3) # made change to default color
	_probe_shape_kwargs.update(probe_shape_kwargs)

	_electrodes_kargs = dict(alpha=0.7, edgecolor=[0.3, 0.3, 0.3], lw=0.5)
	_electrodes_kargs.update(electrodes_kargs)

	n = probe.get_electrode_count()

	if electrode_colors is None and electrode_values is None:
		electrode_colors = ['#5bc5f2'] * n  # made change to default color
	elif electrode_colors is not None:
		electrode_colors = electrode_colors
	elif electrode_values is not None:
		electrode_colors = None

	# electrodes
	positions = probe.electrode_positions

	vertices = probe.get_electrodes_vertices()
	if probe.ndim == 2:
		poly = PolyCollection(vertices, color=electrode_colors, **_electrodes_kargs)
		ax.add_collection(poly)
	elif probe.ndim == 3:
		poly =  Poly3DCollection(vertices, color=electrode_colors, **_electrodes_kargs)
		ax.add_collection3d(poly)
	
	if electrode_values is not None:
		poly.set_array(electrode_values)
		poly.set_cmap(cmap)
		

	# probe shape
	planar_contour = probe.probe_planar_contour
	if planar_contour is not None:
		if probe.ndim == 2:
			poly_contour = PolyCollection([planar_contour], **_probe_shape_kwargs)
			ax.add_collection(poly_contour)
		elif probe.ndim == 3:
			poly_contour = Poly3DCollection([planar_contour], **_probe_shape_kwargs)
			ax.add_collection3d(poly_contour)


	else:
		poly_contour = None
	
	if with_channel_index:
		if probe.ndim == 3:
			raise NotImplementedError('Channel index is 2d only')
		for i in range(n):
			x, y = probe.electrode_positions[i]
			if probe.device_channel_indices is None:
				txt = f'{i + first_index}'
			else:
				chan_ind = probe.device_channel_indices[i]
				txt = f'{chan_ind}'# f'prb{i + first_index}\ndev{chan_ind}' # modification from original
			ax.text(x, y, txt, ha='center', va='center')

	
	
	if xlims is None or ylims is None or (zlims is None and probe.ndim == 3):
		xlims, ylims, zlims = get_auto_lims(probe)
	
	ax.set_xlim(*xlims)
	ax.set_ylim(*ylims)
	ax.set_xlabel(u'Width (\u03bcm)') #modif to legend
	ax.set_ylabel(u'Height (\u03bcm)') #modif to legend

	ax.spines['right'].set_visible(False) #remove external axis
	ax.spines['top'].set_visible(False) #remove external axis

	fig.set_size_inches(18.5, 10.5) #modif set size
	im = plt.imread(workDir+os.sep+'CN_logo-01.jpg')
	newax = fig.add_axes([0.8,0.85,0.2,0.1], anchor='NW', zorder=0)
	newax.imshow(im)
	newax.axis('off')

	if probe.ndim == 3:
		ax.set_zlim(zlims)
		ax.set_zlabel('z')

	if probe.ndim == 2:
		ax.set_aspect('equal')

	if title:
		tmpTitle = probe.get_title()
		ax.set_title('\n' +'CambridgeNeuroTech' +'\n'+  probe.annotations.get('name'), fontsize = 24)
	
	plt.tight_layout() #modif tight layout
	return poly, poly_contour

def plot_probe_group(probegroup, same_axe=True, **kargs):
	"""
	Plot all prbe from a ProbeGroup
	
	Can be in the same axe or separated axes.
	"""
	import matplotlib.pyplot as plt
	n = len(probegroup.probes)

	if same_axe:
		if 'ax' in kargs:
			ax = kargs.pop('ax')
		else:
			if probegroup.ndim == 2:
				fig, ax = plt.subplots()
			else:
				fig = plt.figure()
				ax = fig.add_subplot(1, 1, 1, projection='3d')
		axs = [ax] * n
	else:
		if 'ax' in kargs:
			raise valueError('with same_axe=False do not provide ax')
		if probegroup.ndim == 2:
			fig, axs = plt.subplots(ncols=n, nrows=1)
			if n == 1:
				axs = [axs]
		else:
			raise NotImplementedError
	
	if same_axe:
		# global lims
		xlims, ylims, zlims = get_auto_lims(probegroup.probes[0])
		for i, probe in enumerate(probegroup.probes):
			xlims2, ylims2, zlims2 = get_auto_lims(probe)
			xlims = min(xlims[0], xlims2[0]), max(xlims[1], xlims2[1])
			ylims = min(ylims[0], ylims2[0]), max(ylims[1], ylims2[1])
			if zlims is not None:
				zlims = min(zlims[0], zlims2[0]), max(zlims[1], zlims2[1])
		kargs['xlims'] = xlims
		kargs['ylims'] = ylims
		kargs['zlims'] = zlims
	else:
		# will be auto for each probe in each axis
		kargs['xlims'] = None
		kargs['ylims'] = None
		kargs['zlims'] = None
	
	kargs['title'] = False
	for i, probe in enumerate(probegroup.probes):
		plot_probe(probe, ax=axs[i], **kargs)

def get_auto_lims(probe, margin=40):
	positions = probe.electrode_positions
	planar_contour = probe.probe_planar_contour
	

	xlims = np.min(positions[:, 0]), np.max(positions[:, 0])
	ylims = np.min(positions[:, 1]), np.max(positions[:, 1])
	zlims = None
	
	if probe.ndim == 3:
		zlims = np.min(positions[:, 2]), np.max(positions[:, 2])
	
	if planar_contour is not None:
		
		xlims2 = np.min(planar_contour[:, 0]), np.max(planar_contour[:, 0])
		xlims = min(xlims[0], xlims2[0]), max(xlims[1], xlims2[1])

		ylims2 = np.min(planar_contour[:, 1]), np.max(planar_contour[:, 1])
		ylims = min(ylims[0], ylims2[0]), max(ylims[1], ylims2[1])
		
		if probe.ndim == 3:
			zlims2 = np.min(planar_contour[:, 2]), np.max(planar_contour[:, 2])
			zlims = min(zlims[0], zlims2[0]), max(zlims[1], zlims2[1])

	xlims = xlims[0] - margin, xlims[1] + margin
	ylims = ylims[0] - margin, ylims[1] + margin

	if probe.ndim == 3:
		zlims = zlims[0] - margin, zlims[1] + margin

		# to keep equal ascpect in 3d
		# all axes have the same limits
		lims = min(xlims[0], ylims[0], zlims[0]), max(xlims[1], ylims[1], zlims[1])
		xlims, ylims, zlims =  lims, lims, lims

	
	return xlims, ylims, zlims

def generate_CNprobe(j, probeIdx):
	if j['part'] == 'Fb' or j['part'] == 'F':
		'''
		conditions to build probe F and Fb properly 
		'''
		probe = generate_multi_columns_probe(num_columns=j['electrode_cols_n'], num_elec_per_column=[int(x) for x in convertProbeShape(j['electrode_rows_n'])[probeIdx]],
								 xpitch=float(j['electrodeSpacingWidth_um']), ypitch=j['electrodeSpacingHeight_um'], y_shift_per_column=convertProbeShape(j['electrode_yShiftCol'])[probeIdx],
								 electrode_shapes=j['ElectrodeShape'], electrode_shape_params={'width': j['electrodeWidth_um'], 'height': j['electrodeHeight_um']})
		probe.set_planar_contour(convertProbeShape(j['probeShape']))

	else:
		probe = generate_multi_columns_probe(num_columns=j['electrode_cols_n'], num_elec_per_column=int(j['electrode_rows_n']),
							 xpitch=float(j['electrodeSpacingWidth_um']), ypitch=j['electrodeSpacingHeight_um'], y_shift_per_column=convertElecrodeShape(j['electrode_yShiftCol']),
							 electrode_shapes=j['ElectrodeShape'], electrode_shape_params={'width': j['electrodeWidth_um'], 'height': j['electrodeHeight_um']})
		probe.set_planar_contour(convertProbeShape(j['probeShape']))

	if type(j['electrodesCustomPosition']) == str:
		probe.electrode_positions = np.array(convertProbeShape(j['electrodesCustomPosition']))

	return probe

def exportAll(j, probe):
	exportDir = workDir+os.sep+workDirexport+os.sep+connector+'-'+j['part']
	exportProbe = os.sep+connector+'-'+j['part']
	os.makedirs(exportDir, exist_ok=True)
	os.chdir(exportDir)
	write_probeinterface(exportDir+exportProbe+'.json', probe)
	plot_probeCN(probe, with_channel_index=True)
	plt.savefig(exportDir+exportProbe+'.png')
	plt.savefig(exportDir+exportProbe+'.svg')
	plt.close('all')
	os.chdir('..')

def convertShift(array, xShift, yShift):
	x= probe.electrode_positions
	# x= np.array(convertProbeShape(j['probeShape']))
	# x = convertShift(x, -7, 25)
	array[:, 0] = array[:, 0]+xShift
	array[:, 1] = array[:, 1]+yShift

	array = array.flatten()
	array = list(array)

	with open(r'C:\Users\Windows\Desktop\tst\string.txt', "w") as outfile:
	    outfile.write(" ".join(str(item) for item in array))

	return array
x = probe.electrode_positions
x = np.array(convertProbeShape(j['probeShape']))
y = convertShift(x, -16, 20)

with open(r'C:\Users\Windows\Desktop\tst\string.txt', "w") as outfile:
    outfile.write(" ".join(str(item) for item in y))

###########################################################
## Features that could be added
###########################################################

# set up for local environment / folder
# auto log for error flagging

###########################################################
## application
###########################################################

plt.ioff()

	
probesRef = pd.read_csv(workDir+os.sep+"ProbesDataBase.csv")
# probesRef = probesRef.iloc[30:]

toReDo = []
for i, j in probesRef.iterrows():
	print(i,j['part'])
	# value = 30
	# if i == value:
	#     print(i,j['part'])
	# else:
	#     continue
	# if i == value:
	#     break

	if j['shanks_n'] == 1:
		probe = generate_CNprobe(j, 0)

		for connector in list(j[j.index.str.contains('ASSY')].dropna().index):
			print(connector)
			channelIndex = getChannelIndex(connector = connector, probeType = j['part'])

			probe.annotate(name=connector+'-'+j['part'], manufacturer='cambridgeneurotech') 
			probe.set_device_channel_indices(channelIndex)

			exportAll(j, probe)

	else: # for the shnak more than one

		probe_dict = {} # intialize the probe dictionary
		for probeIdx in np.arange(0,j['shanks_n']):
			print(probeIdx)
			# set the key characteristics of the probe
			probe_dict["probe%s" %probeIdx] = generate_CNprobe(j, probeIdx)
			# list(probe_dict.values())[probeIdx].set_device_channel_indices(channelIndex[probeIdx])
			list(probe_dict.values())[probeIdx].move([j['shankSpacing_um']*probeIdx, 0])
			

		# retrive the values of the dicationary see below
		# dict usage list(probe_dict.items()) list(probe_dict.values()) probe_dict.keys()
		multi_shank_list = list(probe_dict.values())
		multi_shank = combine_probes(multi_shank_list)

		for connector in list(j[j.index.str.contains('ASSY')].dropna().index):
			print(connector)
			channelIndex = getChannelIndex(connector = connector, probeType = j['part'])

			multi_shank.annotate(name=connector+'-'+j['part'], manufacturer='cambridgeneurotech') 
			multi_shank.set_device_channel_indices(channelIndex)

			exportAll(j, multi_shank)
