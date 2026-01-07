'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)#fulltext.

For license issues, please contact:

Dr. Bing Ye
Life Sciences Institute
University of Michigan
210 Washtenaw Avenue, Room 5403
Ann Arbor, MI 48109-2216
USA

Email: bingye@umich.edu
'''

"""
LabGym.core.tools - DEPRECATED SHIM

Video / image helpers now live in LabGym.io.video
Filesystem helpers now live in LabGym.io.filesystem

Plotting and distance-metric helpers remain here temporarily;
they will be relocated to LabGym.workflows.analysis later into my refactoring.
"""
# Standard library imports.
import warnings
import logging
import datetime
import logging
import math
import os

# Log the load of this module (by the module loader, on first import).
# Intentionally positioning these statements before other imports, against the
# guidance of PEP-8, to log the load before other imports log messages.
logger =  logging.getLogger(__name__)  # pylint: disable=wrong-import-position
logger.debug('loading %s', __file__)  # pylint: disable=wrong-import-position

# Related third party imports.
import cv2
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap,Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

# Local application/library specific imports.
# RE-EXPORTING LEGACY HELPER FUNCTIONS THAT WERE MOVED 
from LabGym.io.video import *
from LabGym.io.filesystem import (
	parse_all_events_file,
	sort_examples_from_csv,
)


# WILLADD2 LabGym.workflows.analysis.behavior_plot
def plot_events(result_path,event_probability,time_points,names_and_colors,behavior_to_include,width=0,height=0):

	'''
	This function is used to plot a raster plot for behavior events and probability
	over time based on the 'event_probability' and 'time_points'.

	names_and_colors: the behavior names and their representative colors
	width and height: the size of the plot, when ==0, the size is defined automatically
	'''

	print('Exporting the raster plot for this analysis batch...')
	print(datetime.datetime.now())

	if width==0 or height==0:
		time_length=len(time_points)
		if time_length>30000:
			width=round(time_length/3000)+1
			x_intvl=3000
		elif time_length>3000:
			width=round(time_length/300)+1
			x_intvl=300
		else:
			width=round(time_length/30)+1
			x_intvl=30
		height=round(len(event_probability)/4)+1
		if height<3:
			height=3
		figure,ax=plt.subplots(figsize=(width,height))
		if height<=5:
			figure.subplots_adjust(bottom=0.25)

	for behavior_name in behavior_to_include:

		all_data=[]
		masks=[]

		for i in event_probability:
			data=[]
			mask=[]
			for n in range(len(event_probability[i])):
				if event_probability[i][n][0]==behavior_name:
					data.append(event_probability[i][n][1])
					mask.append(0)
				else:
					data.append(-1)
					mask.append(True)
			all_data.append(data)
			masks.append(mask)

		all_data=np.array(all_data)
		masks=np.array(masks)
		dataframe=pd.DataFrame(all_data,columns=[float('{:.1f}'.format(i)) for i in time_points])

		heatmap=sb.heatmap(dataframe,mask=masks,xticklabels=x_intvl,cmap=LinearSegmentedColormap.from_list('',names_and_colors[behavior_name]),cbar=False,vmin=0,vmax=1)
		heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90)
		# if don't want the ticks
		# ax.tick_params(axis='both',which='both',length=0)

	plt.savefig(os.path.join(result_path,'behaviors_plot.png'))

	for behavior_name in behavior_to_include:

		colorbar_fig=plt.figure(figsize=(5,1))
		ax=colorbar_fig.add_axes([0,1,1,1])
		colorbar=ColorbarBase(ax,orientation='horizontal',cmap=LinearSegmentedColormap.from_list('',names_and_colors[behavior_name]),norm=Normalize(vmin=0,vmax=1),ticks=[])
		colorbar.outline.set_linewidth(0)

		plt.savefig(os.path.join(result_path,behavior_name+'_colorbar.png'),bbox_inches='tight')
		plt.close()

	plt.close('all')

	print('The raster plot stored in: '+str(result_path))


# WILLADD2 LabGym.workflows.analysis.distance_metrics
def calculate_distances(path_to_folder,filename,behavior_to_include,out_path):

	'''
	This function is used to calculate the shortes distance and the total
	traveling dsitance and their ratio among the locations of the animals
	when a selected behavior occurs for the first time.

	For example, an animal explores locations A, B, and C in sequence.
	This function will calculate the shortes distance that connects
	locations A, B, and C, in the exploration sequence of the aninmal.
	It will also calculate the traveling distance of the actual route of
	the animal.

	path_to_folder: The path to the folder that stores the 'all_event_probability.xlsx',
	'all_centers.xlsx', and 'Annotated video.avi'.
	filename: the name of the path_to_folder
	behavior_to_include: the behaviors used in calculation
	'''

	animals=[]
	all_centers=[]
	all_event_probability=[]

	for i in os.listdir(path_to_folder):
		if i.endswith('_centers.xlsx') or i.endswith('_centers.xls') or i.endswith('_centers.XLSX') or i.endswith('_centers.XLS'):
			all_centers.append(i)
		if i.endswith('_event_probability.xlsx') or i.endswith('_event_probability.xls') or i.endswith('_event_probability.XLSX') or i.endswith('_event_probability.XLS'):
			all_event_probability.append(i)

	if len(all_centers)>1:
		for i in all_centers:
			animals.append(i.split('_')[0])
	else:
		if len(all_centers[0].split('_'))>2:
			animals.append(all_centers[0].split('_')[0])
		else:
			animals=['']

	for a,animal in enumerate(animals):

		all_centers_df=pd.read_excel(os.path.join(path_to_folder,all_centers[a]))
		all_events_probability_df=pd.read_excel(os.path.join(path_to_folder,all_event_probability[a]))

		centers={}
		behavior_names={}
		included_behaviors={}
		start_centers={}
		start_indices={}
		frame_count=0
		frame_index=None

		for col_name,col in all_centers_df.items():

			if col_name=='time/ID':
				time_points=[float(i) for i in col]
			else:
				idx=int(col_name)
				centers[idx]=[]
				behavior_names[idx]=[]
				included_behaviors[idx]=[]
				start_centers[idx]={}
				start_indices[idx]={}
				for i in col:
					try:
						value=eval(i)
					except:
						value=None
					centers[idx].append(value)

		for col_name,col in all_events_probability_df.items():

			if col_name!='time/ID':
				idx=int(col_name)
				for n,i in enumerate(col):
					event=eval(i)
					behavior=event[0]
					if behavior!='NA':
						if frame_index is None:
							if behavior in behavior_to_include:
								frame_index=n
						if behavior not in behavior_names[idx]:
							behavior_names[idx].append(behavior)
							start_centers[idx][behavior]=centers[idx][n]
							start_indices[idx][behavior]=n

				if len(behavior_names[idx])<len(behavior_to_include):
					included_behaviors[idx]=behavior_names[idx]
				else:
					included_behaviors[idx]=behavior_to_include

		capture=cv2.VideoCapture(os.path.join(path_to_folder,'Annotated video.avi'))
		while True:
			ret,frame=capture.read()
			if frame_count>frame_index:
				break
			if frame is None:
				break
			frame_count+=1
		capture.release()

		shortest_distances={}
		traveling_distances={}
		durations={}
		speeds={}
		velocities={}
		distance_ratios={}
		diff=int(255/len(behavior_to_include))+25
		diff_animal=int(255/len(centers))+25

		for idx in start_centers:

			shortest_distance=0.0
			traveling_distance=0.0
			centers_for_calculation=[]
			indices_for_calculation=[]

			for behavior in start_centers[idx]:
				if behavior in included_behaviors[idx]:
					centers_for_calculation.append(start_centers[idx][behavior])
					indices_for_calculation.append(start_indices[idx][behavior])

			n=0
			centers_traveled=centers[idx][indices_for_calculation[0]:indices_for_calculation[-1]+1]
			while n<len(centers_traveled)-1:
				if centers_traveled[n] is not None:
					if centers_traveled[n+1] is not None:
						cv2.line(frame,centers_traveled[n],centers_traveled[n+1],(255,0,max(0,255-int(idx*diff_animal))),2)
						traveling_distance+=math.dist(centers_traveled[n],centers_traveled[n+1])
					else:
						cv2.circle(frame,(centers_traveled[n]),2,(255,0,max(0,255-int(idx*diff_animal))),-1)
				n+=1

			n=0
			while n<len(centers_for_calculation):
				if n!=len(centers_for_calculation)-1:
					shortest_distance+=math.dist(centers_for_calculation[n],centers_for_calculation[n+1])
					cv2.circle(frame,(centers_for_calculation[n]),4,(max(0,255-int(n*diff)),max(0,255-int(n*diff)),0),-1)
					cv2.line(frame,centers_for_calculation[n],centers_for_calculation[n+1],(max(0,255-int(n*diff)),max(0,255-int(n*diff))),4)
				n+=1

			shortest_distances[idx]=shortest_distance
			traveling_distances[idx]=traveling_distance
			duration=time_points[indices_for_calculation[-1]]-time_points[indices_for_calculation[0]]
			durations[idx]=duration
			speeds[idx]=traveling_distance/duration
			velocities[idx]=shortest_distance/duration
			distance_ratios[idx]=shortest_distance/traveling_distance

		out_spreadsheet=[]
		out_spreadsheet.append(pd.DataFrame.from_dict(shortest_distances,orient='index',columns=['shortest_distances']).reset_index(drop=True))
		out_spreadsheet.append(pd.DataFrame.from_dict(traveling_distances,orient='index',columns=['traveling_distances']).reset_index(drop=True))
		out_spreadsheet.append(pd.DataFrame.from_dict(durations,orient='index',columns=['durations']).reset_index(drop=True))
		out_spreadsheet.append(pd.DataFrame.from_dict(speeds,orient='index',columns=['speeds']).reset_index(drop=True))
		out_spreadsheet.append(pd.DataFrame.from_dict(velocities,orient='index',columns=['velocities']).reset_index(drop=True))
		out_spreadsheet.append(pd.DataFrame.from_dict(distance_ratios,orient='index',columns=['distance_ratios']).reset_index(drop=True))
		if animals[0]=='':
			pd.concat(out_spreadsheet,axis=1).to_excel(os.path.join(out_path,filename+'_distance_calculation.xlsx'),float_format='%.2f',index_label='ID/parameter')
			cv2.imwrite(os.path.join(out_path,filename+'_shortest_distance.jpg'),frame)
		else:
			pd.concat(out_spreadsheet,axis=1).to_excel(os.path.join(out_path,filename+'_'+animal+'_distance_calculation.xlsx'),float_format='%.2f',index_label='ID/parameter')
			cv2.imwrite(os.path.join(out_path,filename+'_'+animal+'_shortest_distance.jpg'),frame)

	print('Distances calculation completed!')


# DEPRECATION NOTICE
warnings.warn(
	"LabGym.core.tools is deprecated; "
	"import helpers from LabGym.io.video, LabGym.io.filesystem, or "
	"LabGym.workflows.analysis (after the upcoming refactor).",
	DeprecationWarning,
	stacklevel=2,
)
