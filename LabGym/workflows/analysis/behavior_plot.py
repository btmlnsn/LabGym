"""
LabGym.workflows.analysis.behavior_plot

Behavior plot helpers extracted from LabGym.core.tools
Safe to import from any layer; no GUI dependencies.
"""

from __future__ import annotations

import datetime
import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
import seaborn as sb


# ADDEDFRM LabGym.core.tools
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



__all__ = ["plot_events"]
