"""
LabGym.workflows.analysis.distance_metrics

Distance metric helpers extracted from LabGym.core.tools
Safe to import from any layer; no GUI dependencies.
"""

from __future__ import annotations

#Standard library imports
import os
import math

# Related third-party imports
import cv2
import pandas as pd


# ADDEDFRM LabGym.core.tools
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



__all__ = ["calculate_distances"]
