import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import altair as alt
from altair import datum
import subprocess


volume_atlas = ["TL hippocampus R","TL hippocampus L","TL amygdala R","TL amygdala L","TL anterior temporal lobe medial part R","TL anterior temporal lobe medial part L","TL anterior temporal lobe lateral part R","TL anterior temporal lobe lateral part L","TL parahippocampal and ambient gyrus R","TL parahippocampal and ambient gyrus L","TL superior temporal gyrus middle part R","TL superior temporal gyrus middle part L","TL middle and inferior temporal gyrus R","TL middle and inferior temporal gyrus L","TL fusiform gyrus R","TL fusiform gyrus L","cerebellum R","cerebellum L","brainstem excluding substantia nigra","insula posterior long gyrus L","insula posterior long gyrus R","OL lateral remainder occipital lobe L","OL lateral remainder occipital lobe R","CG anterior cingulate gyrus L","CG anterior cingulate gyrus R","CG posterior cingulate gyrus L","CG posterior cingulate gyrus R","FL middle frontal gyrus L","FL middle frontal gyrus R","TL posterior temporal lobe L","TL posterior temporal lobe R","PL angular gyrus L","PL angular gyrus R","caudate nucleus L","caudate nucleus R","nucleus accumbens L","nucleus accumbens R","putamen L","putamen R","thalamus L","thalamus R","pallidum L","pallidum R","corpus callosum","Lateral ventricle excluding temporal horn R","Lateral ventricle excluding temporal horn L","Lateral ventricle temporal horn R","Lateral ventricle temporal horn L","Third ventricle","FL precentral gyrus L","FL precentral gyrus R","FL straight gyrus L","FL straight gyrus R","FL anterior orbital gyrus L","FL anterior orbital gyrus R","FL inferior frontal gyrus L","FL inferior frontal gyrus R","FL superior frontal gyrus L","FL superior frontal gyrus R","PL postcentral gyrus L","PL postcentral gyrus R","PL superior parietal gyrus L","PL superior parietal gyrus R","OL lingual gyrus L","OL lingual gyrus R","OL cuneus L","OL cuneus R","FL medial orbital gyrus L","FL medial orbital gyrus R","FL lateral orbital gyrus L","FL lateral orbital gyrus R","FL posterior orbital gyrus L","FL posterior orbital gyrus R","substantia nigra L","substantia nigra R","FL subgenual frontal cortex L","FL subgenual frontal cortex R","FL subcallosal area L","FL subcallosal area R","FL pre-subgenual frontal cortex L","FL pre-subgenual frontal cortex R","TL superior temporal gyrus anterior part L","TL superior temporal gyrus anterior part R","PL supramarginal gyrus L","PL supramarginal gyrus R","insula anterior short gyrus L","insula anterior short gyrus R","insula middle short gyrus L","insula middle short gyrus R","insula posterior short gyrus L","insula posterior short gyrus R","insula anterior inferior cortex L","insula anterior inferior cortex R","insula anterior long gyrus L","insula anterior long gyrus R"]
surface_atlas = ["lUnknown","rUnknown","lG_and_S_frontomargin","rG_and_S_frontomargin","lG_and_S_occipital_inf","rG_and_S_occipital_inf","lG_and_S_paracentral","rG_and_S_paracentral","lG_and_S_subcentral","rG_and_S_subcentral","lG_and_S_transv_frontopol","rG_and_S_transv_frontopol","lG_and_S_cingul-Ant","rG_and_S_cingul-Ant","lG_and_S_cingul-Mid-Ant","rG_and_S_cingul-Mid-Ant","lG_and_S_cingul-Mid-Post","rG_and_S_cingul-Mid-Post","lG_cingul-Post-dorsal","rG_cingul-Post-dorsal","lG_cingul-Post-ventral","rG_cingul-Post-ventral","lG_cuneus","rG_cuneus","lG_front_inf-Opercular","rG_front_inf-Opercular","lG_front_inf-Orbital","rG_front_inf-Orbital","lG_front_inf-Triangul","rG_front_inf-Triangul","lG_front_middle","rG_front_middle","lG_front_sup","rG_front_sup","lG_Ins_lg_and_S_cent_ins","rG_Ins_lg_and_S_cent_ins","lG_insular_short","rG_insular_short","lG_occipital_middle","rG_occipital_middle","lG_occipital_sup","rG_occipital_sup","lG_oc-temp_lat-fusifor","rG_oc-temp_lat-fusifor","lG_oc-temp_med-Lingual","rG_oc-temp_med-Lingual","lG_oc-temp_med-Parahip","rG_oc-temp_med-Parahip","lG_orbital","rG_orbital","lG_pariet_inf-Angular","rG_pariet_inf-Angular","lG_pariet_inf-Supramar","rG_pariet_inf-Supramar","lG_parietal_sup","rG_parietal_sup","lG_postcentral","rG_postcentral","lG_precentral","rG_precentral","lG_precuneus","rG_precuneus","lG_rectus","rG_rectus","lG_subcallosal","rG_subcallosal","lG_temp_sup-G_T_transv","rG_temp_sup-G_T_transv","lG_temp_sup-Lateral","rG_temp_sup-Lateral","lG_temp_sup-Plan_polar","rG_temp_sup-Plan_polar","lG_temp_sup-Plan_tempo","rG_temp_sup-Plan_tempo","lG_temporal_inf","rG_temporal_inf","lG_temporal_middle","rG_temporal_middle","lLat_Fis-ant-Horizont","rLat_Fis-ant-Horizont","lLat_Fis-ant-Vertical","rLat_Fis-ant-Vertical","lLat_Fis-post","rLat_Fis-post","lMedial_wall","rMedial_wall","lPole_occipital","rPole_occipital","lPole_temporal","rPole_temporal","lS_calcarine","rS_calcarine","lS_central","rS_central","lS_cingul-Marginalis","rS_cingul-Marginalis","lS_circular_insula_ant","rS_circular_insula_ant","lS_circular_insula_inf","rS_circular_insula_inf","lS_circular_insula_sup","rS_circular_insula_sup","lS_collat_transv_ant","rS_collat_transv_ant","lS_collat_transv_post","rS_collat_transv_post","lS_front_inf","rS_front_inf","lS_front_middle","rS_front_middle","lS_front_sup","rS_front_sup","lS_interm_prim-Jensen","rS_interm_prim-Jensen","lS_intrapariet_and_P_trans","rS_intrapariet_and_P_trans","lS_oc_middle_and_Lunatus","rS_oc_middle_and_Lunatus","lS_oc_sup_and_transversal","rS_oc_sup_and_transversal","lS_occipital_ant","rS_occipital_ant","lS_oc-temp_lat","rS_oc-temp_lat","lS_oc-temp_med_and_Lingual","rS_oc-temp_med_and_Lingual","lS_orbital_lateral","rS_orbital_lateral","lS_orbital_med-olfact","rS_orbital_med-olfact","lS_orbital-H_Shaped","rS_orbital-H_Shaped","lS_parieto_occipital","rS_parieto_occipital","lS_pericallosal","rS_pericallosal","lS_postcentral","rS_postcentral","lS_precentral-inf-part","rS_precentral-inf-part","lS_precentral-sup-part","rS_precentral-sup-part","lS_suborbital","rS_suborbital","lS_subparietal","rS_subparietal","lS_temporal_inf","rS_temporal_inf","lS_temporal_sup","rS_temporal_sup","lS_temporal_transverse","rS_temporal_transverse"]
outcome_name = ['CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
	   'RAVLT_learning', 'Ventricles']

col1, col2 = st.columns(2)


@st.experimental_memo
def get_data(path):
	df1 = pd.read_csv(path, index_col=False)
	# df1['Z0'] = df1['Z0'].map({i: ["Female", "Male"][i] for i in range(2)})
	return df1
	


# Using "with" notation
with st.sidebar:
	cofounder_type = st.radio(
		"Confounders",
		('Age', 'Gender'),index=1)
	# use_violinplot = st.checkbox('Violin plot')

st.header("Surface")

cofounder_type_code = 0 if cofounder_type=="Age" else 1
# 	outcome_i = 
age_space = '5'
# age_bin = 10
# if cofounder_type == 'Age':
#     age_space = st.select_slider(
#         'Select age space',
#         options=['2', '5', '10'],value='5')
#     age_dict = {'2':20,'5':10, '10':5}
#     age_bin = age_dict[age_space]
col1, col2 = st.columns(2)
df1 = get_data('train_surface.csv')

#
#vilolet plot
# ============================================ Surface=========================
# ROIs = st.slider('Choose the surface region (X_i)', 1, 152, 1)
ROIs = st.multiselect("Choose the surface regions", surface_atlas, surface_atlas[5:10])
if cofounder_type_code == 0:
	step = 30
	overlap = 0
	chart_surface = alt.Chart(df1,title=f'Cortical thickness at regions against age').transform_bin(
			'Age', field='Z1',bin=alt.Bin(maxbins=2)
	).transform_fold(
		ROIs,
		as_ = ['Cortical thickness', 'value'] #Xi: key value:value
	).transform_density( #["value", "density"]
		'value',
		groupby=['Age','Cortical thickness'],
		extent=[0, 5],
		as_=['value', 'density'],
	).mark_area(orient='vertical',interpolate='monotone',
		fillOpacity=0.5,
	#     stroke='lightgray',
	#     strokeWidth=0.5
			).encode(
			x=alt.X('value:T',title='Cortical thickness',axis=None),
			# color='Z1:O',
			y=alt.Y(
				'density:Q',
				stack=None,
				impute=None,
				title=None,
				axis=None,
				scale=alt.Scale(range=[step, -step * overlap]),
				# scale=alt.Scale(domain=[0, 5])
	#             axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
			),
		color='Age:N',
			row=alt.Row(
				'Cortical thickness:N',
				header=alt.Header(labelAngle=0, labelAlign='right'
	#             header=alt.Header(
	#                 titleOrient='bottom',
	#                 labelOrient='bottom',
	#                 labelPadding=0,
				),
		)
		).properties(
			width=700,
			height=80,
			bounds='flush'
		).configure_facet(
			spacing=0
		).configure_view(
			stroke=None
		).configure_header(
		titleOrient='bottom', labelOrient='bottom',labelAnchor='start',labelAlign='center',labelPadding=-50
	)

	st.altair_chart(
	chart_surface.interactive(), use_container_width=True
)

else:
	chart_surface = alt.Chart(df1,width=40).mark_area(orient='horizontal').transform_fold(
	ROIs,
	as_ = ['ROI', 'value'] #Xi: key value:value
).transform_density( #["value", "density"]
	'value',
	groupby=['Z0','ROI'],
	as_=['value', 'density'],
).encode(
		y=alt.Y('value:Q',title='Cortical thickness'),
		color=alt.Color('Z0:N', scale=alt.Scale(scheme='dark2'),),#legend=None
		x=alt.X(
			'density:Q',
			stack=False,
			impute=None,
			title=None,axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True)),
		column=alt.Column('ROI:N'),
#         column=alt.Column('Xi:N',header=alt.Header(labelAngle=90,titleOrient='bottom', labelOrient='bottom'))
#         fill=alt.Fill(
#         'Z0:O',
#         legend=None)
#         scale=alt.Scale(domain=[30, 5], scheme='redyellowblue'))
).transform_calculate(
	density='datum.Z0=="Female"?datum.density:datum.density*-1',
	cat="datum.ROI + '-' + datum.Z0"
).configure_facet(
	spacing=0
).configure_header(
	titleOrient='bottom', labelOrient='bottom',labelAngle=90,labelAnchor='middle',labelAlign='center',labelPadding=50
).configure_view(
	stroke=None
)

	st.altair_chart(
		chart_surface.interactive(), use_container_width=False
	)

with st.expander("Remarks"):
	 st.write("""
		Remarks: Regions 85 86 can not be visualized because they contain nan values!!
	 """)
# ========================================================== Volume =============================
st.header("Volume")
# col3, col4 = st.columns(2)
df2 = get_data('train_volume.csv')
ROIs = st.multiselect("Choose the volume regions", volume_atlas, volume_atlas[5:10])
if cofounder_type_code == 0:
	step = 30
	overlap = 0
	chart_volume = alt.Chart(df2,title=f'Volume at regions against age').transform_bin(
			'Age', field='Z1',bin=alt.Bin(maxbins=2)
	).transform_fold(
		ROIs,
		as_ = ['Volume', 'value'] #Xi: key value:value
	).transform_density( #["value", "density"]
		'value',
		groupby=['Age','Volume'],
		    # extent=[0, 70],
		as_=['value', 'density'],
	).mark_area(orient='vertical',interpolate='monotone',
		fillOpacity=0.5,
	#     stroke='lightgray',
	#     strokeWidth=0.5
			).encode(
			x=alt.X('value:T',title='Volume',axis=None,scale=alt.Scale(domain=[0, 5])),
			# color='Z1:O',
			y=alt.Y(
				'density:Q',
				stack=None,
				impute=None,
				title=None,
				axis=None,
				# scale=alt.Scale(range=[step, -step * overlap]),
				scale=alt.Scale(domain=[0, 5])
	#             axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
			),
		color='Age:N',
			row=alt.Row(
				'Volume:N',
				header=alt.Header(labelAngle=0, labelAlign='right'
	#             header=alt.Header(
	#                 titleOrient='bottom',
	#                 labelOrient='bottom',
	#                 labelPadding=0,
				),
		)
		).properties(
			# width=700,
			height=80,
			bounds='flush'
		).configure_facet(
			spacing=0
		).configure_view(
			stroke=None
		).configure_header(
		titleOrient='bottom', labelOrient='bottom',labelAnchor='start',labelAlign='center',labelPadding=-20
	)


else:
	chart_volume = alt.Chart(df2,width=40).mark_area(orient='horizontal').transform_fold(
	ROIs,
	as_ = ['Xi', 'value'] #Xi: key value:value
).transform_density( #["value", "density"]
	'value',
	groupby=['Z0','Xi'],
	as_=['value', 'density'],
).encode(
		y=alt.Y('value:Q',title='Volume'),
		color=alt.Color('Z0:N', scale=alt.Scale(scheme='dark2'),),#legend=None
		x=alt.X(
			'density:Q',
			stack=False,
			impute=None,
			title=None,axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True)),
		column=alt.Column('Xi:N'),
).transform_calculate(
	density='datum.Z0=="Female"?datum.density:datum.density*-1',
	cat="datum.Xi + '-' + datum.Z0"
).configure_facet(
	spacing=0
).configure_header(
	titleOrient='bottom', labelOrient='bottom',labelAngle=90,labelAnchor='middle',labelAlign='center',labelPadding=90
).configure_view(
	stroke=None
)
	

st.altair_chart(
	chart_volume.interactive(), use_container_width=False
)
with st.expander("Remarks"):
	 st.write("""
		xxx
	 """)
	 
# ============================================================ Outputs=========================
st.header("Outputs")
outcomes = st.multiselect("Choose the outcomes", outcome_name, outcome_name[:])
if cofounder_type_code == 0:
	step = 30
	overlap = 1
	chart_output = alt.Chart(df1,title=f'Outcomes against age').transform_bin(
			'Age', field='Z1',bin=alt.Bin(maxbins=2)
	).transform_fold(
		outcomes,
		as_ = ['Outcomes', 'value'] #Xi: key value:value
	).transform_density( #["value", "density"]
		'value',
		groupby=['Age','Outcomes'],
		as_=['value', 'density'],
	).mark_area(orient='vertical',interpolate='monotone',
		fillOpacity=0.5,
	#     stroke='lightgray',
	#     strokeWidth=0.5
			).encode(
			x=alt.X('value:T',title='Outcomes',axis=None),
			# color='Z1:O',
			y=alt.Y(
				'density:Q',
				stack=None,
				impute=None,
				title=None,
				axis=None,
				scale=alt.Scale(range=[step, -step * overlap]),
	#             scale=alt.Scale(domain=[0, 5])
	#             axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
			),
		color='Age:N',
			row=alt.Row(
				'Outcomes:N',
				header=alt.Header(labelAngle=0, labelAlign='right'
	#             header=alt.Header(
	#                 titleOrient='bottom',
	#                 labelOrient='bottom',
	#                 labelPadding=0,
				),
		)
		).properties(
			# width=700,
			height=80,
			bounds='flush'
		).configure_facet(
			spacing=0
		).configure_view(
			stroke=None
		).configure_header(
		titleOrient='bottom', labelOrient='bottom',labelAnchor='start',labelAlign='center',labelPadding=-20
	)


else:
	chart_output = alt.Chart(df1,width=40).mark_area(orient='horizontal').transform_fold(
	outcomes,
	as_ = ['Outcomes', 'value'] #Xi: key value:value
).transform_density( #["value", "density"]
	'value',
	groupby=['Z0','Outcomes'],
	as_=['value', 'density'],
).encode(
		y=alt.Y('value:Q',title='Outcomes'),
		color=alt.Color('Z0:N', scale=alt.Scale(scheme='dark2'),),#legend=None
		x=alt.X(
			'density:Q',
			stack=False,
			impute=None,
			title=None,axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True)),
		column=alt.Column('Outcomes:N'),
#         column=alt.Column('Xi:N',header=alt.Header(labelAngle=90,titleOrient='bottom', labelOrient='bottom'))
#         fill=alt.Fill(
#         'Z0:O',
#         legend=None)
#         scale=alt.Scale(domain=[30, 5], scheme='redyellowblue'))
).transform_calculate(
	density='datum.Z0=="Female"?datum.density:datum.density*-1',
	cat="datum.Outcomes + '-' + datum.Z0"
).configure_facet(
	spacing=0
).configure_header(
	titleOrient='bottom', labelOrient='bottom',labelAngle=90,labelAnchor='middle',labelAlign='center',labelPadding=50
).configure_view(
	stroke=None
)
	

st.altair_chart(
	chart_output.interactive(), use_container_width=False
)



with st.expander("See explanation outputs"):
	 st.write("""
		'CDRSB': 
		'ADAS11':,
		'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
	   'RAVLT_learning', 'Ventricles'
	 """)
	 


st.header("Simulation")
uploaded_simulation = st.file_uploader("Upload simulation file")
df_simulation = pd.read_csv('simulation_train_volume.csv')
if uploaded_simulation is not None:
	df_simulation = pd.read_csv(uploaded_simulation)
	 
ROIs_simulation = st.multiselect("Choose the simulation region", surface_atlas, surface_atlas[5:10])
if cofounder_type_code == 0:
	step = 30
	overlap = 0
	chart_surface = alt.Chart(df_simulation,title=f'Cortical thickness at region against age').transform_bin(
			'Age', field='Z1',bin=alt.Bin(maxbins=3)
	).transform_fold(
		ROIs_simulation,		
		as_ = ['Cortical thickness', 'value'] #Xi: key value:value
	).transform_density( #["value", "density"]
		'value',
		# extent=[0, 5],
		groupby=['Age','Cortical thickness'],
		as_=['value', 'density'],
	).mark_area(orient='vertical',interpolate='monotone',
		fillOpacity=0.5,
	#     stroke='lightgray',
	#     strokeWidth=0.5
			).encode(
			x=alt.X('value:T',title='Cortical thickness (normalized)',axis=None),
			# color='Z1:O',
			y=alt.Y(
				'density:Q',
				stack=None,
				impute=None,
				title=None,
				axis=None,
				scale=alt.Scale(range=[step, -step * overlap]),
				# scale=alt.Scale(domain=[0, 5])
	#             axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
			),
		color='Age:N',
			row=alt.Row(
				'Cortical thickness:N',
				header=alt.Header(labelAngle=0, labelAlign='right'
	#             header=alt.Header(
	#                 titleOrient='bottom',
	#                 labelOrient='bottom',
	#                 labelPadding=0,
				),
		)
		).properties(
			# width=700,
			height=80,
			bounds='flush'
		).configure_facet(
			spacing=0
		).configure_view(
			stroke=None
		).configure_header(
		titleOrient='bottom', labelOrient='bottom',labelAnchor='start',labelAlign='center',labelPadding=-20
	)

	st.altair_chart(
	chart_surface.interactive(), use_container_width=True
)

else:
	chart_surface = alt.Chart(df_simulation,width=40).mark_area(orient='horizontal').transform_fold(
	ROIs_simulation,
	as_ = ['ROI', 'value'] #Xi: key value:value
).transform_density( #["value", "density"]
	'value',
	groupby=['Z0','ROI'],
	as_=['value', 'density'],
).encode(
		y=alt.Y('value:Q',title='Cortical thickness'),
		color=alt.Color('Z0:N', scale=alt.Scale(scheme='dark2'),),#legend=None
		x=alt.X(
			'density:Q',
			stack=False,
			impute=None,
			title=None,axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True)),
		column=alt.Column('ROI:N'),
#         column=alt.Column('Xi:N',header=alt.Header(labelAngle=90,titleOrient='bottom', labelOrient='bottom'))
#         fill=alt.Fill(
#         'Z0:O',
#         legend=None)
#         scale=alt.Scale(domain=[30, 5], scheme='redyellowblue'))
).transform_calculate(
	density='datum.Z0=="Female"?datum.density:datum.density*-1',
	cat="datum.ROI + '-' + datum.Z0"
).configure_facet(
	spacing=0
).configure_header(
	titleOrient='bottom', labelOrient='bottom',labelAngle=90,labelAnchor='middle',labelAlign='center',labelPadding=50
).configure_view(
	stroke=None
)

	st.altair_chart(
		chart_surface.interactive(), use_container_width=False
	)
	# ============================================================ Outputs=========================

outcomes_simulation = st.multiselect("Choose the simulation outcomes", outcome_name, outcome_name[:])
if cofounder_type_code == 0:
	step = 30
	overlap = 1
	chart_output = alt.Chart(df_simulation,title=f'Outcomes against age').transform_bin(
			'Age', field='Z1',bin=alt.Bin(maxbins=3)
	).transform_fold(
		outcomes_simulation,
		as_ = ['Outcomes', 'value'] #Xi: key value:value
	).transform_density( #["value", "density"]
		'value',
		groupby=['Age','Outcomes'],
		as_=['value', 'density'],
	).mark_area(orient='vertical',interpolate='monotone',
		fillOpacity=0.5,
	#     stroke='lightgray',
	#     strokeWidth=0.5
			).encode(
			x=alt.X('value:T',title='Outcomes',axis=None),
			# color='Z1:O',
			y=alt.Y(
				'density:Q',
				stack=None,
				impute=None,
				title=None,
				axis=None,
				scale=alt.Scale(range=[step, -step * overlap]),
	#             scale=alt.Scale(domain=[0, 5])
	#             axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
			),
		color='Age:N',
			row=alt.Row(
				'Outcomes:N',
				header=alt.Header(labelAngle=0, labelAlign='right'
	#             header=alt.Header(
	#                 titleOrient='bottom',
	#                 labelOrient='bottom',
	#                 labelPadding=0,
				),
		)
		).properties(
			# width=700,
			height=80,
			bounds='flush'
		).configure_facet(
			spacing=0
		).configure_view(
			stroke=None
		).configure_header(
		titleOrient='bottom', labelOrient='bottom',labelAnchor='start',labelAlign='center',labelPadding=-20
	)


else:
	chart_output = alt.Chart(df_simulation,width=40).mark_area(orient='horizontal').transform_fold(
	outcomes_simulation,
	as_ = ['Outcomes', 'value'] #Xi: key value:value
).transform_density( #["value", "density"]
	'value',
	groupby=['Z0','Outcomes'],
	as_=['value', 'density'],
).encode(
		y=alt.Y('value:Q',title='Outcomes'),
		color=alt.Color('Z0:N', scale=alt.Scale(scheme='dark2'),),#legend=None
		x=alt.X(
			'density:Q',
			stack=False,
			impute=None,
			title=None,axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True)),
		column=alt.Column('Outcomes:N'),
#         column=alt.Column('Xi:N',header=alt.Header(labelAngle=90,titleOrient='bottom', labelOrient='bottom'))
#         fill=alt.Fill(
#         'Z0:O',
#         legend=None)
#         scale=alt.Scale(domain=[30, 5], scheme='redyellowblue'))
).transform_calculate(
	density='datum.Z0=="Female"?datum.density:datum.density*-1',
	cat="datum.Outcomes + '-' + datum.Z0"
).configure_facet(
	spacing=0
).configure_header(
	titleOrient='bottom', labelOrient='bottom',labelAngle=90,labelAnchor='middle',labelAlign='center',labelPadding=50
).configure_view(
	stroke=None
)
	

st.altair_chart(
	chart_output.interactive(), use_container_width=False
)
