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
	st.header("Simulation parameters")
	st.title('Basic')
	# basic_type = st.radio( "Basic type",('svd-orthonomal', 'NMF-nonnegative'),index=0)  #
	gender_w = st.number_input('Gender weight (%)', value=10)
	age_w = st.number_input('Age weight (%)', value=90)
	# use_violinplot = st.checkbox('Violin plot')
	st.title('Gender')
	gender_ratio = st.number_input('gender_ratio', value=0.2)
	# age_col1,age_col2,age_col3 = st.columns(3)
	# with age_col1:
	st.title('Age')
	w_age1 = st.number_input('Weight Age group 1', value=20)
	# with age_col2:
	w_age2 = st.number_input('Weight Age group 2', value=20)
	# with age_col3:
	w_age3 = st.number_input('Weight Age group 3', value=60)	
	st.title('Noise')
	SNR = st.number_input('y SNR', value=15)
	SNR_Z = st.number_input('Zy SNR', value=0)
	SNR_ZX = st.number_input('ZX SNR', value=15)
	with st.expander("Other parameters"):
		random_type = st.radio('Random type', ('rand','randn'),index=0)
		SNR_ZPQ = st.number_input('ZPQ SNR', value=50)
	
	st.title('PQ visualization')
	PQ_idx = st.slider("Select PQ_th to visualize", 0,7,0)

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
	 

# =================================================================== SIMULATION=============================================
st.header("Simulation")
import numpy as np
# from sklearn.decomposition import NMF
from numpy import linalg as LA
import pandas as pd
import copy
rng = np.random.RandomState(0)
N = 1000
X_component = 20 #max features/region: 152
K = 5
selected_components = [2,4,5,8,10]
selected_components_weight = [1,4,2,0.5,2]
R = 2
I = 152
J = 8
K = 5

Rx = I
Ry = J

#simulation
Q = np.random.rand(J,K)


#Confounders
Z = np.zeros((N,R))
Z_norm = np.zeros((N,R))
#gender
# gender_ratio = st.number_input('gender_ratio', 0.15)
# SNR = st.number_input('y SNR', 35)
# SNR_Z = st.number_input('Zy SNR', 35)
# SNR = st.select_slider("Select y SNR", range(-5,50,5),35) #10db
# SNR_Z = st.select_slider("Select ZY SNR", range(-5,50,5),10) #10db

# gender_ratio = 0.15
Z[:,0] = [0]*int(N*gender_ratio) + [1]*(N-int(N*gender_ratio)) 
#age
age_group_weight = np.array([w_age1, w_age2, w_age3])/100 #40-60-80 age group percentage
age_group_weight_norm = np.array([33, 33, 34])/100 #40-60-80 age group percentage
age_group_sample = [int(x*N) for x in age_group_weight]
age_group_sample_norm = [int(x*N) for x in age_group_weight_norm]

Z[:,1] = np.random.randint(30, 50, size=(age_group_sample[0],)).tolist() + np.random.randint(51, 70, size=(age_group_sample[1],)).tolist()+ np.random.randint(71, 90, size=(age_group_sample[2],)).tolist()
alpha = np.eye(K)

Z_weight = np.array([gender_w,age_w])/100 #40-60-80 age group percentage
# Z_norm = copy.copy(Z)
Z_norm[:,0] = [0]*int(N*0.5) + [1]*(N-int(N*0.5)) 
Z_norm[:,0] = Z[:,0]/LA.norm(Z[:,0])
Z_norm[:,1] = np.random.randint(30, 50, size=(age_group_sample_norm[0],)).tolist() + np.random.randint(51, 70, size=(age_group_sample_norm[1],)).tolist()+ np.random.randint(71, 90, size=(age_group_sample_norm[2],)).tolist()
Z_norm[:,1] = Z[:,1]/LA.norm(Z[:,1])
Z_norm = Z_norm*Z_weight


if random_type == 'rand':
	ZY = np.random.rand(R,J)/ np.sqrt(1/12) # var unifor = 1/12
	ZX = np.random.rand(R,I)/ np.sqrt(1/12)
	ZPQ = np.random.rand(R,Rx)/ np.sqrt(1/12)
elif random_type == 'randn':
	ZY = np.random.randn(R,J)
	ZX = np.random.randn(R,I)
	ZPQ = np.random.randn(R,Rx)

#balance or not
# residual
#bottom-up approach from PQ => X =>Y : asssume PQ (sin/sparse (5% regions/voxels)) ~ more realistic 
#top-down approach goes from X =>PQ 
#simulate low rank input matrix
# U, S, V = np.linalg.svd(np.random.randint(1, 5, size=(N, I)), full_matrices=True)
basic_type ='svd-orthonomal'
if basic_type =='svd-orthonomal':
	U, S, V = np.linalg.svd(2.26+0.23*np.random.randn(N, I), full_matrices=False)
	U = U[:,:Rx]
	U = U + Z_norm@ZPQ*np.linalg.norm(U, axis=0)/10**(SNR_ZPQ/10)
	X = U @ np.diag(S)[:Rx,:Rx]@ V[:,:Rx].T 
# elif basic_type =='NMF-nonnegative':
# 	model = NMF(n_components=Rx, init='random', random_state=0)
# 	U = model.fit_transform(2.26+0.23*np.random.randn(N, I))
# 	U = U + Z_norm@ZPQ*np.linalg.norm(U, axis=0)/10**(SNR_ZPQ/10)
# 	V = model.components_
# 	X = U@V




PQ_true = U[selected_components].T*selected_components_weight@alpha@Q.T

X = X + Z_norm@ZX*np.linalg.norm(X, axis=0)/10**(SNR_ZX/10)
y_true = X @ PQ_true

if random_type == 'rand':
	y = y_true + np.random.rand(*y_true.shape)/ np.sqrt(1/12)* np.linalg.norm(y_true, axis=0)/10**(SNR/10)
elif random_type == 'randn':
	y = y_true + np.random.randn(*y_true.shape)* np.linalg.norm(y_true, axis=0)/10**(SNR/10)
# y = y_true + np.random.randn(*y_true.shape)* np.sqrt(LA.norm(y_true,'fro')/10**(SNR/10)) 
y = y + Z_norm@ZY*np.linalg.norm(y_true, axis=0)/10**(SNR_Z/10)

# #Visualize results/evaluate resutlts
a = ["TL hippocampus R","TL hippocampus L","TL amygdala R","TL amygdala L","TL anterior temporal lobe medial part R","TL anterior temporal lobe medial part L","TL anterior temporal lobe lateral part R","TL anterior temporal lobe lateral part L","TL parahippocampal and ambient gyrus R","TL parahippocampal and ambient gyrus L","TL superior temporal gyrus middle part R","TL superior temporal gyrus middle part L","TL middle and inferior temporal gyrus R","TL middle and inferior temporal gyrus L","TL fusiform gyrus R","TL fusiform gyrus L","cerebellum R","cerebellum L","brainstem excluding substantia nigra","insula posterior long gyrus L","insula posterior long gyrus R","OL lateral remainder occipital lobe L","OL lateral remainder occipital lobe R","CG anterior cingulate gyrus L","CG anterior cingulate gyrus R","CG posterior cingulate gyrus L","CG posterior cingulate gyrus R","FL middle frontal gyrus L","FL middle frontal gyrus R","TL posterior temporal lobe L","TL posterior temporal lobe R","PL angular gyrus L","PL angular gyrus R","caudate nucleus L","caudate nucleus R","nucleus accumbens L","nucleus accumbens R","putamen L","putamen R","thalamus L","thalamus R","pallidum L","pallidum R","corpus callosum","Lateral ventricle excluding temporal horn R","Lateral ventricle excluding temporal horn L","Lateral ventricle temporal horn R","Lateral ventricle temporal horn L","Third ventricle","FL precentral gyrus L","FL precentral gyrus R","FL straight gyrus L","FL straight gyrus R","FL anterior orbital gyrus L","FL anterior orbital gyrus R","FL inferior frontal gyrus L","FL inferior frontal gyrus R","FL superior frontal gyrus L","FL superior frontal gyrus R","PL postcentral gyrus L","PL postcentral gyrus R","PL superior parietal gyrus L","PL superior parietal gyrus R","OL lingual gyrus L","OL lingual gyrus R","OL cuneus L","OL cuneus R","FL medial orbital gyrus L","FL medial orbital gyrus R","FL lateral orbital gyrus L","FL lateral orbital gyrus R","FL posterior orbital gyrus L","FL posterior orbital gyrus R","substantia nigra L","substantia nigra R","FL subgenual frontal cortex L","FL subgenual frontal cortex R","FL subcallosal area L","FL subcallosal area R","FL pre-subgenual frontal cortex L","FL pre-subgenual frontal cortex R","TL superior temporal gyrus anterior part L","TL superior temporal gyrus anterior part R","PL supramarginal gyrus L","PL supramarginal gyrus R","insula anterior short gyrus L","insula anterior short gyrus R","insula middle short gyrus L","insula middle short gyrus R","insula posterior short gyrus L","insula posterior short gyrus R","insula anterior inferior cortex L","insula anterior inferior cortex R","insula anterior long gyrus L","insula anterior long gyrus R"]
b = ["lUnknown","rUnknown","lG_and_S_frontomargin","rG_and_S_frontomargin","lG_and_S_occipital_inf","rG_and_S_occipital_inf","lG_and_S_paracentral","rG_and_S_paracentral","lG_and_S_subcentral","rG_and_S_subcentral","lG_and_S_transv_frontopol","rG_and_S_transv_frontopol","lG_and_S_cingul-Ant","rG_and_S_cingul-Ant","lG_and_S_cingul-Mid-Ant","rG_and_S_cingul-Mid-Ant","lG_and_S_cingul-Mid-Post","rG_and_S_cingul-Mid-Post","lG_cingul-Post-dorsal","rG_cingul-Post-dorsal","lG_cingul-Post-ventral","rG_cingul-Post-ventral","lG_cuneus","rG_cuneus","lG_front_inf-Opercular","rG_front_inf-Opercular","lG_front_inf-Orbital","rG_front_inf-Orbital","lG_front_inf-Triangul","rG_front_inf-Triangul","lG_front_middle","rG_front_middle","lG_front_sup","rG_front_sup","lG_Ins_lg_and_S_cent_ins","rG_Ins_lg_and_S_cent_ins","lG_insular_short","rG_insular_short","lG_occipital_middle","rG_occipital_middle","lG_occipital_sup","rG_occipital_sup","lG_oc-temp_lat-fusifor","rG_oc-temp_lat-fusifor","lG_oc-temp_med-Lingual","rG_oc-temp_med-Lingual","lG_oc-temp_med-Parahip","rG_oc-temp_med-Parahip","lG_orbital","rG_orbital","lG_pariet_inf-Angular","rG_pariet_inf-Angular","lG_pariet_inf-Supramar","rG_pariet_inf-Supramar","lG_parietal_sup","rG_parietal_sup","lG_postcentral","rG_postcentral","lG_precentral","rG_precentral","lG_precuneus","rG_precuneus","lG_rectus","rG_rectus","lG_subcallosal","rG_subcallosal","lG_temp_sup-G_T_transv","rG_temp_sup-G_T_transv","lG_temp_sup-Lateral","rG_temp_sup-Lateral","lG_temp_sup-Plan_polar","rG_temp_sup-Plan_polar","lG_temp_sup-Plan_tempo","rG_temp_sup-Plan_tempo","lG_temporal_inf","rG_temporal_inf","lG_temporal_middle","rG_temporal_middle","lLat_Fis-ant-Horizont","rLat_Fis-ant-Horizont","lLat_Fis-ant-Vertical","rLat_Fis-ant-Vertical","lLat_Fis-post","rLat_Fis-post","lMedial_wall","rMedial_wall","lPole_occipital","rPole_occipital","lPole_temporal","rPole_temporal","lS_calcarine","rS_calcarine","lS_central","rS_central","lS_cingul-Marginalis","rS_cingul-Marginalis","lS_circular_insula_ant","rS_circular_insula_ant","lS_circular_insula_inf","rS_circular_insula_inf","lS_circular_insula_sup","rS_circular_insula_sup","lS_collat_transv_ant","rS_collat_transv_ant","lS_collat_transv_post","rS_collat_transv_post","lS_front_inf","rS_front_inf","lS_front_middle","rS_front_middle","lS_front_sup","rS_front_sup","lS_interm_prim-Jensen","rS_interm_prim-Jensen","lS_intrapariet_and_P_trans","rS_intrapariet_and_P_trans","lS_oc_middle_and_Lunatus","rS_oc_middle_and_Lunatus","lS_oc_sup_and_transversal","rS_oc_sup_and_transversal","lS_occipital_ant","rS_occipital_ant","lS_oc-temp_lat","rS_oc-temp_lat","lS_oc-temp_med_and_Lingual","rS_oc-temp_med_and_Lingual","lS_orbital_lateral","rS_orbital_lateral","lS_orbital_med-olfact","rS_orbital_med-olfact","lS_orbital-H_Shaped","rS_orbital-H_Shaped","lS_parieto_occipital","rS_parieto_occipital","lS_pericallosal","rS_pericallosal","lS_postcentral","rS_postcentral","lS_precentral-inf-part","rS_precentral-inf-part","lS_precentral-sup-part","rS_precentral-sup-part","lS_suborbital","rS_suborbital","lS_subparietal","rS_subparietal","lS_temporal_inf","rS_temporal_inf","lS_temporal_sup","rS_temporal_sup","lS_temporal_transverse","rS_temporal_transverse"]
outcome_name = ['CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
       'RAVLT_learning', 'Ventricles']
df_X = pd.DataFrame(np.array(X),columns=[b[x] for x in list(range(X.shape[1]))])
# df_X['idx'] = df_X.index

df_y = pd.DataFrame(np.array(y),columns=[outcome_name[x] for x in list(range(y.shape[1]))])
# df_y['idx'] = df_y.index

df_Z = pd.DataFrame(np.array(Z),columns=[f'Z{x}' for x in list(range(Z.shape[1]))])
# df_Z['idx'] = df_Z.index

df_simulation = pd.concat([df_X,df_y,df_Z],axis=1)
df_simulation['Z0'] = df_simulation['Z0'].map({i: ["Female", "Male"][i] for i in range(2)})


uploaded_simulation = st.file_uploader("Upload simulation file")
# df_simulation = pd.read_csv('simulation_train_volume.csv')
if uploaded_simulation is not None:
	df_simulation = pd.read_csv(uploaded_simulation)
	 
ROIs_simulation = st.multiselect("Choose the simulation region", surface_atlas, surface_atlas[5:10])
if cofounder_type_code == 0:
	step = 30
	overlap = 0
	chart_surface_simulation = alt.Chart(df_simulation,title=f'Cortical thickness at region against age').transform_bin(
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

else:
	chart_surface_simulation = alt.Chart(df_simulation,width=40).mark_area(orient='horizontal').transform_fold(
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
	chart_surface_simulation.interactive(), use_container_width=False
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


# ========================================================= xxxx code simulation

from sklearn.utils import resample
sample_size = int(0.7*N)
indicies = list(range(N))
boot_indicies = resample(indicies, replace=True, n_samples=sample_size, random_state=1)
oob_indicies = [x for x in indicies if x not in boot_indicies]
X_train = X[boot_indicies]
X_test = X[oob_indicies]
y_train = y[boot_indicies]
y_test = y[oob_indicies]
Z_train = Z[boot_indicies]
Z_test = Z[oob_indicies]
# Normalization before regression
from sklearn import preprocessing
x_scaler = True
y_scaler = True
z_scaler = True
residual_scaler = True
use_lib = True 


if x_scaler:
    scalerx = preprocessing.StandardScaler().fit(X_train)
    X_train = scalerx.transform(X_train)
    X_test = scalerx.transform(X_test)

if y_scaler:
    scalery = preprocessing.StandardScaler().fit(y_train)
    y_train = scalery.transform(y_train)
    y_test = scalery.transform(y_test)

if z_scaler:
    scalerz = preprocessing.StandardScaler().fit(Z_train)
    Z_train = scalerz.transform(Z_train)
    Z_test = scalerz.transform(Z_test)

import matplotlib.pyplot as plt
from scipy.stats import t 
import sklearn
def statistic_regression(y_test,y_pred,savefile='final_regression',show_plot=False):
    r = [] # 2x2 sym matrix
    rs = [] 
    MSEs  = []
    residual_r = []
    p_values = []
    y_pred = np.array(y_pred)
    for i in range(y_test.shape[1]):
        r.append(np.corrcoef(y_test[:,i], y_pred[:,i]))
        rs.append(r[i][0,1])

        #Calculate pvalue
        dof = y_test.shape[0]
        t_stat = r[i][0,1]/ np.sqrt(1 - r[i][0,1]**2)* np.sqrt(dof)
        p_value = 2*(t.cdf(-abs(t_stat), dof))
        p_values.append(p_value)
        MSEs.append(mean_squared_error(y_test[:,i],y_pred[:,i]))
    results = {'r':rs, 'MSE':MSEs, 'y_test':y_test,'y_pred':y_pred}
    
    
    return results

### H_hat(z->y) and calculate resisual part
  ############################### Residual ######################################
import os
import copy
import matplotlib    
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression


matplotlib.style.use('ggplot')
outputs = f"results/test"
cofounder_method = 'LR'
if not os.path.exists(outputs):
    os.makedirs(outputs)

#Predict directly
n_components = 5





    
if cofounder_method == "PLS":        
    cofounder_model = PLSRegression(n_components=n_components)
elif cofounder_method == "PCR":
    cofounder_model = make_pipeline(PCA(n_components=n_components), LinearRegression())
elif cofounder_method == "LR":
    cofounder_model = LinearRegression()
    
    
cofounder_model.fit(Z_train, y_train)
reg_zy_train = copy.copy(cofounder_model)
beta_yz_train = reg_zy_train.coef_.T
# y_train_es = reg_zy_train.predict(Z_train)
y_train_es = Z_train@beta_yz_train
y_residuals_train = y_train - y_train_es


cofounder_model.fit(Z_train, X_train)
reg_zx_train = copy.copy(cofounder_model)
beta_zx_train = reg_zx_train.coef_.T
# X_train_es = reg_zx_train.predict(Z_train)
X_train_es = Z_train@beta_zx_train
X_residuals_train = X_train - X_train_es

cofounder_model.fit(Z_test, y_test) #not know, using as GT
reg_zy = copy.copy(cofounder_model)
beta_yz_test = reg_zy.coef_.T
# y_test_es = reg_zy.predict(Z_test)
y_test_es = Z_test@beta_yz_test
y_residuals_test = y_test - y_test_es


cofounder_model.fit(Z_test, X_test)
reg_zx = copy.copy(cofounder_model)
beta_zx_test = reg_zx.coef_.T
# X_test_es = reg_zx.predict(Z_test)
X_test_es = Z_test@beta_zx_test
X_residuals_test = X_test - X_test_es

print("Trainning result z->x, z->y")
results_zx = statistic_regression(X_train[:,:12],np.array(X_train_es)[:,:12],savefile=os.path.join(outputs,'zx_regression'),show_plot=True)
print(f"MSE_xz_train = {mean_squared_error(X_train,X_train_es)}" ,f' r = {results_zx["r"]} ')

results_zy = statistic_regression(y_train,np.array(y_train_es),savefile=os.path.join(outputs,'zy_regression'),show_plot=True)
print(f"MSE_yz_train= {mean_squared_error(y_train,y_train_es)}", f' r = {results_zx["r"]} ')
method = 'PLS'
if residual_scaler:
    scaler = preprocessing.StandardScaler().fit(X_residuals_train)
    X_residuals_train = scaler.transform(X_residuals_train)
    X_residuals_test = scaler.transform(X_residuals_test)

############################################ Predict ############################
if method == "PLS":        
    residual_model = PLSRegression(n_components=n_components)
elif method == "PCR":
    residual_model = make_pipeline(PCA(n_components=n_components), LinearRegression())
elif method == "LR":
    residual_model = LinearRegression()
residual_model.fit(X_residuals_train, y_residuals_train)


#Save results
if use_lib:
    if method == "PLS":
        T = residual_model.x_scores_
        P = residual_model.x_loadings_ #projection

        U = residual_model.y_scores_
        Q = residual_model.y_loadings_
        PQ = residual_model.coef_

#         sio.savemat(os.path.join(outputs,f'PSL_loadings.mat'), load)

    elif method == "PCR":
        P = residual_model['pca'].components_.T
        PQ = np.zeros([P.shape[0],len(outcomes)])

#         sio.savemat(os.path.join(outputs,f'PCR_loadings.mat'), load)
    elif method == "LR":
        PQ = residual_model.coef_.T
        P = np.zeros([PQ.shape[0],n_components])
#         sio.savemat(os.path.join(outputs,f'LR_loadings.mat'), load)
result_dict = {'P': P, 'PQ':PQ, 'method':method}
#Predicting
#directly from X->Z
if method == "PLS":  
    model_PLS_xy = PLSRegression(n_components=n_components)
    model_PLS_xy.fit(X_train, y_train)
    y_pred_xy = model_PLS_xy.predict(X_test) # y_test
elif method == "PCR":  
    model_PCR_xy = make_pipeline(PCA(n_components=n_components), LinearRegression())
    model_PCR_xy.fit(X_train, y_train)
    y_pred_xy = model_PCR_xy.predict(X_test) # y_test
elif method == "LR":  
    model_LR_xy = LinearRegression()
    model_LR_xy.fit(X_train, y_train)
    y_pred_xy = model_LR_xy.predict(X_test) # y_test


#from both X and Z
y_residual = residual_model.predict(X_residuals_test) # y_residuals_test
zy = reg_zy_train.predict(Z_test) # y_test

y_pred = zy + y_residual # y_test


results_test_xy = statistic_regression(y_test,np.array(y_pred_xy),savefile=os.path.join(outputs,'regress_xy'),show_plot=True)
# results_test_residual,fig_scatter_test_residual,fig_bar_test_residual = statistic_regression(y_residuals_test,np.array(y_residual),savefile=os.path.join(outputs,'residual_test'),show_plot=True)
# results_test_zy,fig_scatter_test_zy,fig_bar_test_zy = statistic_regression(y_test,np.array(zy),savefile=os.path.join(outputs,'zy_test'),show_plot=True)
results_test_final = statistic_regression(y_test,np.array(y_pred),savefile=os.path.join(outputs,'combine_reg_test'),show_plot=True)
# results_test_residual2,fig_scatter_test_residual,fig_bar_test_residual = statistic_regression(y_test,np.array(y_residual),savefile=os.path.join(outputs,'residual_test'),show_plot=False)

df_xy = pd.DataFrame({'MSE':results_test_xy['MSE'],'r':results_test_xy['r'], 'outcomes':outcome_name, 'regression':['xy']*len(outcome_name)})
df_re_xyz = pd.DataFrame({'MSE':results_test_final['MSE'],'r':results_test_final['r'], 'outcomes':outcome_name, 'regression':['re_xyz']*len(outcome_name)})

df = pd.concat([df_xy,df_re_xyz],axis=0)
df['regression_code'] = df['regression'].map({"xy":0, "re_xyz" :1})

chart_r = alt.Chart(df,width=60,title='').mark_bar().encode(
    x=alt.X('regression_code:N',title=None,axis=None),
    y=alt.Y('r', title='r'),
    color='regression:N',
    column='outcomes:N'
).configure_header(
    titleOrient='bottom', 
    labelOrient='bottom',labelAngle=90,labelAnchor='middle',labelAlign='center',labelPadding=50
)
chart_rdf_xy = pd.DataFrame({'MSE':results_test_xy['MSE'],'r':results_test_xy['r'], 'outcomes':outcome_name, 'regression':['xy']*len(outcome_name)})
df_re_xyz = pd.DataFrame({'MSE':results_test_final['MSE'],'r':results_test_final['r'], 'outcomes':outcome_name, 'regression':['re_xyz']*len(outcome_name)})

df = pd.concat([df_xy,df_re_xyz],axis=0)
df['regression_code'] = df['regression'].map({"xy":0, "re_xyz" :1})


chart_r = alt.Chart(df,width=60,title='').mark_bar().encode(
    x=alt.X('regression_code:N',title=None,axis=None),
    y=alt.Y('r', title='r'),
    color='regression:N',
    column=alt.Column('outcomes:N',title=None)
).configure_header(
    titleOrient='bottom', 
    labelOrient='bottom',labelAngle=90,labelAnchor='middle',labelAlign='center',labelPadding=50
)

chart_MSE = alt.Chart(df,width=60,title='').mark_bar().encode(
    x=alt.X('regression_code:N',title=None,axis=None),
    y=alt.Y('MSE', title='r'),
    color='regression:N',
    column=alt.Column('outcomes:N',title=None)
).configure_header(
    titleOrient='bottom', 
    labelOrient='bottom',labelAngle=90,labelAnchor='middle',labelAlign='center',labelPadding=50
)


st.header("Simulation results")
st.altair_chart(
	chart_r.interactive(), use_container_width=False
)
st.altair_chart(
	chart_MSE.interactive(), use_container_width=False
)

st.title('Feature visualization')
if st.button("Run TSNE"):
	from sklearn.manifold import TSNE
	df_Z = pd.DataFrame(np.array(Z[boot_indicies]),columns=[f'Z{x}' for x in list(range(Z[boot_indicies].shape[1]))])
	feature_pls = X_train@model_PLS_xy.x_loadings_
	feature_pls = TSNE(n_components=2, learning_rate='auto',
					init='random').fit_transform(feature_pls)
	feature_repls = X_train@P
	feature_repls = TSNE(n_components=2, learning_rate='auto',
					init='random').fit_transform(feature_repls)
	df_Z['y_embedded_0'] = feature_pls[:,0]
	df_Z['y_embedded_1'] = feature_pls[:,1]

	df_Z['y_embedded_re_0'] = feature_repls[:,0]
	df_Z['y_embedded_re_1'] = feature_repls[:,1]
	chart_feature = alt.Chart(df_Z,width=350,title='re-PLS').mark_circle().encode(
		y=alt.Y('y_embedded_re_0:Q',title='feature0'),
		x=alt.X('y_embedded_re_1:Q',title="feature1"),
		color=alt.Color('Z1', bin=True,scale=alt.Scale(scheme='dark2'),title="Age"),
	#     shape=alt.Shape('AGE', bin=True)
	)|alt.Chart(df_Z,width=350,title='PLS').mark_circle().encode(
		y=alt.Y('y_embedded_0:Q',title='feature0'),
		x=alt.X('y_embedded_1:Q',title="feature1"),
		color=alt.Color('Z1', bin=True,scale=alt.Scale(scheme='dark2'),title="Age"),
	#     shape=alt.Shape('AGE', bin=True)
	)   
	st.altair_chart(
		chart_feature, use_container_width=False
	)

st.title('PQ comparision')
st.subheader('Correlation coefficient')
import matplotlib.pyplot as plt

fig = plt.figure()
plt.title(f'PQ{PQ_idx}(True-rePLS-PLS)')
PQ_PLS = model_PLS_xy.coef_
PQ_i = np.vstack([PQ_true[:,PQ_idx],PQ[:,PQ_idx],PQ_PLS[:,PQ_idx]])
cov_PQ_i = np.corrcoef(PQ_i)
plt.imshow(cov_PQ_i, extent=[0, 3, 0,3])
plt.colorbar()
for i in range(3):
    for j in range(3):
#         print(cov_PQ_i[i, j].round(5)," ")
        text = plt.text(i+0.5, 3-(j+0.5), cov_PQ_i[i, j].round(5),
                       va="center",ha="center", color="w")
#     print("\n")
st.pyplot(fig)


st.subheader('Angle')
def angle(x,y):
    inner_product = np.dot(x/LA.norm(x),y/LA.norm(y))
    angle1 = np.arccos(np.clip(inner_product, -1.0, 1.0))
    angle1 = np.rad2deg(angle1)
#     print(angle1)
    return angle1.round(2) 

angles = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        angles[i,j] = angle(PQ_true[:,i],PQ[:,j])
        
# print(angles)
fig_angle1 = plt.figure(figsize=(15,15))
plt.title(f'PQ_true vs PQ_rePLS')
plt.imshow(angles, extent=[0, 8, 0,8])
plt.colorbar()
for i in range(8):
    for j in range(8):
#         print(angles[i, j].round(5)," ")
        text = plt.text(i+0.5, 8-(j+0.5), angles[i, j].round(1), color="w")

# st.pyplot(fig_angle1)
angles = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        angles[i,j] = angle(PQ_true[:,i],PQ_PLS[:,j])
        
# print(angles)
fig_angle2 = plt.figure(figsize=(15,15))
plt.title(f'PQ_true vs PQ_PLS')
plt.imshow(angles, extent=[0, 8, 0,8])
plt.colorbar()
for i in range(8):
    for j in range(8):
#         print(angles[i, j].round(5)," ")
        text = plt.text(i+0.5, 8-(j+0.5), angles[i, j].round(1), color="w")
# plt.show()


col_angle1,col_angle2 = st.columns(2)
with col_angle1:
	st.pyplot(fig_angle1)
with col_angle2:
	st.pyplot(fig_angle2)

st.subheader('Boostrap and Confident interval')
if st.button("Run 32 bootstrap"):
	df_PQ = pd.DataFrame([],columns= [f'PQ{x}' for x in range(8)])
	for random_idx in range(32):
		print(random_idx,"==========================")
		boot_indicies = resample(indicies, replace=True, n_samples=sample_size, random_state=random_idx)
		oob_indicies = [x for x in indicies if x not in boot_indicies]
		X_train = X[boot_indicies]
		X_test = X[oob_indicies]
		y_train = y[boot_indicies]
		y_test = y[oob_indicies]
		Z_train = Z[boot_indicies]
		Z_test = Z[oob_indicies]
		
		if x_scaler:
			scalerx = preprocessing.StandardScaler().fit(X_train)
			X_train = scalerx.transform(X_train)
			X_test = scalerx.transform(X_test)

		if y_scaler:
			scalery = preprocessing.StandardScaler().fit(y_train)
			y_train = scalery.transform(y_train)
			y_test = scalery.transform(y_test)

		if z_scaler:
			scalerz = preprocessing.StandardScaler().fit(Z_train)
			Z_train = scalerz.transform(Z_train)
			Z_test = scalerz.transform(Z_test)
		cofounder_method = 'LR'
		if not os.path.exists(outputs):
			os.makedirs(outputs)

		#Predict directly
		n_components = 5

		if cofounder_method == "PLS":        
			cofounder_model = PLSRegression(n_components=n_components)
		elif cofounder_method == "PCR":
			cofounder_model = make_pipeline(PCA(n_components=n_components), LinearRegression())
		elif cofounder_method == "LR":
			cofounder_model = LinearRegression()


		cofounder_model.fit(Z_train, y_train)
		reg_zy_train = copy.copy(cofounder_model)
		beta_yz_train = reg_zy_train.coef_.T
		y_train_es = reg_zy_train.predict(Z_train)
		# y_train_es = Z_train@beta_yz_train
		y_residuals_train = y_train - y_train_es


		cofounder_model.fit(Z_train, X_train)
		reg_zx_train = copy.copy(cofounder_model)
		beta_zx_train = reg_zx_train.coef_.T
		X_train_es = reg_zx_train.predict(Z_train)
		# X_train_es = Z_train@beta_zx_train
		X_residuals_train = X_train - X_train_es

		cofounder_model.fit(Z_test, y_test) #not know, using as GT
		reg_zy = copy.copy(cofounder_model)
		beta_yz_test = reg_zy.coef_.T
		y_test_es = reg_zy.predict(Z_test)
		# y_test_es = Z_test@beta_yz_test
		y_residuals_test = y_test - y_test_es


		cofounder_model.fit(Z_test, X_test)
		reg_zx = copy.copy(cofounder_model)
		beta_zx_test = reg_zx.coef_.T
		X_test_es = reg_zx.predict(Z_test)
		# X_test_es = Z_test@beta_zx_test
		X_residuals_test = X_test - X_test_es

		print("Trainning result z->x, z->y")
		results_zx = statistic_regression(X_train[:,:12],np.array(X_train_es)[:,:12],savefile=os.path.join(outputs,'zx_regression'),show_plot=True)
		print(f"MSE_xz_train = {mean_squared_error(X_train,X_train_es)}" ,f' r = {results_zx["r"]} ')

		results_zy = statistic_regression(y_train,np.array(y_train_es),savefile=os.path.join(outputs,'zy_regression'),show_plot=True)
		print(f"MSE_yz_train= {mean_squared_error(y_train,y_train_es)}", f' r = {results_zx["r"]} ')
		method = 'PLS'
		if residual_scaler:
			scaler = preprocessing.StandardScaler().fit(X_residuals_train)
			X_residuals_train = scaler.transform(X_residuals_train)
			X_residuals_test = scaler.transform(X_residuals_test)

		############################################ Predict ############################
		if method == "PLS":        
			residual_model = PLSRegression(n_components=n_components)
		elif method == "PCR":
			residual_model = make_pipeline(PCA(n_components=n_components), LinearRegression())
		elif method == "LR":
			residual_model = LinearRegression()
		residual_model.fit(X_residuals_train, y_residuals_train)


		#Save results
		if use_lib:
			if method == "PLS":
				T = residual_model.x_scores_
				P = residual_model.x_loadings_ #projection

				U = residual_model.y_scores_
				Q = residual_model.y_loadings_
				PQ = residual_model.coef_

		#         sio.savemat(os.path.join(outputs,f'PSL_loadings.mat'), load)

			elif method == "PCR":
				P = residual_model['pca'].components_.T
				PQ = np.zeros([P.shape[0],len(outcomes)])

		#         sio.savemat(os.path.join(outputs,f'PCR_loadings.mat'), load)
			elif method == "LR":
				PQ = residual_model.coef_.T
				P = np.zeros([PQ.shape[0],n_components])
		#         sio.savemat(os.path.join(outputs,f'LR_loadings.mat'), load)
		PQ = PQ/ np.linalg.norm(PQ, axis=0)
		df_PQ_temp = pd.DataFrame(PQ,columns= [f'PQ{x}' for x in range(8)])
		df_PQ_temp['idx'] = df_PQ_temp.index
		df_PQ_temp['bootstrap_idx'] = random_idx
		df_PQ = pd.concat([df_PQ,df_PQ_temp],axis=0)
	PQ_true = PQ_true/ np.linalg.norm(PQ_true, axis=0)
	df_PQ_true = pd.DataFrame(PQ_true,columns= [f'PQ{x}' for x in range(8)])
	df_PQ_true['idx'] = df_PQ_true.index

	chart_CI = alt.Chart(df_PQ_true).mark_line().encode(
		x=alt.X('idx:T'),
		y=alt.Y(f'PQ{PQ_idx}:Q'),
		color=alt.value("#FF0000")
	)+alt.Chart(df_PQ).mark_errorband(extent='ci').encode(
		x=alt.X('idx:T'),
		y=alt.Y(f'mean(PQ{PQ_idx}):Q')
	)

	st.altair_chart(chart_CI)