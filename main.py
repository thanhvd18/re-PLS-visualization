import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import altair as alt
from altair import datum
import subprocess

alt.data_transformers.disable_max_rows()
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
    evaluation = st.radio( "Evaluation on",("Bootstrapping",'New testing set'),index=0)  #Bootstrapping, New testing set
    basic_type = st.radio( "Basic type",('svd-orthonomal', 'NMF-nonnegative', 'pca-orthonomal','sparse-pca'),index=2)  #
    gender_w = st.number_input('Gender weight (%)', value=10)
    age_w = st.number_input('Age weight (%)', value=90)
    # use_violinplot = st.checkbox('Violin plot')
    st.title('Gender')
    gender_ratio = st.number_input('gender_ratio', value=0.2)
    gender_ratio_test = st.number_input('gender_ratio_test', value=0.2)
    # age_col1,age_col2,age_col3 = st.columns(3)
    # with age_col1:
    st.title('Age')
    w_age1 = st.number_input('Weight Age group 1', value=20)
    # with age_col2:
    w_age2 = st.number_input('Weight Age group 2', value=20)
    # with age_col3:
    w_age3 = st.number_input('Weight Age group 3', value=60)	
    w_age1_test = st.number_input('Weight Age test group 1', value=33)
    # with age_col2:
    w_age2_test = st.number_input('Weight Age test group 2', value=33)
    # with age_col3:
    w_age3_test = st.number_input('Weight Age test group 3', value=34)	
    st.title('Noise')
    SNR = st.number_input('y SNR', value=15)
    SNR_Z = st.number_input('Zy SNR', value=500)
    SNR_ZX = st.number_input('ZX SNR', value=500)
    with st.expander("Other parameters"):
        random_type = st.radio('Random type', ('rand','randn'),index=1)
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
        color=alt.Color('Z0:N',scale=alt.Scale(domain=['Male', 'Female'],range=['#DB7093', '#2E8B57'])),#legend=None
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
        color=alt.Color('Z0:N', scale=alt.Scale(domain=['Male', 'Female'],range=['#DB7093', '#2E8B57'])),#legend=None
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
        color=alt.Color('Z0:N', scale=alt.Scale(domain=['Male', 'Female'],range=['#DB7093', '#2E8B57'])),#legend=None
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


from pathlib import Path
import streamlit as st

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

outputs_markdown = read_markdown_file("outputs.md")
with st.expander("See explanation outputs"):
    st.markdown(outputs_markdown,unsafe_allow_html=True
    )
     
     

# =================================================================== SIMULATION=============================================
st.header("Simulation")
import numpy as np
from sklearn.decomposition import NMF
from numpy import linalg as LA
import pandas as pd
import copy
import random



rng = np.random.RandomState(0)
N = 700
X_component = 20 #max features/region: 152
K = 5
selected_components = [3,1,5,8,10]
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
random.shuffle(Z[:,0])

#age
age_group_weight = np.array([w_age1, w_age2, w_age3])/100 #40-60-80 age group percentage
age_group_weight_norm = np.array([33, 33, 34])/100 #40-60-80 age group percentage


age_group_sample = [int(x*N) for x in age_group_weight]
age_group_sample_norm = [int(x*N) for x in age_group_weight_norm]


Z[:,1] = np.random.randint(30, 50, size=(age_group_sample[0],)).tolist() + np.random.randint(51, 70, size=(age_group_sample[1],)).tolist()+ np.random.randint(71, 90, size=(age_group_sample[2],)).tolist()
random.shuffle(Z[:,1])
alpha = np.eye(K)

Z_weight = np.array([gender_w,age_w])/100 #40-60-80 age group percentage
Z_norm = copy.deepcopy(Z)
# Z_norm[:,0] = [0]*int(N*0.5) + [1]*(N-int(N*0.5)) 
# Z_norm[:,0] = Z[:,0]/LA.norm(Z[:,0])
random.shuffle(Z_norm[:,0])
Z_norm[:,1] = np.random.randint(30, 50, size=(age_group_sample_norm[0],)).tolist() + np.random.randint(51, 70, size=
(age_group_sample_norm[1],)).tolist()+ np.random.randint(71, 90, size=(age_group_sample_norm[2],)).tolist()
random.shuffle(Z_norm[:,1])                                                                                                       
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

from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
X = 2.26+0.23*np.random.randn(N, I)
# basic_type ='svd-orthonomal'
basic_type = 'svd-orthonomal'
if basic_type =='svd-orthonomal':
    U, S, V = np.linalg.svd(X, full_matrices=False)
    PQ_true = V[:,selected_components]*selected_components_weight@alpha@Q.T
elif basic_type =='NMF-nonnegative':
    model = NMF(n_components=I, init='random', random_state=0)
    W = model.fit_transform(X)
# 	U = U_true + Z_norm@ZPQ* np.sqrt(np.linalg.norm(U_true, axis=0)/10**(SNR_ZPQ/10))
    H = model.components_
#     X = W@H
    V=H
    PQ_true = H[selected_components,:].T*selected_components_weight@alpha@Q.T
elif basic_type =='pca-orthonomal':
    pca = PCA(n_components=I)
    pca.fit(X)
    PQ_true =pca.components_[selected_components,:].T*selected_components_weight@alpha@Q.T
elif basic_type =='sparse-pca':
    sparse_pca = SparsePCA(n_components=I)
    sparse_pca.fit(X)
    PQ_true =sparse_pca.components_[selected_components,:].T*selected_components_weight@alpha@Q.T





X_true = X 
true_ZX_p = np.sqrt(np.linalg.norm(X, axis=0)/10**(SNR_ZX/10))
X = X_true + Z_norm@ZX*np.sqrt(np.linalg.norm(X, axis=0)/10**(SNR_ZX/10))
y_true = X @ PQ_true

if random_type == 'rand':
    y = y_true + np.random.rand(*y_true.shape)/ np.sqrt(1/12)* np.sqrt(np.linalg.norm(y_true, axis=0)/10**(SNR/10))
elif random_type == 'randn':
    y = y_true + np.random.randn(*y_true.shape)*np.sqrt( np.linalg.norm(y_true, axis=0)/10**(SNR/10))
# y = y_true + np.random.randn(*y_true.shape)* np.sqrt(LA.norm(y_true,'fro')/10**(SNR/10)) 
y = y + Z_norm@ZY*np.sqrt(np.linalg.norm(y_true, axis=0)/10**(SNR_Z/10))




# #Visualize results/evaluate resutlts
# a = ["TL hippocampus R","TL hippocampus L","TL amygdala R","TL amygdala L","TL anterior temporal lobe medial part R","TL anterior temporal lobe medial part L","TL anterior temporal lobe lateral part R","TL anterior temporal lobe lateral part L","TL parahippocampal and ambient gyrus R","TL parahippocampal and ambient gyrus L","TL superior temporal gyrus middle part R","TL superior temporal gyrus middle part L","TL middle and inferior temporal gyrus R","TL middle and inferior temporal gyrus L","TL fusiform gyrus R","TL fusiform gyrus L","cerebellum R","cerebellum L","brainstem excluding substantia nigra","insula posterior long gyrus L","insula posterior long gyrus R","OL lateral remainder occipital lobe L","OL lateral remainder occipital lobe R","CG anterior cingulate gyrus L","CG anterior cingulate gyrus R","CG posterior cingulate gyrus L","CG posterior cingulate gyrus R","FL middle frontal gyrus L","FL middle frontal gyrus R","TL posterior temporal lobe L","TL posterior temporal lobe R","PL angular gyrus L","PL angular gyrus R","caudate nucleus L","caudate nucleus R","nucleus accumbens L","nucleus accumbens R","putamen L","putamen R","thalamus L","thalamus R","pallidum L","pallidum R","corpus callosum","Lateral ventricle excluding temporal horn R","Lateral ventricle excluding temporal horn L","Lateral ventricle temporal horn R","Lateral ventricle temporal horn L","Third ventricle","FL precentral gyrus L","FL precentral gyrus R","FL straight gyrus L","FL straight gyrus R","FL anterior orbital gyrus L","FL anterior orbital gyrus R","FL inferior frontal gyrus L","FL inferior frontal gyrus R","FL superior frontal gyrus L","FL superior frontal gyrus R","PL postcentral gyrus L","PL postcentral gyrus R","PL superior parietal gyrus L","PL superior parietal gyrus R","OL lingual gyrus L","OL lingual gyrus R","OL cuneus L","OL cuneus R","FL medial orbital gyrus L","FL medial orbital gyrus R","FL lateral orbital gyrus L","FL lateral orbital gyrus R","FL posterior orbital gyrus L","FL posterior orbital gyrus R","substantia nigra L","substantia nigra R","FL subgenual frontal cortex L","FL subgenual frontal cortex R","FL subcallosal area L","FL subcallosal area R","FL pre-subgenual frontal cortex L","FL pre-subgenual frontal cortex R","TL superior temporal gyrus anterior part L","TL superior temporal gyrus anterior part R","PL supramarginal gyrus L","PL supramarginal gyrus R","insula anterior short gyrus L","insula anterior short gyrus R","insula middle short gyrus L","insula middle short gyrus R","insula posterior short gyrus L","insula posterior short gyrus R","insula anterior inferior cortex L","insula anterior inferior cortex R","insula anterior long gyrus L","insula anterior long gyrus R"]
# b = ["lUnknown","rUnknown","lG_and_S_frontomargin","rG_and_S_frontomargin","lG_and_S_occipital_inf","rG_and_S_occipital_inf","lG_and_S_paracentral","rG_and_S_paracentral","lG_and_S_subcentral","rG_and_S_subcentral","lG_and_S_transv_frontopol","rG_and_S_transv_frontopol","lG_and_S_cingul-Ant","rG_and_S_cingul-Ant","lG_and_S_cingul-Mid-Ant","rG_and_S_cingul-Mid-Ant","lG_and_S_cingul-Mid-Post","rG_and_S_cingul-Mid-Post","lG_cingul-Post-dorsal","rG_cingul-Post-dorsal","lG_cingul-Post-ventral","rG_cingul-Post-ventral","lG_cuneus","rG_cuneus","lG_front_inf-Opercular","rG_front_inf-Opercular","lG_front_inf-Orbital","rG_front_inf-Orbital","lG_front_inf-Triangul","rG_front_inf-Triangul","lG_front_middle","rG_front_middle","lG_front_sup","rG_front_sup","lG_Ins_lg_and_S_cent_ins","rG_Ins_lg_and_S_cent_ins","lG_insular_short","rG_insular_short","lG_occipital_middle","rG_occipital_middle","lG_occipital_sup","rG_occipital_sup","lG_oc-temp_lat-fusifor","rG_oc-temp_lat-fusifor","lG_oc-temp_med-Lingual","rG_oc-temp_med-Lingual","lG_oc-temp_med-Parahip","rG_oc-temp_med-Parahip","lG_orbital","rG_orbital","lG_pariet_inf-Angular","rG_pariet_inf-Angular","lG_pariet_inf-Supramar","rG_pariet_inf-Supramar","lG_parietal_sup","rG_parietal_sup","lG_postcentral","rG_postcentral","lG_precentral","rG_precentral","lG_precuneus","rG_precuneus","lG_rectus","rG_rectus","lG_subcallosal","rG_subcallosal","lG_temp_sup-G_T_transv","rG_temp_sup-G_T_transv","lG_temp_sup-Lateral","rG_temp_sup-Lateral","lG_temp_sup-Plan_polar","rG_temp_sup-Plan_polar","lG_temp_sup-Plan_tempo","rG_temp_sup-Plan_tempo","lG_temporal_inf","rG_temporal_inf","lG_temporal_middle","rG_temporal_middle","lLat_Fis-ant-Horizont","rLat_Fis-ant-Horizont","lLat_Fis-ant-Vertical","rLat_Fis-ant-Vertical","lLat_Fis-post","rLat_Fis-post","lMedial_wall","rMedial_wall","lPole_occipital","rPole_occipital","lPole_temporal","rPole_temporal","lS_calcarine","rS_calcarine","lS_central","rS_central","lS_cingul-Marginalis","rS_cingul-Marginalis","lS_circular_insula_ant","rS_circular_insula_ant","lS_circular_insula_inf","rS_circular_insula_inf","lS_circular_insula_sup","rS_circular_insula_sup","lS_collat_transv_ant","rS_collat_transv_ant","lS_collat_transv_post","rS_collat_transv_post","lS_front_inf","rS_front_inf","lS_front_middle","rS_front_middle","lS_front_sup","rS_front_sup","lS_interm_prim-Jensen","rS_interm_prim-Jensen","lS_intrapariet_and_P_trans","rS_intrapariet_and_P_trans","lS_oc_middle_and_Lunatus","rS_oc_middle_and_Lunatus","lS_oc_sup_and_transversal","rS_oc_sup_and_transversal","lS_occipital_ant","rS_occipital_ant","lS_oc-temp_lat","rS_oc-temp_lat","lS_oc-temp_med_and_Lingual","rS_oc-temp_med_and_Lingual","lS_orbital_lateral","rS_orbital_lateral","lS_orbital_med-olfact","rS_orbital_med-olfact","lS_orbital-H_Shaped","rS_orbital-H_Shaped","lS_parieto_occipital","rS_parieto_occipital","lS_pericallosal","rS_pericallosal","lS_postcentral","rS_postcentral","lS_precentral-inf-part","rS_precentral-inf-part","lS_precentral-sup-part","rS_precentral-sup-part","lS_suborbital","rS_suborbital","lS_subparietal","rS_subparietal","lS_temporal_inf","rS_temporal_inf","lS_temporal_sup","rS_temporal_sup","lS_temporal_transverse","rS_temporal_transverse"]
# outcome_name = ['CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
    #    'RAVLT_learning', 'Ventricles']
a = [f"region{i}" for i in range(95)]
b = [f"region{i}" for i in range(152)]
outcome_name = [f"output{i}" for i in range(8)]
df_X = pd.DataFrame(np.array(X),columns=[b[x] for x in list(range(X.shape[1]))])
# df_X['idx'] = df_X.index

df_y = pd.DataFrame(np.array(y),columns=[outcome_name[x] for x in list(range(y.shape[1]))])
# df_y['idx'] = df_y.index

df_Z = pd.DataFrame(np.array(Z),columns=[f'Z{x}' for x in list(range(Z.shape[1]))])
# df_Z['idx'] = df_Z.index

df_simulation = pd.concat([df_X,df_y,df_Z],axis=1)
df_simulation['Z0'] = df_simulation['Z0'].map({i: ["Female", "Male"][i] for i in range(2)})



ROIs_simulation = st.multiselect("Choose the simulation region", b, b[5:10])
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
        color=alt.Color('Z0:N', scale=alt.Scale(domain=['Male', 'Female'],range=['#DB7093', '#2E8B57'])),#legend=None
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
        color=alt.Color('Z0:N', scale=alt.Scale(domain=['Male', 'Female'],range=['#DB7093', '#2E8B57'])),#legend=None
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
).configure_range(
    category=['#DB7093', '#2E8B57']
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
if evaluation == "Bootstrapping":
    X_train = X[boot_indicies]
    X_test = X[oob_indicies]
    y_train = y[boot_indicies]
    y_test = y[oob_indicies]
    Z_train = Z[boot_indicies]
    Z_test = Z[oob_indicies]
    y_test = (X_test-Z_norm[oob_indicies,:]@ZX*true_ZX_p) @ PQ_true
elif evaluation == "New testing set": # Bootstrapping, New testing set
# Normalization before regression

    y_train = y
    X_train = X 
    Z_train = Z

    N_test = 300
    indicies = list(range(N))
    boot_indicies = resample(indicies, replace=True, n_samples=N_test, random_state=1)
    oob_indicies = [x for x in indicies if x not in boot_indicies]
    X_test_true = np.random.rand(N_test,I)@(np.random.rand(I,1)*np.eye(I))@ V[:,:Rx].T 
    Z_test = np.zeros((N_test,R))
    age_group_weight_norm = np.array([w_age1_test, w_age2_test, w_age3_test])/100 #40-60-80 age group percentage
    age_group_sample = [int(x*N_test) for x in age_group_weight]
    Z_test[:,0] = [0]*int(N_test*gender_ratio_test) + [1]*(N_test-int(N_test*gender_ratio_test)) 
    random.shuffle(Z_test[:,0])
    Z_test[:,1] = np.random.randint(30, 50, size=(age_group_sample[0],)).tolist() + np.random.randint(51, 70, size=
    (age_group_sample[1],)).tolist()+ np.random.randint(71, 90, size=(age_group_sample[2],)).tolist()
    random.shuffle(Z_test[:,1])
    X_test = X_test_true + Z_test@ZX*true_ZX_p
    y_test = X_test_true @ PQ_true
    

from sklearn import preprocessing
x_scaler = True
y_scaler = True
z_scaler = True
residual_scaler = True
use_lib = True 
Z_train_ = Z_train

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
from rePLS import Residual_regression
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

def evaluation_regression(y_test,y_pred):
    rs = [] 
    MSEs  = []
    p_values = []
    for i in range(y_test.shape[1]):
        r_matrix = np.corrcoef(np.array(y_test[:,i]), np.array(y_pred[:,i]))
        rs.append(r_matrix[0,1])

        #Calculate pvalue
        dof = y_test.shape[0]
        t_stat = r_matrix[0,1]/ np.sqrt(1 - r_matrix[0,1]**2)* np.sqrt(dof)
        p_value = 2*(t.cdf(-abs(t_stat), dof))
        p_values.append(p_value)
        MSEs.append(mean_squared_error(y_test[:,i],y_pred[:,i]))
    return rs,MSEs,p_values


n_components = 6
MLR = LinearRegression()
PCR = make_pipeline(PCA(n_components=n_components), LinearRegression())
PLS = PLSRegression(n_components=n_components)





rePLS = Residual_regression(Z=Z_train,name="rePLS",n_components=n_components)
rePCR = Residual_regression(Z=Z_train,name="rePCR",n_components=n_components)
reMLR = Residual_regression(Z=Z_train,name="reMLR",n_components=n_components)


pipelines = [MLR,PCR,PLS,reMLR,rePCR,rePLS]
pipeline_names = ["MLR","PCR","PLS","reMLR","rePCR","rePLS"]
outcome_name = ['CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
                'RAVLT_learning', 'Ventricles']
import pandas as pd
df_simulation = pd.DataFrame({"r":[], "MSE":[], "pvalue":[], "method":[]})
for i,pipe in enumerate(pipelines):

    pipe.fit(X_train, y_train)

for i,model in enumerate(pipelines):
    if i<3:
        print("{} Test MSE: {:.3f}".format(pipeline_names[i],mean_squared_error(y_test,model.predict(X_test))))
        r,MSE,p_value = evaluation_regression(y_test,np.array(model.predict(X_test)))
        result =  pd.DataFrame({"r":r, "MSE":MSE, "pvalue":p_value, "method":pipeline_names[i], "output":outcome_name})
        df_simulation = pd.concat([df_simulation,result],axis=0)
    else:
        print("{} Test MSE: {:.3f}".format(pipeline_names[i],mean_squared_error(y_test,model.predict(X_test,Z=Z_test))))   
        r,MSE,p_value = evaluation_regression(y_test,np.array(model.predict(X_test,Z=Z_test)))
        result =  pd.DataFrame({"r":r, "MSE":MSE, "pvalue":p_value, "method":pipeline_names[i],"output":outcome_name})
        df_simulation = pd.concat([df_simulation,result],axis=0)
# pipeline_dict = {"0":"MLR", "0.5",}
df_simulation["isRe"] = df_simulation.method.apply(lambda x: pipeline_names.index(x)%3 + int(pipeline_names.index(x)/3)*0.5 )

import altair as alt
from altair import datum
chart_r= alt.Chart(df_simulation,width=90).mark_bar().encode(
    x=alt.X('isRe:N',axis=None),
    y=alt.Y('r:Q'),
#     y2=alt.Y2('MSE:Q'),
    color=alt.Color('method:N',scale=alt.Scale(domain=['MLR','reMLR','PCR','rePCR','PLS','rePLS'],range= ['#9ecae9', '#4c78a8', '#ffbf79', '#f58518', '#88d27a', '#54a24b']), title="Method"),
#         color=alt.Color('method:N',scale=alt.Scale(scheme='tableau20')),
    column=alt.Column('output',title=None)
).configure_header(
titleOrient='bottom', labelOrient='bottom',labelAngle=00,labelAnchor='middle',labelAlign='center',labelPadding=50
).configure_view(
    stroke=None,
)

chart_MSE= alt.Chart(df_simulation,width=90).mark_bar().encode(
    x=alt.X('isRe:N',axis=None),
    y=alt.Y('MSE:Q'),
#     y2=alt.Y2('MSE:Q'),
    color=alt.Color('method:N',scale=alt.Scale(domain=['MLR','reMLR','PCR','rePCR','PLS','rePLS'],range= ['#9ecae9', '#4c78a8', '#ffbf79', '#f58518', '#88d27a', '#54a24b']), title="Method"),
#         color=alt.Color('method:N',scale=alt.Scale(scheme='tableau20')),
    column=alt.Column('output',title=None)
).configure_header(
titleOrient='bottom', labelOrient='bottom',labelAngle=00,labelAnchor='middle',labelAlign='center',labelPadding=50
).configure_view(
    stroke=None,
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
    df_Z = pd.DataFrame(np.array(Z_train),columns=[f'Z{x}' for x in list(range(Z_train.shape[1]))])
    
    # df_Z["age_group"] = list(map(lambda x: 0 if (x>=50 and x<=60) else (1 if (x>60 and x<=70) else (2 if (x>70 and x<=80) else 3)) ,Z_train_[:,1] ))   
    df_Z["age_group"] = Z_train_[:,1]
    feature_pls = X_train@PLS.x_loadings_
    feature_pls = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(feature_pls)
    feature_repls = X_train@rePLS.P
    feature_repls = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(feature_repls)
    df_Z['y_embedded_0'] = feature_pls[:,0]
    df_Z['y_embedded_1'] = feature_pls[:,1]

    df_Z['y_embedded_re_0'] = feature_repls[:,0]
    df_Z['y_embedded_re_1'] = feature_repls[:,1]
    chart_feature = alt.Chart(df_Z,width=350,title='re-PLS').mark_circle().encode(
        y=alt.Y('y_embedded_re_0:Q',title='feature0'),
        x=alt.X('y_embedded_re_1:Q',title="feature1"),
        color=alt.Color('age_group',bin=True,scale=alt.Scale(scheme='dark2'),title="Age"),
    #     shape=alt.Shape('AGE', bin=True)
    )|alt.Chart(df_Z,width=350,title='PLS').mark_circle().encode(
        y=alt.Y('y_embedded_0:Q',title='feature0'),
        x=alt.X('y_embedded_1:Q',title="feature1"),
        color=alt.Color('age_group',bin=True,scale=alt.Scale(scheme='dark2'),title="Age"),
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
PQ_PLS = PLS.coef_
PQ = rePLS.PQ
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

import os
st.subheader('Boostrap and Confident interval')
if st.button("Run 100 bootstrap"):
    df_PQ = pd.DataFrame([],columns= [f'PQ{x}' for x in range(8)])
    df_PQ_re = pd.DataFrame([],columns= [f'PQre{x}' for x in range(8)])
    for random_idx in range(100):
        print(random_idx,"==========================")
        boot_indicies = resample(indicies, replace=True, n_samples=sample_size, random_state=random_idx)
        oob_indicies = [x for x in indicies if x not in boot_indicies]
        X_train = X[boot_indicies]
        X_test = X[oob_indicies]
        y_train = y[boot_indicies]
        y_test = y[oob_indicies]
        Z_train = Z[boot_indicies]
        Z_test = Z[oob_indicies]
        y_test = (X_test-Z_norm[oob_indicies,:]@ZX*true_ZX_p) @ PQ_true
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

        # if not os.path.exists(outputs):
        #     os.makedirs(outputs)


        MLR = LinearRegression()
        PCR = make_pipeline(PCA(n_components=n_components), LinearRegression())
        PLS = PLSRegression(n_components=n_components)





        rePLS = Residual_regression(Z=Z_train,name="rePLS",n_components=n_components)
        rePCR = Residual_regression(Z=Z_train,name="rePCR",n_components=n_components)
        reMLR = Residual_regression(Z=Z_train,name="reMLR",n_components=n_components)


        pipelines = [PLS,rePLS]
        pipeline_names = ["PLS","rePLS"]
        outcome_name = ['CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
                'RAVLT_learning', 'Ventricles']

        #df = pd.DataFrame({"r":[], "MSE":[], "pvalue":[], "method":[]})
        for i,pipe in enumerate(pipelines):

            pipe.fit(X_train, y_train)

        for i,model in enumerate(pipelines):
            if i<1:
                print("{} Test MSE: {:.3f}".format(pipeline_names[i],mean_squared_error(y_test,model.predict(X_test))))
                r,MSE,p_value = evaluation_regression(y_test,np.array(model.predict(X_test)))
                #result =  pd.DataFrame({"r":r, "MSE":MSE, "pvalue":p_value, "method":pipeline_names[i], "output":outcome_name})
                #df = pd.concat([df,result],axis=0)
            else:
                print("{} Test MSE: {:.3f}".format(pipeline_names[i],mean_squared_error(y_test,model.predict(X_test,Z=Z_test))))   
                r,MSE,p_value = evaluation_regression(y_test,np.array(model.predict(X_test,Z=Z_test)))
                #result =  pd.DataFrame({"r":r, "MSE":MSE, "pvalue":p_value, "method":pipeline_names[i],"output":outcome_name})
                #df = pd.concat([df,result],axis=0)


    #		df["isRe"] = df.method.apply(lambda x: pipeline_names.index(x)%3 + int(pipeline_names.index(x)/3)*0.5 )
        PQ = PLS.coef_/ np.linalg.norm(PLS.coef_, axis=0)
        PQ_re = rePLS.PQ/ np.linalg.norm(rePLS.PQ, axis=0)
    #     print(PQ_re)
        df_PQ_temp = pd.DataFrame(PQ,columns= [f'PQ{x}' for x in range(8)])
        df_PQ_re_temp = pd.DataFrame(PQ_re,columns= [f'PQre{x}' for x in range(8)])
        df_PQ_temp['idx'] = df_PQ_temp.index
        df_PQ_re_temp['idx'] = df_PQ_temp.index
        df_PQ_temp['bootstrap_idx'] = random_idx
        df_PQ = pd.concat([df_PQ,df_PQ_temp],axis=0)
        df_PQ_re = pd.concat([df_PQ_re,df_PQ_re_temp],axis=0)
    PQ_true = PQ_true/ np.linalg.norm(PQ_true, axis=0)
    df_PQ_true = pd.DataFrame(PQ_true,columns= [f'PQ{x}' for x in range(8)])
    df_PQ_true['idx'] = df_PQ_true.index
    PQ_idx = 0

    GT_line = alt.Chart(df_PQ_true).mark_line().encode(
        x=alt.X('idx:T'),
        y=alt.Y(f'PQ{PQ_idx}:Q'),
        color=alt.value("#FF0000")
    )
    chart_CI = GT_line+alt.Chart(df_PQ).mark_errorband(extent='ci').encode(
        x=alt.X('idx:T'),
        y=alt.Y(f'mean(PQ{PQ_idx}):Q')
    ).properties(
        title='PLS'
    )

    chart_CI_re = GT_line+alt.Chart(df_PQ_re).mark_errorband(extent='ci').encode(
        x=alt.X('idx:T'),
        y=alt.Y(f'mean(PQre{PQ_idx}):Q')
    ).properties(
        title='rePLS'
    )



    st.altair_chart(chart_CI)

    st.altair_chart(chart_CI_re)
