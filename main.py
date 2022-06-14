from math import ceil
import streamlit as st
import numpy as np
from utils import *
import scipy
from scipy.io import savemat



############### Initialize parameter
with st.sidebar:
    st.title('Simulation parameters')
    N = int(st.number_input('Number of samples', 1000,step=1))
    
    
    #========================== Simulate X,Z,PQ, Y ===============================
    st.header('Simulate Z')
    with st.expander('Confounder parameters'):
        gender_ratio = st.number_input('male/female ratio',1)


    st.header('Simulate X')
    with st.expander('Input parameters'):
        I = int(st.number_input('Number of input features',202,step=1))
        SNR_ZX = st.number_input('SNR_ZX',-5)
        seed = int(st.number_input('seed',0,step=1))
        Z_affect_regions = st.number_input('\% region Z affect X', 5)/100


    st.header('Simulate PQ')
    with st.expander('PQ parameters'):
        principle_component = int(st.number_input('principle_component', 5,step=1))


    st.header('Simulate Y')
    with st.expander('Output parameters'):
        J = int(st.number_input('Number of output', 8,step=1))
        SNR_ZY = st.number_input('SNR_ZY',-15)
        SNR_XY = st.number_input('SNR_XY', 5)
    

################### ==================  values
params = {
    'gender_ratio':gender_ratio,
    'age_group_weight':np.array([10, 20, 70])/100, #40-60-80 age group percentage
    'normalized':True
}

Z = gen_confounder(N, **params)

# Simulate X
params = {
    'SNR_ZX':SNR_ZX,
    'seed':seed, 
    'Z_affect_regions':Z_affect_regions,
}
X,X_true,affected_brain_map= gen_X2(I,Z,**params)

params = {
    'seed':0, 
    'principle_component':principle_component,
    'method':'PCA'
}

P_true,PQ_true = gen_P_PQ(X_true,J,**params)
# P_true = np.random.rand(5,202)
# PQ_true = np.random.rand(202,8)
print(f"SNR_ZX = {10*np.log10(np.linalg.norm(X_true)/np.linalg.norm(X-X_true))}")
print(f"rank/dim = {np.linalg.matrix_rank(X)}/{X.shape[1]} = {np.linalg.matrix_rank(X)/X.shape[1]:.2f}")


# ======== =================== Simulate Y ===================
# Simulate Y
params = {
    'SNR_ZY': SNR_ZY, #/f_ZY
    'SNR_XY': SNR_XY #Y_noise
}
# Y = Y_true + f_ZY + Y_noise

                                                                                                                                                                                                
Y,Y_true,Y_noise = gen_Y(X_true,Z,PQ_true,**params)

print(f"SNR_XY={10*np.log10(np.linalg.norm(Y_true)/np.linalg.norm(Y_noise)):.2f}")
print(f"SNR_ZY={10*np.log10(np.linalg.norm(Y_true)/np.linalg.norm(Y-Y_true-Y_noise)):.2f}")
# Y = Y/np.linalg.norm(Y,axis=0)
# Y_true = Y_true/np.linalg.norm(Y_true,axis=0)

###################============================================= Run exp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from nilearn import datasets, surface


vertices, triangles = surface.load_surf_mesh('mesh.inflated.freesurfer.gii')
pathToAnnotationRh = 'lh.Schaefer2018_200Parcels_7Networks_order.annot'

x, y, z = vertices.T
i, j, k = np.asarray(triangles).T

# P_thresh = min_max_normalize(P_true)
vertex_color = []
for ii in range(5):
#     P_thresh = min_max_normalize(P_true)
    P_surface_i = region2surface(P_true[ii,:],pathToAnnotationRh)
    vertex_color.append(vertex2color(P_surface_i))

PQ_vertex_color = []
for ii in range(5):
#     P_thresh = min_max_normalize(P_true)
    PQ_surface_i = region2surface(PQ_true[:,ii],pathToAnnotationRh)
    PQ_vertex_color.append(vertex2color(PQ_surface_i))



st.title('Visualize the ground truth P and PQ')
P_idx = st.slider('Choose P_i', 0, principle_component, 0)
# if st.button(f'Show P_i'):
fig5 = make_subplots(
rows=1, cols=1,   
specs=[[{'type': 'surface'}]])
fig5.add_trace(
go.Mesh3d(x=x, y=y, z=z,i=i, j=j, k=k,vertexcolor=vertex_color[P_idx], showscale=True),
row=1, col=1
)
st.plotly_chart(fig5, use_container_width=True)

PQ_idx = st.slider('Choose PQ_i', 0, J, 0)
# if st.button(f'Show PQ_i'):
fig_PQ = make_subplots(
rows=1, cols=1,   
specs=[[{'type': 'surface'}]])
fig_PQ.add_trace(
go.Mesh3d(x=x, y=y, z=z,i=i, j=j, k=k,vertexcolor=PQ_vertex_color[PQ_idx], showscale=True),
row=1, col=1
)
st.plotly_chart(fig_PQ, use_container_width=True)


