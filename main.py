from math import ceil
import streamlit as st
import numpy as np
from utils import *
import scipy
from scipy.io import savemat
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt




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

def map_arrow_scale(SNR,min=0,max=150,SNR_min=-20, SNR_max=50):
	#no effect
	if SNR >= SNR_max:
		return min
	elif SNR <= SNR_min:
		return max
	else:
		return (SNR_max- SNR)*(max-min)/(SNR_max-SNR_min)

def plot_confounder_model(SNR_ZX,SNR_ZY,SNR_XY):
# def 
	
	fig  = plt.figure()

	arrow_zx = mpatches.FancyArrowPatch((0, 5),(-5, 0),
									 mutation_scale=map_arrow_scale(SNR_ZX),shrinkA=20, shrinkB=20,color='r')
	arrow_zy = mpatches.FancyArrowPatch((0, 5), (5, 0),
									 mutation_scale=map_arrow_scale(SNR_ZY),shrinkA=20, shrinkB=20, color= 'r')
	arrow_xy = mpatches.FancyArrowPatch((-5,0), (5, 0),
									 mutation_scale=map_arrow_scale(SNR_XY),shrinkA=20, shrinkB=20)
	plt.xlim([-5,5])
	plt.ylim([-1,5])
	fig.gca().add_patch(arrow_zx)
	fig.gca().add_patch(arrow_zy)
	fig.gca().add_patch(arrow_xy)
	plt.text(-0.25, 5, 'Z', fontsize=30)
	plt.text(-5.5, 0, 'X', fontsize=30)
	plt.text(5, 0, 'Y', fontsize=30)
	plt.text(-6, 0, f"SNR_ZX={SNR_ZX}", size=15, rotation=0,
			 ha="right", va="center",
			 
			 )
	plt.text(6, 0, f"SNR_ZY={SNR_ZY}", size=15, rotation=0,
			 ha="left", va="center",
			 )

	plt.text(-0, -0.75, f"SNR_XY={SNR_XY}", size=20, rotation=0,
			 ha="center", va="center",
			 )

	plt.axis('off')
	return fig
	

fig = plot_confounder_model(SNR_ZX,SNR_ZY,SNR_XY)
st.pyplot(fig)

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


@st.cache(suppress_st_warning=True)
def plot_single_brain(P,pathToAnnotationRh,normalize=True):
    x, y, z = vertices.T
    i, j, k = np.asarray(triangles).T
    fig = make_subplots(
    rows=1, cols=1,   
    specs=[[{'type': 'surface'}]])

    if normalize:
        P = min_max_normalize(P)
    P_surface_i = region2surface(P,pathToAnnotationRh)
    fig.add_trace(
    go.Mesh3d(x=x, y=y, z=z,i=i, j=j, k=k,vertexcolor=P_surface_i, showscale=True),
     row=1, col=1)
    return fig

@st.cache(suppress_st_warning=True)
def boostrapping2(X,Y,Y_true,Z,N,**params):
	'''
		Inputs:
			X: input matrix
			Y: output matrix
			Y_true
			Z:
		Output:
			 X_train,X_test,Y_train, Y_test,Z_train, Z_test
	'''
	from sklearn.utils import resample
	sample_size = int(params['population_rate']*N)
	indicies = list(range(N))
	boot_indicies = resample(indicies, replace=True, n_samples=sample_size, random_state=params['random_state'])
	oob_indicies = [x for x in indicies if x not in boot_indicies]
	X_train = X[boot_indicies]
	X_test = X[oob_indicies]
	Y_train = Y[boot_indicies]
    
    # Y = Y_true + f_ZY
	Y_test = Y_true[oob_indicies] +(Y[oob_indicies] -Y_true[oob_indicies] -  Y_noise[oob_indicies])

	Z_train = Z[boot_indicies]
	Z_test = Z[oob_indicies]

	
	return X_train,X_test,Y_train, Y_test,Z_train, Z_test


st.header('Simulated affected regions')
fig_affected_region = plot_single_brain(affected_brain_map,pathToAnnotationRh,normalize=False)
st.plotly_chart(fig_affected_region, use_container_width=True)


#Run regressions
N_repeat = st.number_input("Number of bootstraps", 1,100,10)
df_mean_results = pd.DataFrame({},columns=['r','MSE', 'pvalue'])
outcomes = [f"outcome{x}" for x in range(8)]
for i in range(N_repeat):
    params = {
        'population_rate':0.7,
        'random_state': i
    }
    X_train,X_test,Y_train, Y_test,Z_train, Z_test = boostrapping2(X,Y,Y_true,Z,N,**params)
    
    params = {
        'x_scaler': True,
        'y_scaler': True,
        'z_scaler': True,
        'n_components': 5,
        'PCR_thresh': 1-1e-7,
        'outcomes':outcomes
    }
    if i == 0:
        df_mean_results,model_PLS,model_rePLS,model_MLR,model_reMLR= compare_methods(X_train,X_test, Y_train,Y_test, Z_train, Z_test, **params)
    else:
        df_results,_,_,_,_ = compare_methods(X_train,X_test, Y_train,Y_test, Z_train, Z_test, **params)
        df_mean_results[['r','MSE', 'pvalue']] = df_mean_results[['r','MSE', 'pvalue']] + df_results[['r','MSE', 'pvalue']]
df_mean_results[['r','MSE', 'pvalue']] = df_mean_results[['r','MSE', 'pvalue']]/N_repeat    
#show correlation coefficient
#show correlation coefficient
import altair as alt
from altair import datum
chart_result= alt.Chart(df_mean_results,width=90).mark_bar().encode(
    x=alt.X('isRe:N',axis=None, ),
    y=alt.Y('r:Q',axis=alt.Axis(grid=False)),#scale=alt.Scale(domain=[0.5,1.5])),
#     y2=alt.Y2('MSE:Q'),
    color=alt.Color('method:N',scale=alt.Scale(domain=['MLR','reMLR','PCR','rePCR','PLS','rePLS'],range= ['#9ecae9', '#4c78a8', '#ffbf79', '#f58518', '#88d27a', '#54a24b']), title="Method"),
#         color=alt.Color('method:N',scale=alt.Scale(scheme='tableau20')),
    column=alt.Column('output:N',title=None)
).configure_header(
labelFontSize=15,titleOrient='bottom', labelOrient='bottom',labelAngle=-30,labelAnchor='middle',labelAlign='center',labelPadding=50
).configure_view(
    stroke=None,
).configure_axis(
    labelFontSize=20,
    titleFontSize=20
)



st.altair_chart(chart_result, use_container_width=False)


from sklearn.linear_model import LinearRegression
from rePLS import rePLS, rePCR, reMLR
from sklearn.cross_decomposition import PLSRegression


st.title('PLS scatter plot')
y_predict = model_PLS.predict(X_test)
# y_predict = model_rePLS.predict(X_test,Z=Z_test) #- model_MLR.predict(Z_test)
df_y_truth = pd.DataFrame(Y_test, columns=[outcomes[x] for x in range(8)])
df_y_predict = pd.DataFrame(y_predict, columns=[f'{outcomes[x]}_p' for x in range(8)])
df_y = pd.concat([df_y_truth,df_y_predict],axis=1)
import altair as alt
from altair import datum
from scipy.stats import t,pearsonr
df_y_i = pd.DataFrame({})
charts = []
for idx in range(8):
    r,p = pearsonr(np.array(df_y[[outcomes[idx]]]).ravel(),np.array(df_y[[f'{outcomes[idx]}_p']]).ravel())
    df_y_i = df_y[[outcomes[idx], f'{outcomes[idx]}_p']]
    df_y_i['r'] = f'r={r:1.2f}'
    df_y_i['p'] = f'P={p:1.1e}'
    text_p = alt.Chart(df_y_i).mark_text( 
        x=220,y=260.0
    ).encode(
    text='p:N'
    )
    text_r = alt.Chart(df_y_i).mark_text( 
        x=220,y=250.0
    ).encode(
    text='r:N'
    )
    
    base = alt.Chart(df_y_i,width=250).mark_circle().encode(
        x=alt.X(f"{outcomes[idx]}:Q",axis=alt.Axis(grid=False, ticks= False, domain=True), title=f'Observed {outcomes[idx]}'),
        y=alt.Y(f"{outcomes[idx]}_p:Q",axis=alt.Axis(grid=False, ticks= False, domain=True),title=f'Predicted {outcomes[idx]}'),
    )
#     scatter = base.mark_circle()+ base.transform_regression(f"{outcome_name[idx]}",f"{outcome_name[idx]}_p").mark_line().encode( color=alt.value("#636363")) 
    scatter = text_p+text_r+base.mark_circle()+ base.transform_regression(f"{outcomes[idx]}",f"{outcomes[idx]}_p").mark_line().encode( color=alt.value("#636363")) 
    chart = scatter
    # chart.save(f"outcome_{idx}.html")
    charts.append(chart)

row1 = alt.hconcat()
row2 = alt.hconcat()
for i in range(0,8):
    if i<4:
        row1 = alt.hconcat(row1,charts[i],spacing=100 if i >0 else 0)
        
    else:
#         row2 |= charts[i] 
        row2 = alt.hconcat(row2,charts[i],spacing=100 if i >4 else 0)

chart_scatter_PLS = alt.vconcat(row1,row2,spacing=50).configure_view(stroke=None).configure_facet(
    spacing=0)

st.altair_chart(chart_scatter_PLS)


st.title('rePLS scatter plot')
# ====repls
# 
# y_predict = model_PLS.predict(X_test)
y_predict = model_rePLS.predict(X_test,Z=Z_test) #- model_MLR.predict(Z_test)
df_y_truth = pd.DataFrame(Y_test, columns=[outcomes[x] for x in range(8)])
df_y_predict = pd.DataFrame(y_predict, columns=[f'{outcomes[x]}_p' for x in range(8)])
df_y = pd.concat([df_y_truth,df_y_predict],axis=1)
import altair as alt
from altair import datum
from scipy.stats import t,pearsonr
df_y_i = pd.DataFrame({})
charts = []
for idx in range(8):
    r,p = pearsonr(np.array(df_y[[outcomes[idx]]]).ravel(),np.array(df_y[[f'{outcomes[idx]}_p']]).ravel())
    df_y_i = df_y[[outcomes[idx], f'{outcomes[idx]}_p']]
    df_y_i['r'] = f'r={r:1.2f}'
    df_y_i['p'] = f'P={p:1.1e}'
    text_p = alt.Chart(df_y_i).mark_text( 
        x=220,y=260.0
    ).encode(
    text='p:N'
    )
    text_r = alt.Chart(df_y_i).mark_text( 
        x=220,y=250.0
    ).encode(
    text='r:N'
    )
    
    base = alt.Chart(df_y_i,width=250).mark_circle().encode(
        x=alt.X(f"{outcomes[idx]}:Q",axis=alt.Axis(grid=False, ticks= False, domain=True), title=f'Observed {outcomes[idx]}'),
        y=alt.Y(f"{outcomes[idx]}_p:Q",axis=alt.Axis(grid=False, ticks= False, domain=True),title=f'Predicted {outcomes[idx]}'),
    )
#     scatter = base.mark_circle()+ base.transform_regression(f"{outcome_name[idx]}",f"{outcome_name[idx]}_p").mark_line().encode( color=alt.value("#636363")) 
    scatter = text_p+text_r+base.mark_circle()+ base.transform_regression(f"{outcomes[idx]}",f"{outcomes[idx]}_p").mark_line().encode( color=alt.value("#636363")) 
    chart = scatter
    # chart.save(f"outcome_{idx}.html")
    charts.append(chart)

row1 = alt.hconcat()
row2 = alt.hconcat()
for i in range(0,8):
    if i<4:
        row1 = alt.hconcat(row1,charts[i],spacing=100 if i >0 else 0)
        
    else:
#         row2 |= charts[i] 
        row2 = alt.hconcat(row2,charts[i],spacing=100 if i >4 else 0)

chart_scatter_rePLS = alt.vconcat(row1,row2,spacing=50).configure_view(stroke=None).configure_facet(
    spacing=0)
st.altair_chart(chart_scatter_rePLS)