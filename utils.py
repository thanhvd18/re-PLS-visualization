
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

outcome_name = ['CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate','RAVLT_learning', 'Ventricles']
cofounders = ["PTGENDER", "AGE"]

def gen_P_PQ(X_true,J,**params):
    from sklearn.decomposition import SparsePCA
    N,I = X_true.shape
    K = params['principle_component']
    #generate P_true, PQ_true in PLS
    sparse_pca = SparsePCA(n_components=K,alpha=0.95)
    sparse_pca.fit(X_true)
    
    P_true = sparse_pca.components_
    alpha = np.eye(K)
    Q = np.random.rand(J,K)
    PQ_true =P_true.T@alpha@Q.T
#     PQ_true = min_max_normalize(PQ_true)
    return P_true, PQ_true
    
def split_data(df,split_type = "kfold",test_split=0.3,fold=0, seed=0,outcomes=outcome_name,x_scaler=True,y_scaler=True,z_scaler=True):
	from sklearn import preprocessing
	from sklearn.model_selection import KFold
	if split_type != 'kfold':
		fold = None
	elif split_type != 'randomfold':
		test_split = None
	elif split_type != 'leakage':
		seed = None

	subject_all = df.SubjectID.unique()
	feature_type = 'surface'#'volume' # 'surface'
	input_feature = 'Schaefer_200_7'#'hammers' # 'volume' = [cobra	hammers	ibrs	lpba40	neuromorphometrics], surface= [	a2009s	DK40]

	if split_type != 'leakage': 
		if split_type == "kfold":

			print("Number of subjects:",len(subject_all))
			kf = KFold(n_splits=4,shuffle=True,random_state=seed)
			kf.get_n_splits(subject_all)
			df_test_fold = []
			df_train_fold = []
			for train_index, test_index in kf.split(subject_all):
			#     print("TRAIN:", train_index, "TEST:", test_index)
				df_test_fold.append(df[df.SubjectID.isin(subject_all[test_index])])
				df_train_fold.append(df[df.SubjectID.isin(subject_all[train_index])])
	#             print(len(df_test),len(df_train),len(df_train)/len(df_test))

		elif split_type == "randomfold":
			np.random.seed(seed)
			subject_all = df.SubjectID.unique()
			print("Number of subjects:",len(subject_all))
			perm = np.random.permutation(len(subject_all))
			num_test_subj = int(len(subject_all)*test_split)
			sub_index = list(perm[:num_test_subj])
			subject_test = [subject_all[i] for i in sub_index]
			df_test = df[df.SubjectID.isin(subject_test)]
			df_train = df[~df.SubjectID.isin(subject_test)]
	else: # select last run of sujects having multiple run    
		df_processed = df.sort_values(['SubjectID', 'ScanDate'], ascending=[True, True])
		df_test = df[df.duplicated(subset='SubjectID', keep=False)].drop_duplicates(subset='SubjectID', keep='last')
		df_train_index = df.index.difference(df_test.index)
		df_train = df.iloc[df_train_index].sort_values(['SubjectID', 'ScanDate'], ascending=[True, True])


	if split_type == 'kfold':
		df_test = df_test_fold[fold]
		df_train = df_train_fold[fold]

	X_test = df_test[input_feature].apply(lambda x: np.array(eval(x)), 0)
	X_test = np.vstack(X_test)
	y_test = df_test[outcomes]
	Z_test = df_test[cofounders]

	X_train = df_train[input_feature].apply(lambda x: np.array(eval(x)), 0)
	X_train = np.vstack(X_train)
	y_train = df_train[outcomes]
	Z_train = df_train[cofounders]

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

	df_train.reset_index(inplace=True)
	df_test.reset_index(inplace=True)
	return X_train,y_train,Z_train,df_train,X_test,y_test, Z_test,df_test



def get_data(data_path,features,outcomes=outcome_name,cofounders=cofounders):
	"""
	
		return: filtered+ dropout + encoded dataframe
	"""
	
	base_fields = ["SubjectID", "ScanDate","UID","image_path"] + features
	# Select outcomes, cofounders and dropout samples do not have these values
	df = pd.read_csv(data_path,index_col=False,low_memory=False)
	df_processed = df[~df.Schaefer_200_7.isna()] # processed, cobra atlas feature
   
	filted_fields = base_fields + outcomes + cofounders
	df_processed = df_processed[filted_fields].dropna(axis='rows')
	df_processed = df_processed.drop_duplicates(subset=['SubjectID', 'ScanDate'])
	# df_processed_drop#.head()
	df_processed.PTGENDER = df_processed.PTGENDER.map({'Female': '0', 'Male': '1'})
	# df_processed["age_group"] = df_processed.AGE.apply(lambda x: 0 if (x>=50 and x<=60) else (1 if (x>60 and x<=70) else (2 if (x>70 and x<=80) else 3)) )
	return  df_processed



def get_violin_chart(df,idx,pvalue=True,outcome_name=outcome_name):
	"""
		xxx
	"""

	df_plot = df.copy()
	#Setup chart data+properties
	gender_chart = alt.Chart(df_plot,width=85).mark_area(orient='horizontal').transform_density( #["value", "density"]
		f'{outcome_name[idx]}',
		groupby=['Z0'],
		as_=['value', 'density'],
		).encode(
			x=alt.X('density:Q',stack=False,impute=None,title=None,axis=alt.Axis(labels=False ,grid=False, ticks=False, domain=False)),
			y=alt.Y('value:Q',title="Cortical thickness",axis=alt.Axis(grid=False, ticks=False)),
			color= alt.Color('Z0:N',title="Gender",scale=alt.Scale(domain=["Male", "Female","50-60","60.1-70","70.1-80",">80.1"],range= ['#ff9da6','#9ecae9','#66c2a5','#8da0cb', '#fc8d62','#e78ac3'])),
		).transform_calculate(
		density='datum.Z0=="Female"?datum.density:datum.density*-1',
		)
	text2 = alt.Chart(df_plot).mark_text( 
			x=40,y=30
		).encode(
		text='p:N'
		)
	return gender_chart+text2 if pvalue else gender_chart
	
def get_boxplot_age_chart(df,df_test,idx,pvalue=True,outcome_name=outcome_name):
	"""
		df_1: xxx
		df_test: xxx
		return: xxx
	""" 
	from scipy import stats
	from scipy.stats import pearsonr

	df_plot = df.copy()
	df_anova = pd.DataFrame({})
	p_list = []

	#calculate ANOVA test
	a0 = df_plot[(df_plot.age_group == 0)][outcome_name[idx]].tolist()
	a1 = df_plot[(df_plot.age_group == 1)][outcome_name[idx]].tolist()
	a2 = df_plot[(df_plot.age_group == 2)][outcome_name[idx]].tolist()
	a3 = df_plot[(df_plot.age_group == 3)][outcome_name[idx]].tolist()
	F, p = stats.f_oneway(a0,a1,a2,a3)
	# print(p)
	#save p-value in table
	df_plot['p_age'] = f"P={p:1.1e}"

	#plot
	age_chart = alt.Chart(df_plot,width=65).mark_boxplot(extent='min-max').encode(
		x=alt.X('age_group:N',axis=None),
		y=alt.Y(f'{outcome_name[idx]}:Q',title=None,axis=alt.Axis(grid=False, ticks=False,domain=False,labels=False)),  

		color= alt.Color('age_group:N',title="Age group",scale=alt.Scale(domain=["50-60","60.1-70","70.1-80",">80.1"],range= ['#66c2a5','#8da0cb', '#fc8d62','#e78ac3'])),
	).transform_calculate(
	age_group='datum.age_group=="0"?"50-60":datum.age_group=="1"?"60.1-70":datum.age_group=="2"?"70.1-80":">80.1"',
	)

	text2 = alt.Chart(df_plot).mark_text( 
		x=35,y=10.0
	).encode(
	text='p_age'
	)
	return (age_chart+text2) if pvalue else age_chart


def evaluation_regression(y_test,y_pred):
	from sklearn.metrics import mean_squared_error
	from scipy.stats import t,pearsonr
	"""
		y_test: xxx
		y_pred: xxx
		return: xxx
	"""
	rs = [] 
	MSEs  = []
	p_values = []
	for i in range(y_test.shape[1]):
#         r_matrix = np.corrcoef(np.array(y_test[:,i]), np.array(y_pred[:,i]))
#         r = r_matrix[0,1]
		# r,_= pearsonr(np.array(y_test[:,i]), np.array(y_pred[:,i]))
		r,p = pearsonr(np.array(y_test[:,i]).ravel(),np.array(y_pred[:,i]).ravel())
		rs.append(r)

		# #Calculate pvalue
		# dof = y_test.shape[0]
		# t_stat = r/ np.sqrt(1 - r**2)* np.sqrt(dof)
		# p_value = 2*(t.cdf(-abs(t_stat), dof))
		p_values.append(p)
		MSEs.append(mean_squared_error(y_test[:,i],y_pred[:,i]))
	return rs,MSEs,p_values


def gen_Y(X_true,Z,PQ_true,**params):
	"""
		Input:        
			J: number of outcomes        
			Z: confounder matrix
			X_true: input matrix that directly effect Y
			
			params:
			
				
		Return: Y: JxN matrix
		
	"""
	
	N,_ = X_true.shape
	_,R = Z.shape
	I,J = PQ_true.shape
	
	
	# the output matrix that is not effected by the confounder matrix
	Y_true = X_true @ PQ_true
	
	Y_noise = np.random.randn(*Y_true.shape)
	Y_noise = Y_noise* (np.linalg.norm(Y_true, axis=0)/ (np.linalg.norm(Y_noise, axis=0)* 10**(params['SNR_XY']/10)))
	
	#check SNR fomular
	assert  abs(params['SNR_XY'] - 10*np.log10(np.linalg.norm(Y_true)/np.linalg.norm(Y_noise))) <1e-6

	
	
	#relationship between Z and Y: linear mapping
	ZY = np.random.randn(R,J)
	f_ZY =  Z@ZY
	f_ZY =  f_ZY * (np.linalg.norm(Y_true, axis=0)/ (np.linalg.norm(f_ZY, axis=0)* 10**(params['SNR_ZY']/10)))
	
	assert  abs(params['SNR_ZY'] - 10*np.log10(np.linalg.norm(Y_true)/np.linalg.norm(f_ZY))) <1e-6
	Y = Y_true + f_ZY + Y_noise
	
	
	return Y,Y_true,Y_noise

def gen_X(I,Z,**params):
	"""
		Input: 
			I:
			Z:
			params:
				multiColinearRate: [0-1]: 0:no multicolinear => full rank, 1=>rank = 0
				SNR_ZX
				
		Output: X: NxI matrix
	"""
	
	N,R = Z.shape
	
	#rank of X
	Rx = int(params['multiColinearRate']*I)
	U, S, V = np.linalg.svd(np.random.randn(N, I), full_matrices=False)
	U = U[:,:Rx]
	
	#causal X -> output Y
	X_true = U @ np.diag(S)[:Rx,:Rx]@ V[:,:Rx].T 
	
	#relationship between Z and X: linear mapping
	ZX = np.random.randn(R,I)
	f_ZX =  Z@ZX
	f_ZX =  f_ZX * (np.linalg.norm(X_true, axis=0)/ (np.linalg.norm(f_ZX, axis=0)* 10**(params['SNR_ZX']/10)))
		
	
	#check SNR fomular
	assert  abs(params['SNR_ZX'] - 10*np.log10(np.linalg.norm(X_true)/np.linalg.norm(f_ZX))) <1e-6
	X = X_true + f_ZX

	return X,X_true
	
def gen_confounder(N,**params):
	'''
		Input:
			N: number of samples
			R: number of confounders
			params (dict):
				'gender_ratio': [0-1]
				'age_group_weight':
		
		Output: confounder matrix Z: NxR matrix
	'''
	import random
	gender_ratio = params['gender_ratio']
	age_group_weight = params['age_group_weight']
	age_group_sample = [int(x*N) for x in age_group_weight]
	Z = np.zeros((N,2))
	
	Z0 = [0]*int(N*gender_ratio) + [1]*(N-int(N*gender_ratio)) 
	random.shuffle(Z0)
	Z[:,0] = Z0
	
	Z1 = np.random.randint(30, 50, size=(age_group_sample[0],)).tolist() + np.random.randint(51, 70, size=(age_group_sample[1],)).tolist()+ np.random.randint(71, 90, size=(age_group_sample[2],)).tolist()
	random.shuffle(Z1)    
	Z[:,1] = Z1 
	# if params['normalized']:
		# Z = Z/np.linalg.norm(Z,axis=0)
	return Z
def plot_heatmap(X_train,show_heatmap=False):
	import seaborn as sns
	from scipy.stats import t,pearsonr
	from numpy.linalg import matrix_rank

	r = np.zeros((X_train.shape[1],X_train.shape[1]))
	for i in range(X_train.shape[1]):
		for j in range(X_train.shape[1]):
			r_temp, _ = pearsonr(X_train[i],X_train[j])
			r[i,j] = r_temp
	df_X_cor = pd.DataFrame(r, columns=[f'X{x}' for x in range(X_train.shape[1])])
	df_X_cor.index = [f'X{x}' for x in range(X_train.shape[1])]  
	if show_heatmap:  
		sns.heatmap(df_X_cor)
		return df_X_cor
	# print(f"Rank: {matrix_rank(r)}")
	return r

def boostrapping(X,Y,Y_true,Z,N,**params):
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
	Y_test = Y_true[oob_indicies]

	Z_train = Z[boot_indicies]
	Z_test = Z[oob_indicies]

	
	return X_train,X_test,Y_train, Y_test,Z_train, Z_test
def compare_methods(X_train,X_test, Y_train,Y_test, Z_train, Z_test, **params):
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import Pipeline,make_pipeline
    # from rePLS import Residual_regression
    from rePLS import rePLS,rePCR,reMLR
    #scaler X,Y,Z
    
    if params['x_scaler']:
      scalerx = preprocessing.StandardScaler().fit(X_train)
      X_train = scalerx.transform(X_train)
      X_test = scalerx.transform(X_test)

    if params['y_scaler']:
      scalery = preprocessing.StandardScaler().fit(Y_train)
      Y_train = scalery.transform(Y_train)
      Y_test = scalery.transform(Y_test)

    if params['z_scaler']:
      scalerz = preprocessing.StandardScaler().fit(Z_train)
      Z_train = scalerz.transform(Z_train)
      Z_test = scalerz.transform(Z_test)

    #initialize models
    n_components = params['n_components']
    r = plot_heatmap(X_train)
    eigen_values,eigen_vectors = np.linalg.eig(r)
    eigen_values = np.real(eigen_values)
    eigen_values = eigen_values/np.linalg.norm(eigen_values)
    energy_cum = np.cumsum(np.square(eigen_values), dtype=float)

    PCR_n_components = np.argmax(energy_cum>params['PCR_thresh'])
    print(f"PCR components = {PCR_n_components}")
    PCR_n_components = max(n_components,PCR_n_components)
    MLR = LinearRegression()
    PCR = make_pipeline(PCA(n_components=PCR_n_components), LinearRegression())
    PLS = PLSRegression(n_components=n_components)

    rePLS_model = rePLS(Z=Z_train,n_components=n_components)
    rePCR_model = rePCR(Z=Z_train,n_components=PCR_n_components)
    reMLR_model = reMLR(Z=Z_train,n_components=n_components)


    #edit later
    pipelines = [MLR,PCR,PLS,reMLR_model,rePCR_model,rePLS_model]
    pipeline_names = ["MLR","PCR","PLS","reMLR","rePCR","rePLS"]


    #edit later
    outcome_name = params['outcomes']
    import pandas as pd
    df = pd.DataFrame({"r":[], "MSE":[], "pvalue":[], "method":[]})
    for i,pipe in enumerate(pipelines):

      pipe.fit(X_train, Y_train)

    for i,model in enumerate(pipelines):
      #edit later
      if i<3:
  #             print("{} Test MSE: {:.3f}".format(pipeline_names[i],mean_squared_error(Y_test,model.predict(X_test))))
        r,MSE,p_value = evaluation_regression(Y_test,np.array(model.predict(X_test)))
        result =  pd.DataFrame({"r":r, "MSE":MSE, "pvalue":p_value, "method":pipeline_names[i], "output":outcome_name})
        df = pd.concat([df,result],axis=0)
      else:
  #             print("{} Test MSE: {:.3f}".format(pipeline_names[i],mean_squared_error(Y_test,model.predict(X_test,Z=Z_test))))   
        r,MSE,p_value = evaluation_regression(Y_test,np.array(model.predict(X_test,Z=Z_test)))
        result =  pd.DataFrame({"r":r, "MSE":MSE, "pvalue":p_value, "method":pipeline_names[i],"output":outcome_name})
        df = pd.concat([df,result],axis=0)
    # pipeline_dict = {"0":"MLR", "0.5",}
    df["isRe"] = df.method.apply(lambda x: pipeline_names.index(x)%3 + int(pipeline_names.index(x)/3)*0.5 )
    # PLS.fit(X_train, Y_train),rePLS_model.fit(X_train, Y_train),MLR.fit(X_train, Y_train),reMLR_model.fit(X_train, Y_train)
    return df,PLS,rePLS_model,MLR,reMLR_model
    
def region2surface(P,pathToAnnotationRh):
    from nibabel import freesurfer as nfs
    rh = nfs.read_annot(pathToAnnotationRh)
    P = P.ravel()
    atlas_left_cdata  = rh[0]
    atlas_left_id = rh[1][:,4]

    w_left_cdata = np.zeros(atlas_left_cdata.shape)
    w_right_cdata = np.zeros(atlas_left_cdata.shape)
    for i in range(int(len(P)/2)):
        w_left_cdata[atlas_left_cdata == i] = P[2*i]
        w_right_cdata[atlas_left_cdata == i] = P[2*i+1]

    surface_data = np.hstack([w_left_cdata, w_right_cdata])
    return surface_data
def vertex2color(gg_mesh,is_normalize=True,colormap = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', '#eff3ff','#CCCCCC','#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']):
    if is_normalize:
        min_val = -1
        max_val = 1
    else:
        min_val = np.min(gg_mesh)
        max_val = np.max(g_mesh)
        
    range_ = max_val - min_val
    step = range_/(len(colormap)-1)
    vertex_color = [colormap[int((x+1+1e-6)/(step))] for x in (gg_mesh.ravel())]
    return vertex_color
def gen_X2(I,Z,**params):
    N,R = Z.shape
    
    X_true = np.random.randn(N, I)
    X_true[:,:2] = 0
    
    #relationship between Z and X: linear mapping
    #impose effect of Z on P, PQ
    np.random.seed(params['seed'])
    number_affected_region = int(params['Z_affect_regions']*I)
    affected_regions = np.random.randint(0, I,number_affected_region)
    ZX = np.random.randn(R,number_affected_region)
    f_ZX =  Z@ZX
    f_ZX =  f_ZX * (np.linalg.norm(X_true[:,affected_regions], axis=0)/ (np.linalg.norm(f_ZX, axis=0)* 10**(params['SNR_ZX']/10)))


    #check SNR fomular
    assert  abs(params['SNR_ZX'] - 10*np.log10(np.linalg.norm(X_true[:,affected_regions])/np.linalg.norm(f_ZX))) <1e-6
    X = X_true.copy();
    X[:,affected_regions] = X_true[:,affected_regions]+ f_ZX
    
    
    affected_brain_map = np.zeros(I)
    affected_brain_map[affected_regions] = 1 *np.sign((sum(X_true[:,affected_regions])))
    return X,X_true,affected_brain_map

def min_max_normalize(P,thresh_per=0.8):
    P_std = (P - P.min()) / (P.max() - P.min())
    P_scaled = P_std * 2 + -1

    # Threshold min max
    vol = np.sort(P_scaled.flatten())
    min_thresh = vol[int(len(vol)*thresh_per)]
    max_thresh = vol[int(len(vol)*(1-thresh_per))]
    P_thresh = np.zeros(P_scaled.shape)
    P_thresh[P_scaled >=max_thresh] = 1
    # P_thresh[P_scaled <max_thresh and P_scaled>min_thresh] = 0
    P_thresh[P_scaled<=min_thresh] = -1
    return P_thresh

def gen_P_PQ(X_true,J,**params):
	from sklearn.decomposition import SparsePCA
	from sklearn.decomposition import PCA
	N,I = X_true.shape
	K = params['principle_component']
	#generate P_true, PQ_true in PLS
	if params['method'] == 'sparsePCA':
		model = SparsePCA(n_components=K,alpha=0.95)
	elif params['method'] == 'PCA':
		model = PCA(n_components=K)
	model.fit(X_true)

	P_true = model.components_
	alpha = np.eye(K)
	Q = np.random.rand(J,K)
	PQ_true =P_true.T@alpha@Q.T
	#     PQ_true = min_max_normalize(PQ_true)
	return P_true, PQ_true
if __name__ == '__main__':
	pass
