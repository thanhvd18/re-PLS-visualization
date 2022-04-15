import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import altair as alt
from altair import datum
import subprocess

col1, col2 = st.columns(2)


@st.experimental_memo
def get_data(path):
    df1 = pd.read_csv(path, index_col=False)
    df1['Z0'] = df1['Z0'].map({i: ["Female", "Male"][i] for i in range(2)})
    return df1
    


# Using "with" notation
with st.sidebar:
    cofounder_type = st.radio(
        "Confounders",
        ('Age', 'Gender'))
    use_violinplot = st.checkbox('Violin plot')

st.header("Surface")

cofounder_type_code = 0 if cofounder_type=="Age" else 1
# 	outcome_i = 
age_space = '5'
# age_bin = 10
if cofounder_type == 'Age':
    age_space = st.select_slider(
        'Select age space',
        options=['2', '5', '10'],value='5')
    age_dict = {'2':20,'5':10, '10':5}
    age_bin = age_dict[age_space]
col1, col2 = st.columns(2)
df1 = get_data('train_surface.csv')
outcome_name = ['CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
       'RAVLT_learning', 'Ventricles']
#
#vilolet plot
# ============================================ Surface=========================
ROIs = st.slider('Choose the surface region (X_i)', 1, 152, 1)
if cofounder_type_code == 0:
    if use_violinplot:
        chart_region_age = alt.Chart(df1,width=800,title=f'Cortical thickness at region against age').transform_bin(
        'Age', field='Z1',bin=alt.Bin(maxbins=age_bin)
    ).transform_density(
        f'X{ROIs-1}',
        as_=[f'X{ROIs-1}', 'density'],
    #     extent=[5, 50],
        groupby=['Age']
    ).mark_area(orient='horizontal').encode(
        y=alt.Y(f'X{ROIs-1}:Q',title='Cortical thickness (normalized)'),
        # color='Z1:O',
        x=alt.X(
            'density:Q',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            'Age:N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),)
    ).properties(
        width=80
    ).configure_facet(
        spacing=2
    ).configure_view(
        stroke=None
    )
    else:
        chart_region_age = alt.Chart(df1,width=700,title=f'Cortical thickness at region {ROIs}th against age').transform_bin(
            'Z1', field='Z1',bin=alt.Bin(maxbins=age_bin)
        ).mark_boxplot().encode(
            x=alt.X('Z1:O', title='Age'),
            y=alt.Y(f'X{ROIs-1}:Q', title='Cortical thickness (normalized)'))



else:
    if use_violinplot:
        chart_region_age = alt.Chart(df1,width=800,title=f'Cortical thickness at region {ROIs}th against gender').transform_density(
        f'X{ROIs-1}',
        as_=[f'X{ROIs-1}', 'density'],
    #     extent=[5, 50],
        groupby=['Z0']
    ).mark_area(orient='horizontal').encode(
        y=alt.Y(f'X{ROIs-1}:Q',title='Cortical thickness (normalized)'),
        # color='Z1:O',
        x=alt.X(
            'density:Q',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            'Z0:N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),)
    ).properties(
        width=80
    ).configure_facet(
        spacing=2
    ).configure_view(
        stroke=None
    )
    else:
        chart_region_age = alt.Chart(df1,width=700,title=f'Cortical thickness at region {ROIs}th against gender').mark_boxplot().encode(
            x=alt.X('Z0:O', title='Gender'),
            y=alt.Y(f'X{ROIs-1}:Q', title='Cortical thickness (normalized)'))

st.altair_chart(
    chart_region_age.interactive(), use_container_width=False
)

with st.expander("Remarks"):
     st.write("""
        Remarks: Regions 85 86 can not be visualized bz they contain nan values
     """)
# ========================================================== Volume =============================
st.header("Volume")
# col3, col4 = st.columns(2)
df2 = get_data('train_volume.csv')
ROI = st.slider('Choose the volume region (X_i)', 1, 95, 1)

if cofounder_type_code == 0:
    if use_violinplot:
        chart_region_age = alt.Chart(df1,width=800,title=f'Cortical thickness at region {ROI}th against age').transform_bin(
        'Age', field='Z1',bin=alt.Bin(maxbins=age_bin)
    ).transform_density(
        f'X{ROI-1}',
        as_=[f'X{ROI-1}', 'density'],
    #     extent=[5, 50],
        groupby=['Age']
    ).mark_area(orient='horizontal').encode(
        y=alt.Y(f'X{ROI-1}:Q',title='Cortical thickness (normalized)'),
        # color='Z1:O',
        x=alt.X(
            'density:Q',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            'Age:N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),)
    ).properties(
        width=80
    ).configure_facet(
        spacing=2
    ).configure_view(
        stroke=None
    )
    else:
        chart_region_age = alt.Chart(df1,width=700,title=f'Cortical thickness at region {ROI}th against age').transform_bin(
            'Z1', field='Z1',bin=alt.Bin(maxbins=age_bin)
        ).mark_boxplot().encode(
            x=alt.X('Z1:O', title='Age'),
            y=alt.Y(f'X{ROI-1}:Q', title='Cortical thickness (normalized)'))



else:
    if use_violinplot:
        chart_region_age = alt.Chart(df1,width=800,title=f'Cortical thickness at region {ROI}th against gender').transform_density(
        f'X{ROI-1}',
        as_=[f'X{ROI-1}', 'density'],
    #     extent=[5, 50],
        groupby=['Z0']
    ).mark_area(orient='horizontal').encode(
        y=alt.Y(f'X{ROI-1}:Q',title='Cortical thickness (normalized)'),
        # color='Z1:O',
        x=alt.X(
            'density:Q',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            'Z0:N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),)
    ).properties(
        width=80
    ).configure_facet(
        spacing=2
    ).configure_view(
        stroke=None
    )
    else:
        chart_region_age = alt.Chart(df1,width=700,title=f'Cortical thickness at region {ROI}th against gender').mark_boxplot().encode(
            x=alt.X('Z0:O', title='Gender'),
            y=alt.Y(f'X{ROI-1}:Q', title='Cortical thickness (normalized)'))

st.altair_chart(
    chart_region_age.interactive(), use_container_width=False
)
with st.expander("Remarks"):
     st.write("""
        xxx
     """)
     
# ============================================================ Outputs=========================
st.header("Outputs")
outcome_i = st.slider('Choose the outcome (y_i)', 0, 7, 1)
if cofounder_type_code == 0:
    if use_violinplot:
        chart_region_age = alt.Chart(df1,width=800,title=f'{outcome_name[outcome_i]} against age').transform_bin(
        'Age', field='Z1',bin=alt.Bin(maxbins=age_bin)
    ).transform_density(
        f'y{outcome_i-1}',
        as_=[f'y{outcome_i-1}', 'density'],
    #     extent=[5, 50],
        groupby=['Age']
    ).mark_area(orient='horizontal').encode(
        y=alt.Y(f'y{outcome_i-1}:Q',title=f'y{outcome_i}'),
        # color='Z1:O',
        x=alt.X(
            'density:Q',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            'Age:N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),)
    ).properties(
        width=80
    ).configure_facet(
        spacing=2
    ).configure_view(
        stroke=None
    )
    else:
        chart_region_age = alt.Chart(df1,width=700,title=f'{outcome_name[outcome_i]} against age').transform_bin(
            'Z1', field='Z1',bin=alt.Bin(maxbins=age_bin)
        ).mark_boxplot().encode(
            x=alt.X('Z1:O', title='Age'),
            y=alt.Y(f'y{outcome_i-1}:Q', title=f'y{outcome_i}'))



else:
    if use_violinplot:
        chart_region_age = alt.Chart(df1,width=800,title=f'{outcome_name[outcome_i]} against gender').transform_density(
        f'y{outcome_i-1}',
        as_=[f'y{outcome_i-1}', 'density'],
    #     extent=[5, 50],
        groupby=['Z0']
    ).mark_area(orient='horizontal').encode(
        y=alt.Y(f'y{outcome_i-1}:Q',title=f'y{outcome_i}'),
        # color='Z1:O',
        x=alt.X(
            'density:Q',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            'Z0:N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),)
    ).properties(
        width=80
    ).configure_facet(
        spacing=2
    ).configure_view(
        stroke=None
    )
    else:
        chart_region_age = alt.Chart(df1,width=700,title=f'{outcome_name[outcome_i]} against gender').mark_boxplot().encode(
            x=alt.X('Z0:O', title='Gender'),
            y=alt.Y(f'y{outcome_i}:Q', title=f'y{outcome_i}'))

st.altair_chart(
    chart_region_age.interactive(), use_container_width=False
)




with st.expander("See explanation outputs"):
     st.write("""
        'CDRSB': 
        'ADAS11':,
        'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
       'RAVLT_learning', 'Ventricles'
     """)
     

