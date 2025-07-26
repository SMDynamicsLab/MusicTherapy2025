#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pacheco et al 2025, "Music interventions and rhythm processing in traumatic brain injury: case report"
Created on Mon May  5 19:10:05 2025

@author: rodrigo
"""

#%% Import libraries

import pandas as pd
import numpy as np
import scipy.stats as ss
import scipy.signal as sg
from plotnine import *
import pingouin as pg
import statsmodels.formula.api as smf
import statsmodels.api as sm


#%% Load data

datafile = 'assigned_data.csv'

end_tap = 200 # just in case, discard last taps
outsync_tap = 150 # observed change of behavior
outsync_threshold = 0.2

# metronome nominal data
bpm_df = pd.DataFrame({
	'Session':[1, 1, 1, 1, 1, 2, 2],
	'Metronome_label':[1, 2, 3, 4, 5, 1, 2],
	'bpm':[60, 68, 98, 114, 143, 60, 78]
	})
bpm_df['ISI_nominal'] = 60/bpm_df['bpm']
# set correct types
bpm_df[['Session','Metronome_label']] = bpm_df[['Session','Metronome_label']].astype('category')

# load data (leave out-of-sync regime out)
data_df = pd.read_csv(datafile) \
			.query('indice_tambor<=@outsync_tap') \
			.rename(columns={'indice_tambor':'Resp_number',
						'tiempo_tambor':'Resp_time',
						'amplitud_tambor':'Resp_amp',
						'indice_metronomo':'Stim_number',
						'tiempo_metronomo':'Stim_time',
						'diferencia':'Asynchrony',
						'repetido':'Repeated',
						'sin_asignacion':'Unassigned',
						'audio':'Session',
						'frecuencia':'Metronome_label'}) \
			.reset_index(drop=True)
# replace unassigned stimuli
data_df.loc[data_df['Unassigned']==1,
			['Resp_number', 'Resp_time', 'Resp_amp','Asynchrony']] = [-1,np.nan,np.nan,np.nan]
# replace unassigned responses
data_df.loc[data_df['Repeated']==1,
			['Stim_number', 'Stim_time','Asynchrony']] = [-1,np.nan,np.nan]
# merge nominal bpm information
data_df = pd.merge(
			    data_df,
			    bpm_df,
			    on=['Session', 'Metronome_label'],
			    how='inner')

# set correct types
data_df['Session'] = 'S' + data_df['Session'].map(str)
data_df[['Session','Metronome_label']] = data_df[['Session','Metronome_label']].astype('category')
data_df[['Resp_number','Stim_number']] = data_df[['Resp_number','Stim_number']].astype('int32')


# compute ITI and ISI
data_df = (data_df
		   # compute intertap interval (ITI) after ordering responses
		   .groupby('Session', as_index=False, observed=True)
		   .apply(lambda df: df.sort_values('Resp_number'))
		   .reset_index(drop=True)
		   .assign(
			   ITI = lambda df: df.groupby(['Session','Metronome_label'], as_index=False, observed=True)['Resp_time'].diff())
		   .reset_index(drop=True)
		   # compute interstimulus interval (ISI) after ordering stimuli
		   .groupby('Session', as_index=False, observed=True)
		   .apply(lambda df: df.sort_values('Stim_number'))
		   .reset_index(drop=True)
		   .assign(
			   ISI = lambda df: df.groupby(['Session','Metronome_label'], as_index=False, observed=True)['Stim_time'].diff())
		   .reset_index(drop=True)
		   )
# difference between produced and target intervals
data_df['Interval_diff'] = data_df['ITI'] - data_df['ISI_nominal']


# reshape to long for ANOVA and plotnine
data2_df = pd.melt(data_df,
				   id_vars = ['Session','Metronome_label','ISI_nominal','Resp_number',],
				   value_vars = ['Asynchrony','ITI','Interval_diff'],
				   var_name = 'Event_type',
				   value_name = 'Value')
data2_df['Event_type'] = (data2_df['Event_type'].astype('category')).cat.reorder_categories(['ITI','Asynchrony','Interval_diff'], ordered=True)

# summary data
data2_summary_df = (data2_df
					.groupby(['Session','Event_type'], as_index=False, observed=True)
					.agg(Mean = ('Value','mean'),
						 Std = ('Value','std'),
						 N = ('Value','count'))
					)
# standard error of the standard deviation: https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
data2_summary_df['Sterr_of_mean'] = data2_summary_df['Std']/np.sqrt(data2_summary_df['N'])
data2_summary_df['Sterr_of_std'] = data2_summary_df['Std']/np.sqrt(2*data2_summary_df['N']-1)

data2_summary_df['lower_CI_of_std'] = data2_summary_df['Std'] - 1.96*data2_summary_df['Sterr_of_std']
data2_summary_df['higher_CI_of_std'] = data2_summary_df['Std'] + 1.96*data2_summary_df['Sterr_of_std']




#%% Plot time series

x_lims = (0,end_tap)
marker_size = 0.5
fig_xsize = 10
fig_ysize = 6


plot_timeseries = (
 		 # ggplot(data2_df, # show all event types
 		 ggplot(data2_df.query("Event_type!='Interval_diff'"), # show only Asynchrony and ITI
				   aes(x = 'Resp_number', y = 'Value',
								group = 'Session',
							    color = 'Metronome_label'))
 		 + geom_line()
		 + geom_point(size = marker_size)
		 + geom_line(data2_df.query("Event_type=='ITI'"),
			   aes(x='Resp_number',y='ISI_nominal'), size=1)
		 + facet_grid(rows='Event_type',cols='Session', scales="free")
		 + labs(x = "Response number n",
				  y = 'ITI ($p_n$) and Asynchrony ($a_n$) (s)')
 		 + theme_bw()
 		 + theme(legend_position='none',
				strip_background_x=element_rect(fill='lightgray'),
				strip_background_y=element_rect(fill="lightgray"),
				panel_border=element_rect(color='black'),
		 		figure_size = (fig_xsize, fig_ysize))
		 )

print(plot_timeseries)
plot_timeseries.save('timeseries.pdf')
plot_timeseries.save('timeseries.png',dpi=150)


#%% statistics


# FDR-corrected post-hoc comparisons with Hedges'g effect size
print('==============')
print('Posthoc comparisons between sessions')
print('==============')
# Interval_diff changees sign between sessions, so we need to switch sign of Interval_diff to test whether it is smaller
data2_aux_df = (data2_df
				.assign(Value = np.select([(data2_df['Event_type']=='Interval_diff') & (data2_df['Session']==1)],
							  [-data2_df['Value']],
							  default=data2_df['Value']))
				)
posthoc = pg.pairwise_tests(data=data2_aux_df.query("Event_type!='ITI'"),
							dv='Value', between=['Event_type','Session'],
                            parametric=True, padjust='fdr_bh', effsize='eta-square', return_desc=True)
# select mean and CI from results
posthoc_df = pd.DataFrame(posthoc.to_dict())
asyn_S1_n = data2_df[(data2_df['Session']==1) & (data2_df['Event_type']=='Asynchrony')]['Value'].dropna().count()
asyn_S2_n = data2_df[(data2_df['Session']==2) & (data2_df['Event_type']=='Asynchrony')]['Value'].dropna().count()
intervaldiff_S1_n = data2_df[(data2_df['Session']==1) & (data2_df['Event_type']=='Interval_diff')]['Value'].dropna().count()
intervaldiff_S2_n = data2_df[(data2_df['Session']==2) & (data2_df['Event_type']=='Interval_diff')]['Value'].dropna().count()
posthoc_ave_ci_df = (posthoc_df
					 .query("Event_type=='Asynchrony' | Event_type=='Interval_diff'")
					 .loc[:,['Event_type','mean(A)','std(A)','mean(B)','std(B)','T','dof','p-corr','eta-square']]
					 .assign(n_A = [asyn_S1_n, intervaldiff_S1_n],
							 n_B = [asyn_S2_n, intervaldiff_S2_n])
					 )
posthoc_ave_ci_df['lower 95% CI (A)'] = posthoc_ave_ci_df['mean(A)'] - 1.96*posthoc_ave_ci_df['std(A)']/np.sqrt(posthoc_ave_ci_df['n_A'])
posthoc_ave_ci_df['upper 95% CI (A)'] = posthoc_ave_ci_df['mean(A)'] + 1.96*posthoc_ave_ci_df['std(A)']/np.sqrt(posthoc_ave_ci_df['n_A'])
posthoc_ave_ci_df['lower 95% CI (B)'] = posthoc_ave_ci_df['mean(B)'] - 1.96*posthoc_ave_ci_df['std(B)']/np.sqrt(posthoc_ave_ci_df['n_B'])
posthoc_ave_ci_df['upper 95% CI (B)'] = posthoc_ave_ci_df['mean(B)'] + 1.96*posthoc_ave_ci_df['std(B)']/np.sqrt(posthoc_ave_ci_df['n_B'])
posthoc_ave_ci_df['95% CI (A)'] = posthoc_ave_ci_df[['lower 95% CI (A)','upper 95% CI (A)']].values.tolist()
posthoc_ave_ci_df['95% CI (B)'] = posthoc_ave_ci_df[['lower 95% CI (B)','upper 95% CI (B)']].values.tolist()
# drop columns
posthoc_ave_ci_df = posthoc_ave_ci_df[['Event_type', 'mean(A)', '95% CI (A)',
									   'mean(B)', '95% CI (B)','T', 'dof','p-corr', 'eta-square']]
# Pretty printing of table
# pg.print_table(posthoc, floatfmt='.3g')
pg.print_table(posthoc_ave_ci_df, floatfmt='.3g')



# # lag-1 autocorrelation
# print('==============')
# print('Lag-1 autocorrelation (session average)')
# print('==============')
# lag1_autocorrelation = (data_df
# 						.groupby(['Session','Metronome_label'], as_index=False, observed=True)[['Asynchrony','ITI']]
# 						.agg(lambda x: ss.pearsonr(x.dropna()[0:-1].values,x.dropna()[1:].values).statistic)
# 						)
# lag1_autocorrelation_persession = (lag1_autocorrelation
# 								   .groupby(['Session'], as_index=False, observed=True)
# 								   .agg(Asynchrony = ('Asynchrony','mean'),
# 										   ITI = ('ITI','mean'))
# 								   )
# pg.print_table(lag1_autocorrelation_persession)



# print('==============')
# print('Linear regression')
# print('==============')
# model = smf.ols('Value ~ Event_type + Session + Event_type:Session', data=data2_df).fit()
# print(model.summary())
# print('\nSession comparison (interaction effects not supported):')
# pg.print_table(model_comps.result_frame)
# model_anova = sm.stats.anova_lm(model)
# model_comps = model.t_test_pairwise('Session')
# pg.print_table(model_anova.reset_index(drop=False,names='Parameter'))



print('==============')
print('Test of difference between variances')
print('==============')
data_asyn_df = data2_df.query("Event_type=='Asynchrony'")[['Session','Value']].dropna()
degreesf_asyn_levene = [2-1,len(data_asyn_df)-2]
levene_Asyn_res = pg.homoscedasticity(data=data_asyn_df,
									  dv='Value', group='Session',nan_policy='omit')
levene_Asyn_res[['df1','df2']] = degreesf_asyn_levene

data_IDiff_df = data2_df.query("Event_type=='ITI'")[['Session','Value']].dropna()
degreesf_IDiff_levene = [2-1,len(data_IDiff_df)-2]
levene_IntervalDiff_res = pg.homoscedasticity(data=data_IDiff_df,
											  dv='Value', group='Session',nan_policy='omit')
levene_IntervalDiff_res[['df1','df2']] = degreesf_IDiff_levene


summary_std = data2_summary_df[data2_summary_df['Event_type']!='ITI'][['Session','Event_type','Std','lower_CI_of_std','higher_CI_of_std']] \
				.pivot(index='Event_type',columns='Session',values=['Std','lower_CI_of_std','higher_CI_of_std']) \
				.reset_index(drop=False)


pg.print_table(summary_std)
print("\nLevene's test for Asyn:")
print(levene_Asyn_res)
print("\nLevene's test for Interval_diff:")
print(levene_IntervalDiff_res)


#%%

