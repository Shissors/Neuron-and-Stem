import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

df_stem = pd.read_csv("GSE60905_series_matrix.txt", comment="!", sep="\t", index_col=0)
df_neuron = pd.read_csv("GSE12499_series_matrix.txt", comment="!", sep="\t", index_col=0)
meta_df=pd.read_csv("metadata.tsv", sep="\t")

common_genes = df_stem.index.intersection(df_neuron.index)

df_stem = df_stem.loc[common_genes]
df_neuron = df_neuron.loc[common_genes]
df_combined = pd.concat([df_stem, df_neuron], axis=1)
df_combined.to_csv("merged_expression_data.tsv", sep="\t")


metadata_reset = meta_df.set_index('sample')
df_combined_reset = df_combined.T
df_combined_reset['condition'] = df_combined_reset.index.map(metadata_reset['condition'])
df_genes=df_combined_reset
print(df_combined_reset)

    