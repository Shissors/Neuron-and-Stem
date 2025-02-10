import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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


df_long = pd.melt(df_genes.iloc[:, :-1].T, var_name="Sample", value_name="Expression Level")
df_long["condition"] = df_long["Sample"].map(df_genes["condition"])



plt.figure(figsize=(12, 5))
sns.boxplot(data=df_long, x="Sample", y="Expression Level", hue="condition", palette="Set2")
plt.xticks([])
plt.title("Expression Distribution Across Samples")
plt.savefig("Boxplots of expression and samples.png", dpi=300)

print(df_genes)




def one_sample_ttest(row, condition_data):#does the same t tests for rsv samples
    condition_values=row[condition_data]
    t_stat, p_value = stats.ttest_1samp(condition_values, 0, nan_policy="omit")
    return p_value

df_stem = df_genes[df_genes["condition"] == "stem"]
df_neuron = df_genes[df_genes["condition"] == "neuron"]


results_stem = []
results_neuron = []

for gene in df_stem.columns[:-1]:  # Exclude the 'condition' column
    p_value = one_sample_ttest(df_stem[gene], df_stem)
    results_stem.append([gene, p_value])


for gene in df_neuron.columns[:-1]:  # Exclude the 'condition' column
    p_value = one_sample_ttest(df_neuron[gene], df_neuron)
    results_neuron.append([gene, p_value])

results_stem_df = pd.DataFrame(results_stem, columns=["Gene", "p-value"])
results_neuron_df = pd.DataFrame(results_neuron, columns=["Gene", "p-value"])
