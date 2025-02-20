import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


df_stem = pd.read_csv("GSE60905_series_matrix.txt", comment="!", sep="\t", index_col=0)
df_neuron = pd.read_csv("GSE12499_series_matrix.txt", comment="!", sep="\t", index_col=0)
meta_df = pd.read_csv("metadata.csv", index_col=0)
meta_df.index.name = None

common_genes = df_stem.index.intersection(df_neuron.index)

df_stem = df_stem.loc[common_genes]
df_neuron = df_neuron.loc[common_genes]
df_combined = pd.concat([df_stem, df_neuron], axis=1)
df_combined.to_csv("merged_expression_data.tsv", sep="\t")

df_combined_reset = df_combined.T
df_genes=df_combined_reset
df_merged=pd.merge(df_genes, meta_df, left_index=True,right_index=True)
df_merged=df_merged.reset_index()


df_for_pca=df_genes
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_for_pca)
pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_results, columns=["PC1", "PC2"])
df_merged_pca=pd.merge(df_merged, pca_df, left_index=True,right_index=True)
#important one


df_merged_stem=df_merged[df_merged["condition"]=="stem"]
df_merged_stem=df_merged_stem[df_merged_stem["day"]==4]
df_merged_stem_mod=df_merged_stem.drop(["condition","location","day","index"], axis=1)
scaled_data_stem=scaler.fit_transform(df_merged_stem_mod)
pca_s=PCA(n_components=2)
pca_results_s=pca_s.fit_transform(scaled_data_stem)
pca_results_s_df = pd.DataFrame(pca_results_s, columns=["PC1", "PC2"])
pca_s_df = pd.merge(df_merged_stem, pca_results_s_df, left_index=True, right_index=True, suffixes=('', '_pca'))



plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_s_df, x="PC1", y="PC2", hue="location", palette="Set1", alpha=0.8)
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.title("PCA of Stem cells")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.legend(title="Condition")
plt.tight_layout()
plt.savefig("PCstemloc.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_s_df, x="PC1", y="PC2", hue="day", palette="Set1", alpha=0.8)
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.title("PCA of Stem cells")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.legend(title="Condition")
plt.tight_layout()
plt.savefig("PCstemday.png", dpi=300)
plt.close()


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_merged_pca, x="PC1", y="PC2", hue="condition", palette="Set1", alpha=0.8)
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.title("PCA of Gene Expression Data")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.legend(title="Condition")
plt.tight_layout()
plt.savefig("PCAtotal.png", dpi=300)


df_merged_stem#contains only day 4
df_merged_neuron=df_merged_pca[df_merged_pca["condition"]=="neuron"]
df_merged_neuron=df_merged_neuron.drop(["PC1","PC2"], axis=1)



df_combined = pd.concat([df_merged_stem, df_merged_neuron], axis=0)
df_stem_separated=df_combined[df_combined["condition"]=="stem"]
df_combined_mod=df_stem_separated.drop(["condition","location","day"], axis=1)
df_combined_mod=df_combined_mod.reset_index(drop=True)
df_combined_mod = df_combined_mod.set_index('index')
df_combined_mod=df_combined_mod.T
mean_expression=df_combined_mod.mean(axis=1)
df_combined_mod["mean expression stem"]=mean_expression
df_combined_mod=df_combined_mod.T
df_combined.set_index("index", inplace=True)
selected=df_combined[["condition", "location","day"]]
df_final=pd.concat([df_combined_mod, selected], axis=1)
df_final=df_final.dropna()
df_final.loc["mean expression stem"]=mean_expression

#separate the dataframes and calculate the means 
df_combined = pd.concat([df_merged_stem, df_merged_neuron], axis=0)
df_neuron_separated=df_combined[df_combined["condition"]=="neuron"]
df_combined_n=df_neuron_separated.drop(["condition","location","day"], axis=1)
df_combined_n=df_combined_n.reset_index(drop=True)
df_combined_n = df_combined_n.set_index('index')
df_combined_n=df_combined_n.T
mean_expression_n=df_combined_n.mean(axis=1)
df_combined_n["mean expression neuron"]=mean_expression_n
df_combined_n=df_combined_n.T
df_combined.set_index("index", inplace=True)
selectedn=df_combined[["condition", "location","day"]]
df_finaln=pd.concat([df_combined_n, selectedn], axis=1)
df_finaln.loc["mean expression neuron"]=mean_expression_n
df_finaln = df_finaln.drop(df_finaln.index[11:])
selected_row_n = df_finaln.loc["mean expression neuron"]
selected_row_s=df_final.loc["mean expression stem"]
df_combined.loc["mean expression stem"]=selected_row_s
df_combined.loc["mean expression neuron"]=selected_row_n
df_combined=df_combined.T

meta_df=meta_df.reset_index()
meta_df_stem=meta_df[meta_df["condition"]=="stem"]
ID_stem=meta_df_stem["index"].tolist()

meta_df_neuron=meta_df[meta_df["condition"]=="neuron"]
ID_neuron=meta_df_neuron["index"].tolist()
df_no_mean= df_combined.iloc[:, :-2]
df_no_mean= df_no_mean.drop(df_no_mean.index[-3:])
df_o_stem=df_combined[ID_stem]
df_o_neuron=df_combined[ID_neuron]
ID_combined=pd.concat([df_o_stem,df_o_neuron], axis=1)
print(ID_combined)



def two_sample_t_test(row):
    stem_values=row[ID_stem]
    neuron_values=row[ID_neuron]
    t_stat, p_value=stats.ttest_ind(neuron_values, stem_values, nan_policy="omit")
    return p_value
df_combined["p_value"]=df_no_mean.apply(two_sample_t_test, axis=1)
_, fdr_values, _, _=multipletests(df_combined["p_value"], method="fdr_bh")
df_combined["fdr values"]=fdr_values


df_log_neuron=df_combined[ID_neuron]
df_log_stem=df_combined[ID_stem]
df_combined["mean log ratio"]=df_log_neuron.mean(axis=1)-df_log_stem.mean(axis=1)