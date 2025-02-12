import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



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
df_combined_mod=df_combined.drop(["condition","location","day"], axis=1)
df_combined_mod=df_combined_mod.reset_index(drop=True)
df_combined_mod = df_combined_mod.set_index('index')
df_combined_mod=df_combined_mod.T
mean_expression=df_combined_mod.mean(axis=1)
df_combined_mod["mean expression"]=mean_expression
df_combined_mod=df_combined_mod.T
df_combined.set_index("index", inplace=True)
selected=df_combined[["condition", "location","day"]]
df_final=pd.concat([df_combined_mod, selected], axis=1)
print(df_final)

#separate the dataframes and calculate the means 