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

conditions = df_genes['condition']
df_pca = df_genes.drop(columns=['condition'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_pca)
pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_results, columns=["PC1", "PC2"])
pca_df["condition"] = df_genes["condition"].values
print(pca_df)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="condition", palette="Set1", alpha=0.8)
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.title("PCA of Gene Expression Data")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.legend(title="Condition")
plt.tight_layout()
plt.savefig("PCA.png", dpi=300)

df_genes['condition_encoded'] = df_genes['condition'].map({'stem': 0, 'neuron': 1})
df_dea = df_genes.drop(columns=['condition'])

results = []
for gene in df_dea.columns:
    gene_expression = df_dea[gene]
    gene_expression = pd.to_numeric(gene_expression, errors='coerce').dropna()
    conditions_aligned = df_genes['condition_encoded'].loc[gene_expression.index]
    X = pd.DataFrame({'condition': conditions_aligned})
    X = sm.add_constant(X)

    model = sm.OLS(gene_expression, X)
    results_gene = model.fit()
    p_value = results_gene.pvalues['condition']
    results.append([gene, p_value])
results_df = pd.DataFrame(results, columns=['Gene', 'p_value'])
results_df['fdr'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
significant_genes = results_df[results_df['fdr'] < 0.05]
print(significant_genes)
print(pca_df)

