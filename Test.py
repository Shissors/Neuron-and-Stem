import pandas as pd #to generate and handle dataframes
import numpy as np# to calculate log values
import scipy.stats as stats #to perform the one sample t test
from statsmodels.stats.multitest import multipletests #to calculate FDR values
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("metadata.csv") #using pandas to read the csv files, contains sample ID's, age, infection type and gender 
df_gpl = pd.read_csv("GPL1261-56135.txt", sep="\t", comment="#", low_memory=False) #using pandas to read the text file, seperating each value based on tab space, ignores lines staring with "#" and low memory=false enables handling multiple data types, this contains the gene symbols, ontology, etc.
df_gpl['Gene Symbol'] = df_gpl['Gene Symbol'].str.split('///').str[0]


df_stem=df[df["condition"]=="stem"]
df_neuron=df[df["condition"]=="neuron"]


id_stem=df_stem["sample"].to_list() 
id_neuron=df_neuron["sample"].to_list()


df1=pd.read_csv("GSE12499_series_matrix.txt",sep="\t",index_col=0)
df2=pd.read_csv("GSE60905_series_matrix.txt",sep="\t",index_col=0)


df1=df1.iloc[:-1]
df2=df2.iloc[:-1]
df_stem_subset = df1
df_neuron_subset=df2

def perform_ttest(stem_values, neuron_values):
    t_stat, p_value = stats.ttest_ind(stem_values, neuron_values, nan_policy="omit")
    return p_value
df_results = pd.DataFrame(index=df_stem_subset.index)
df_results['p_value'] = df_stem_subset.apply(lambda row: perform_ttest(row, df_neuron_subset.loc[row.name]), axis=1)

_, df_results["fdr_corrected_p_value"], _, _ = multipletests(df_results["p_value"], method="fdr_bh")
significant_genes = df_results[df_results["fdr_corrected_p_value"] < 0.01]



df_stem_mean = df_stem_subset.mean(axis=1)
df_neuron_mean = df_neuron_subset.mean(axis=1)
df_results['log2FC'] = np.log2(df_stem_mean / df_neuron_mean)
df_results['gene_name'] = df_results.index
top_10_genes = df_results.nsmallest(10, 'fdr_corrected_p_value')


fig = px.scatter(df_results, 
                 x='log2FC', 
                 y=-np.log10(df_results['fdr_corrected_p_value']), 
                 color='fdr_corrected_p_value', 
                 color_continuous_scale='viridis', 
                 title='Volcano Plot of Differential Gene Expression',
                 hover_data=["gene_name"])

fig.update_layout(
    yaxis_title='-Log10 FDR-corrected p-value',
    xaxis_title='Log2 Fold Change'
)
fig.write_html("top_10_genes_plot.html")


col=["ID","Gene Symbol", "Gene Ontology Biological Process", "Gene Ontology Cellular Component","Gene Ontology Molecular Function"]
df_gpl=df_gpl[col]

gene_id_list = top_10_genes['gene_name'].tolist()

matching_rows = df_gpl[df_gpl['ID'].isin(gene_id_list)]

heatmap_data = df_stem_subset.loc[top_10_genes.index].join(df_neuron_subset.loc[top_10_genes.index])

if 'gene_name' in top_10_genes.columns:
    heatmap_data.index = top_10_genes['gene_name'].values


merged_data = heatmap_data.copy()
merged_data['Gene Symbol'] = merged_data.index.map(matching_rows.set_index('ID')['Gene Symbol'])
merged_data.set_index('Gene Symbol', inplace=True)

fig = px.imshow(
    merged_data,
    labels=dict(x="Samples", y="Genes", color="Expression"),
    x=merged_data.columns,
    y=merged_data.index,
    color_continuous_scale="viridis"
)
fig.update_layout(title="Heatmap of Top 10 Most Significant Genes", autosize=False, width=800, height=600)
fig.write_html("heatmap.html")
matching_rows.to_csv("matching_genes.tsv", sep="\t", index=False)



df_combined = pd.concat([df_stem_subset, df_neuron_subset], axis=1).T
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_combined)
pca = PCA(n_components=3)
pca_results = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(pca_results, columns=['PC1', 'PC2', 'PC3'])
df_pca['Condition'] = ['Stem'] * len(df_stem_subset.columns) + ['Neuron'] * len(df_neuron_subset.columns)


fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Condition', 
                     title="3D PCA of Stem vs Neuron Samples",
                     labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
                     color_discrete_map={'Stem': 'blue', 'Neuron': 'red'})

fig.write_html("pca_3d_plot.html")


print(merged_data)