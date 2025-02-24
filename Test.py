import pandas as pd #to generate and handle dataframes
import numpy as np# to calculate log values
import scipy.stats as stats #to perform the one sample t test
from statsmodels.stats.multitest import multipletests #to calculate FDR values
import plotly.express as px
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
df_stem_subset = df1.iloc[:, :10]
df_neuron_subset=df2.iloc[:, :10]

def perform_ttest(stem_values, neuron_values):
    t_stat, p_value = stats.ttest_ind(stem_values, neuron_values, nan_policy="omit")
    return p_value
df_results = pd.DataFrame(index=df_stem_subset.index)
df_results['p_value'] = df_stem_subset.apply(lambda row: perform_ttest(row, df_neuron_subset.loc[row.name]), axis=1)

_, df_results["fdr_corrected_p_value"], _, _ = multipletests(df_results["p_value"], method="fdr_bh")
significant_genes = df_results[df_results["fdr_corrected_p_value"] < 0.05]
print(significant_genes)


df_stem_mean = df_stem_subset.mean(axis=1)
df_neuron_mean = df_neuron_subset.mean(axis=1)
df_results['log2FC'] = np.log2(df_stem_mean / df_neuron_mean)
df_results['gene_name'] = df_results.index
top_10_genes = df_results.nsmallest(10, 'fdr_corrected_p_value')
print(top_10_genes)

fig = px.scatter(df_results, 
                 x='log2FC', 
                 y=-np.log10(df_results['p_value']), 
                 color='fdr_corrected_p_value', 
                 color_continuous_scale='viridis', 
                 labels={'log2FC': 'Log2 Fold Change', '-np.log10(p_value)': '-Log10 p-value'},
                 title='Volcano Plot of Differential Gene Expression',
                 hover_data=["gene_name"])
fig.write_html("top_10_genes_plot.html")


col=["ID","Gene Symbol", "Gene Ontology Biological Process", "Gene Ontology Cellular Component","Gene Ontology Molecular Function"]
df_gpl=df_gpl[col]

gene_id_list = top_10_genes['gene_name'].tolist()

matching_rows = df_gpl[df_gpl['ID'].isin(gene_id_list)]


heatmap_data = df_stem_subset.loc[top_10_genes.index].join(df_neuron_subset.loc[top_10_genes.index])

if 'gene_name' in top_10_genes.columns:
    heatmap_data.index = top_10_genes['gene_name'].values

fig = px.imshow(
    heatmap_data,
    labels=dict(x="Samples", y="Genes", color="Expression"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale="viridis"
)
fig.update_layout(title="Heatmap of Top 10 Most Significant Genes", autosize=False, width=800, height=600)
fig.write_html("heatmap.html")
matching_rows.to_csv("matching_genes.tsv", sep="\t", index=False)
   
    