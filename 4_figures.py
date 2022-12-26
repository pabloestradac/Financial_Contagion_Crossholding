# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from functions import cascade_failures, result_failures


sns.set_style("whitegrid")
plt.rc('text', usetex=True)

data_path = "../0_data/"
read_path = "../2_pipeline/2_crossholding_analysis/"
fig_path = "../3_results/figures/"
year = 2016
dpi = 500

# Read Graph Data
G = nx.read_gpickle(read_path + "graph_crossholding_{}.gpickle".format(year))
# Creating C matrix
C = nx.adjacency_matrix(G).todense()
C = np.array(C)
# Creating C_hat matrix
Chat = np.diag(1 - sum(C))
# N and names
n = len(G)   # Number of firms
names = np.array(G.nodes)
# Read Graph of Industries
df_edges_industry = pd.read_csv(read_path
                                + "edgelist_industry{}.csv".format(year))
Gind = nx.from_pandas_edgelist(df_edges_industry,
                               "level_0_sector", "level_1_sector",
                               edge_attr=True, create_using=nx.DiGraph)
# Define DP
p = np.array(list(nx.get_node_attributes(G, "Total_Assets").values()))
p = p.reshape(len(p), 1)
# Normalize assets
min_value = min(p)
pnorm = p / min_value[0]
# Define A (dependency matrix)
A = np.dot(Chat, np.linalg.inv(np.eye(n) - C))
# Creating C matrix at industry level
C_ind = nx.adjacency_matrix(Gind, weight="pct_share").toarray()
Chat_ind = np.diag(1 - sum(C_ind))
# Define A (dependency matrix) at industry level
A_ind = np.dot(Chat_ind, np.linalg.inv(np.eye(8) - C_ind))
# Threshold change
df_threshold = pd.read_csv(read_path + "threshold_change{}.csv".format(year),
                           index_col="Threshold")
# Industry shocks
df_ind_shock = pd.read_csv(read_path + "industry_shocks{}.csv".format(year))
# Nodes dataframe
df_nodes = pd.read_csv(read_path + "df_nodes.csv", dtype={"Ruc": "str"})
ruc_id = dict(zip(df_nodes.Ruc, df_nodes.ID))


# FIGURE 3
# Histogram of Diversification
out_deg = pd.Series(list(dict(G.out_degree(weight=None)).values()))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
out_deg.plot(kind="hist", grid="on", label="Out-degree distribution")
plt.axvline(x=out_deg.mean(), color="black", linewidth=3, linestyle="--",
            label='Average out-degree = {:.4f}'.format(out_deg.mean()))
plt.ylabel('Total\nFirms', fontsize=20, rotation=0, labelpad=75)
plt.xlabel('Out-degree', fontsize=20, rotation=0, labelpad=15)
plt.yticks(fontsize=15)
plt.xticks(range(0, 11), fontsize=15)
plt.legend(fontsize=20, fancybox=True, framealpha=0.5)
plt.savefig(fig_path + "hist_diversification_{}.png".format(year),
            dpi=dpi,
            bbox_inches='tight',
            transparent=True)
plt.close()


# Figure 4
# Histogram of Integration
sum_shares = pd.Series(C.sum(axis=0))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
sum_shares.plot(kind="hist", grid="on", label="Integration distribution")
plt.axvline(x=sum_shares.mean(), color="black", linewidth=3, linestyle="--",
            label='Average Integration = {:.4f}'.format(sum_shares.mean()))
plt.ylabel('Total\nFirms', fontsize=20, rotation=0, labelpad=75)
plt.xlabel('Integration', fontsize=20, rotation=0, labelpad=15)
plt.yticks(fontsize=15)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=15)
plt.legend(fontsize=20, fancybox=True, framealpha=0.5)
plt.savefig(fig_path + "hist_integration_{}.png".format(year),
            dpi=dpi,
            bbox_inches='tight',
            transparent=True)
plt.close()


# FIGURE 1
# Heatmap of Cross-holding Industry
df_crossholding_industry = pd.DataFrame(data=C_ind, index=Gind.nodes,
                                        columns=Gind.nodes) * 100
sns.set(font_scale=2)
fig, ax = plt.subplots(figsize=(15, 9))
sns.heatmap(
    ax=ax, data=df_crossholding_industry.round(2),
    vmin=0, vmax=df_crossholding_industry.max().max(), center=-3,
    cmap=sns.diverging_palette(0, 20, n=200), annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
plt.savefig(fig_path + "crossholding_industry_{}.png".format(year),
            dpi=dpi,
            bbox_inches='tight',
            transparent=True)
plt.close()


# FIGURE 2
# Heatmap of Dependency Industry
df_dependency_industry = pd.DataFrame(data=A_ind, index=Gind.nodes,
                                      columns=Gind.nodes) * 100
sns.set(font_scale=2)
fig, ax = plt.subplots(figsize=(15, 9))
sns.heatmap(
    ax=ax, data=df_dependency_industry.round(1),
    vmin=0, vmax=df_dependency_industry.max().max(), center=-3,
    cmap=sns.diverging_palette(110, 120, n=100),
    annot=True, fmt='.2f')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
plt.savefig(fig_path + "dependency_industry_{}.png".format(year),
            dpi=dpi,
            bbox_inches='tight',
            transparent=True)
plt.close()


# FIGURE 5
# Plot of threshold change
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
df_threshold.plot(kind="line", ax=ax, linewidth=3, grid="on", label="")
plt.ylabel('Total\nFirms', fontsize=20, rotation=0, labelpad=75)
plt.xlabel('Threshold', fontsize=20, rotation=0, labelpad=15)
plt.yticks(fontsize=18)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=18)
plt.legend(fontsize=20, fancybox=True, framealpha=0.5)
plt.savefig(fig_path + "simulations_firms_{}.png".format(year),
            dpi=dpi,
            bbox_inches='tight',
            transparent=True)
plt.close()


# FIGURE 6 - 9
# Dictionary of initials for each economics sector
econ_sector = nx.get_node_attributes(G, "Economic_Sector")
initial_sector = {'Trade': "T",
                  'General Services': "S",
                  'Real Estate': "E",
                  'Financial Activities': "F",
                  'Manufacturing': "M",
                  'Construction': "C",
                  'Agriculture': "A",
                  'Transportation': "B"}
t = [0.85, 0.9, 0.95]
d = [0, 0.5]
c = [0.20, 0.30]
final_list_firms = set()
cut = 20

for i in range(len(c)):

    for j in range(len(d)):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 20),
                                 sharey=True)
        axes = axes.reshape(-1)
        plt.subplots_adjust(hspace=0.4)

        for k in range(len(t)):
            df = result_failures(t[k], d[j], c[i], econ_sector,
                                 G, A, pnorm, names, n, group=False)
            df['ID'] = df["Firms"].map(ruc_id)
            df['ID'] = df['ID'].astype("str")
            df = df.sort_values("Total_Affected", ascending=False).head(cut)
            df["Industry"] = df["Firms"].map(econ_sector)
            df["Initial_Industry"] = df["Industry"].map(initial_sector)
            final_list_firms.update(df["ID"].unique())
            bar = axes[k].bar(df["ID"], df["Total_Affected"],
                              color="LightGrey", edgecolor="Black")
            axes[k].set_title("Threshold: " + r'$\theta = {}$'.format(t[k]),
                              fontsize=20)
            axes[k].set_ylabel('Total \n Firms \n Affected', fontsize=15,
                               rotation=0, labelpad=50)
            axes[k].set_xlabel('Firm ID', fontsize=15, rotation=0, labelpad=12)
            axes[k].tick_params(labelsize=12)
            axes[k].set_yticks(np.arange(0, 21, 2))
            # Add initial sector above the bar
            ii = 0
            for rect in bar:
                height = rect.get_height()
                initial = df["Initial_Industry"].iloc[ii]
                axes[k].text(rect.get_x() + rect.get_width()/2.0, height,
                             initial, ha='center', va='bottom')
                ii += 1
        from matplotlib.lines import Line2D
        legend_elements = legend_elements = [Line2D([0], [0], color='k', ls="", label='T: Trade'),
                                             Line2D([0], [0], color='k', ls="", label='S: General Services'),
                                             Line2D([0], [0], color='k', ls="", label='E: Real Estate'),
                                             Line2D([0], [0], color='k', ls="", label='F: Financial Activities'),
                                             Line2D([0], [0], color='k', ls="", label='M: Manufacturing'),
                                             Line2D([0], [0], color='k', ls="", label='C: Construction'),
                                             Line2D([0], [0], color='k', ls="", label='A: Agriculture'),
                                             Line2D([0], [0], color='k', ls="", label='B: Transportation')]
        plt.legend(handles=legend_elements, fontsize=15, fancybox=True, framealpha=0.5, bbox_to_anchor=(1, -0.3), ncol=4)

        name = "important_firms_drop{:.0f}_cost{:.0f}_year{}.png".format((1 - d[j])*100, c[i]*100, year)
        plt.savefig(fig_path + name,
                    dpi=dpi,
                    bbox_inches='tight',
                    transparent=True)
        plt.close()


# FIGURE 10
# Plot of simulations by industry
# Dictionary of colors for each economics sector
color_sector = {'Trade': "Grey",
                'Manufacturing': "Black",
                'General Services': "Black",
                'Real Estate': "Grey",
                'Financial Activities': "Black",
                'Construction': "Grey",
                'Agriculture': "Grey",
                'Transportation': "Black"}
# Dictionary of styles for each economics sector
style_sector = {'Trade': ":",
                'Manufacturing': "-",
                'General Services': "-.",
                'Real Estate': "--",
                'Financial Activities': ":",
                'Construction': "-",
                'Agriculture': "-.",
                'Transportation': "--"}
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
for key, group in df_ind_shock.groupby(["Firms"]):
    ax = group.plot(ax=ax, kind='line', x="Drop_Value", y="Total_Affected",
                    style=style_sector[key], label=key, linewidth=2,
                    color=color_sector[key], legend=None)
plt.text(0.9, 43, "Manufacturing", fontsize=14)
plt.text(0.9, 38, "Trade", fontsize=14)
plt.text(0.9, 28, "General Services", fontsize=14)
plt.text(0.9, 19, "Financial Activities", fontsize=14)
plt.text(0.9, 14, "Real State", fontsize=14)
plt.text(0.9, 10, "Agriculture", fontsize=14)
plt.text(0.9, 7, "Transportation", fontsize=14)
plt.text(0.9, 4, "Construction", fontsize=14)
plt.ylabel('Total\nFirms', fontsize=20, rotation=0, labelpad=75)
plt.xlabel('Drop in Market Value', fontsize=20, rotation=0, labelpad=15)
plt.yticks(np.arange(0, 50, 10), fontsize=18)
plt.xticks(np.arange(0, 1.2, 0.1), fontsize=18)
plt.savefig(fig_path + "simulations_industry_{}.png".format(year),
            dpi=dpi,
            bbox_inches='tight',
            transparent=True)
plt.close()


# FIGURE 11 - 12
# paint contagious firm
G_plot = G.reverse()
data = pd.DataFrame(list(G_plot.nodes), columns=["Node"])
contagious_firms = ['1790008967001',  '1790016919001']
color_type = {"infected": "dimgrey", "safe": "lightgrey", "seed": "black"}

for f in contagious_firms:
    data["Contagious"] = "safe"
    f_index = np.where(names == f)[0]
    results = cascade_failures(A, 0.95, pnorm, f_index, 0, 0.3)
    result_firms, result_costs, wave_firms_list = results

    for waves in np.arange(len(wave_firms_list)):
        if waves == 0:
            data["Contagious"][wave_firms_list[waves]] = "seed"
        else:
            data["Contagious"][wave_firms_list[waves]] = "infected"

    # Creating dictionary of attributes
    node_contag = data.set_index("Node").Contagious.to_dict()
    # Adding attributes to firms
    nx.set_node_attributes(G_plot, node_contag, name="Contagious_"+f)
    nodes_c = nx.shortest_path(G_plot, f).keys()
    S = G_plot.subgraph(nodes_c)
    contagious = nx.get_node_attributes(S, "Contagious_"+f)
    cat_colors = pd.Series(list(contagious.values())).map(color_type).values
    # Draw Graph
    pos = nx.spring_layout(S)
    plt.figure(figsize=(10, 10))
    nx.draw(S, pos, node_color=cat_colors, node_size=500)
    plt.savefig(fig_path + "contagious_{}.png".format(f),
                dpi=dpi,
                bbox_inches='tight',
                transparent=True)
    plt.close()
