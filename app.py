import matplotlib.pyplot as plt
import networkx as nx
import pandas
import seaborn as sns
import streamlit

from analysis import (
    node_fmt,
    read_graph,
    extract_data_graph,
    extract_node_graph,
    analyze_data_graph,
    create_pydot_viz,
    get_relations_df,
)


streamlit.set_page_config(page_title="dbt Graph Analysis", layout="centered")
manifest_file = streamlit.sidebar.file_uploader("Manifest File", type=[".json"]) or None

# Prep work
G = read_graph(manifest_file)
g_data = extract_data_graph(G)
metrics = analyze_data_graph(g_data)
default_selected_model = (
    metrics.loc[metrics.resource_type == "model"]
    .degree.sort_values(ascending=False)
    .index[0]
)

# Main charts
streamlit.title("dbt Graph Analysis")
streamlit.markdown(
    """
    This Streamlit application reads in a dbt_ graph, performs some light
    network analysis at the global level, and then provides functionality
    for exploring individual nodes and their dependencies. 
"""
)

streamlit.header("Global Analysis")
streamlit.markdown(
    """
    Below is a summary of the nodes contained in this graph.
"""
)
streamlit.dataframe(
    metrics.reset_index().pivot_table(
        index="resource_type",
        columns="package_name",
        values="unique_id",
        aggfunc="count",
        fill_value=0,
    )
)

streamlit.header("Node Detail")
streamlit.subheader("Node Viz Options")
visible_node_types = streamlit.multiselect(
    label="dbt Node Resource Types",
    options=list(node_fmt.keys()),
    default=["seed", "source", "model"],
)
available_nodes = metrics.loc[
    metrics.resource_type.isin(visible_node_types)
].index.tolist()
default_selected_node = (
    metrics.loc[available_nodes].centrality.sort_values(ascending=False).index[0]
)
selected_node = streamlit.selectbox(
    label="Which node?",
    options=available_nodes,
    index=available_nodes.index(default_selected_node),
)
max_node_distance = streamlit.slider(
    label="Max Relation Distance", min_value=1, max_value=10, value=10
)

# Derivative analysis
g_node = extract_node_graph(G, selected_node)
related_node_distances = get_relations_df(g_node, selected_node)
related_nodes = metrics.join(related_node_distances, how="inner")

with streamlit.beta_container():
    streamlit.markdown(
        f"""
        The selected node has the following properties:

        - Node degree: {metrics.loc[selected_node, "degree"]}
        - Node degree rank: {metrics.degree.rank(pct=True)[selected_node]}
        - Betweenness Centrality rank: {metrics.centrality.rank(pct=True)[selected_node]}
        - Clustering: {metrics.loc[selected_node, "clustering"]}
        """
    )

dot_viz = create_pydot_viz(g_node, selected_node, visible_node_types, max_node_distance)
streamlit.graphviz_chart(dot_viz, use_container_width=True)

with streamlit.beta_container():
    streamlit.subheader("Ancestor and Descendant Summary")
    streamlit.markdown(
        f"""
        The below section lists the nodes that are ancestors or descendants of the
        selected node, and their distance from the selected node.

        #### Count of nodes at at varying distances from target node
        """
    )
    sns.countplot(x="distance", hue="relationship", data=related_nodes)
    fig = plt.gcf()
    streamlit.pyplot(fig)
    plt.clf()

    streamlit.markdown(
        """
        The following tags were found on the related nodes.
        """
    )
    sns.countplot(y="tags", hue="relationship", data=related_nodes.explode("tags"))
    fig = plt.gcf()
    streamlit.pyplot(fig)
    plt.clf()


with streamlit.beta_container():
    streamlit.subheader("Raw Data")
    display_cols = [
        "relationship",
        "name",
        "tags",
        "degree",
        "centrality",
        "clustering",
        "distance",
    ]
    streamlit.dataframe(
        related_nodes.loc[:, display_cols]
        .sort_values(by="relationship")
        .reset_index(drop=True)
    )
