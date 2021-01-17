import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas
import seaborn as sns
import streamlit


# Shapes; https://graphviz.org/doc/info/shapes.html
node_fmt = {
    "model": {
        "shape": "box",
        "fillcolor": "white",
        "fontcolor": "black",
        "color": "black",
        "style": "filled",
    },
    "test": {
        "shape": "ellipses",
        "fillcolor": "yellow",
        "fontcolor": "black",
        "color": "black",
        "style": "filled",
    },
    "source": {
        "shape": "cds",
        "fillcolor": "white",
        "fontcolor": "black",
        "color": "blue",
        "style": "filled",
    },
    "seed": {
        "shape": "cds",
        "fillcolor": "white",
        "fontcolor": "black",
        "color": "blue",
        "style": "filled",
    },
    "snapshot": {
        "shape": "component",
        "fillcolor": "yellow",
        "fontcolor": "black",
        "color": "black",
        "style": "filled",
    },
    "analysis": {
        "shape": "note",
        "fillcolor": "yellow",
        "fontcolor": "black",
        "color": "black",
        "style": "filled",
    },
    "operation": {
        "shape": "diamond",
        "fillcolor": "yellow",
        "fontcolor": "black",
        "color": "black",
        "style": "filled",
    },
    "exposure": {
        "shape": "diamond",
        "fillcolor": "yellow",
        "fontcolor": "black",
        "color": "black",
        "style": "filled",
    },
}

def read_graph(p="example_manifest.json"):
    with open(p, "r") as f:
        manifest = json.loads(f.read())

    G = nx.DiGraph()
    for n, d in manifest["nodes"].items():
        G.add_node(n, **d)
    for n, d in manifest["sources"].items():
        G.add_node(n, **d)
    for n, d in manifest["exposures"].items():
        G.add_node(n, **d)

    for n, children in manifest["child_map"].items():
        for child in children:
            G.add_edge(n, child)
    for n, parents in manifest["parent_map"].items():
        for parent in parents:
            G.add_edge(parent, n)
    
    return G


def extract_data_graph(G):
    "Returns subset of graph containing data nodes"
    data_resources = ["seed", "source", "model", "analysis", "snapshot"]
    data_nodes = [
        n for n, e in G.nodes(data=True) if e.get("resource_type") in data_resources
    ]
    return G.subgraph(data_nodes)


def extract_node_graph(G, node):
    "Returns subgraph containing selected node's ancestors and descendants."
    descendants = [n for n in nx.dag.descendants(G, node)]
    ancestors = [n for n in nx.dag.ancestors(G, node)]
    selected_nodes = [node] + ancestors + descendants
    return G.subgraph(selected_nodes)


def analyze_data_graph(G):
    df = pandas.DataFrame.from_records(
        {n: e for n, e in G.nodes(data=True)}
    ).transpose()
    df["degree"] = pandas.Series({n: d for n, d in nx.degree(G)})
    df["centrality"] = pandas.Series(nx.betweenness_centrality(G))
    # connectivity = pandas.DataFrame(nx.node_connectivity(G), columns=['node_id', 'connectivity']).set_index('node_id')
    df["clustering"] = pandas.Series(nx.clustering(G))

    return df


def create_pydot_viz(G, selected_node, visible_node_types, max_node_distance):
    node_list = []
    node_str = ""
    for n, e in G.nodes(data=True):
        try:
            dist = nx.shortest_path_length(G, n, selected_node)
        except:
            dist = nx.shortest_path_length(G, selected_node, n)
        if dist > max_node_distance or e.get("resource_type") not in visible_node_types:
            continue
        formatting = node_fmt[e.get("resource_type")].copy()
        formatting["label"] = (
            f"{e.get('resource_type')}.{e.get('name')}"
            if e.get("resource_type") != "model"
            else e.get("name")
        )
        if n == selected_node:
            formatting["fillcolor"] = "green"
        format_str = " ".join(f'{k}="{v}"' for k, v in formatting.items())
        node_str += f'"{n}" [{format_str}]\n'
        node_list.append(n)

    g = G.subgraph(node_list)
    edge_str = ""
    for u, v, e in g.edges(data=True):
        edge_str += f'"{u}" -> "{v}" [label=""]\n'

    dot_viz = f"""
    digraph models {{
        rankdir="LR"
        nodesep=0.1
        graph [margin=0 ratio=auto size=10]
        node [fontsize=10 height=0.25]
        edge [fontsize=10]
        {node_str}
        {edge_str}
    }}
    """
    return dot_viz


def get_relations_df(G, n):
    data = {}
    descendants = [("descendant", n) for n in nx.dag.descendants(G, n)]
    ancestors = [("ancestor", n) for n in nx.dag.ancestors(G, n)]
    for r, target in ancestors:
        dist = nx.shortest_path_length(G, target, n)
        data[target] = {"distance": dist, "relationship": r}
    for r, target in descendants:
        dist = nx.shortest_path_length(G, n, target)
        data[target] = {"distance": dist, "relationship": r}
    return pandas.DataFrame.from_dict(data, orient="index")


streamlit.set_page_config(page_title="dbt Graph Analysis", layout="centered")


# Prep work
G = read_graph()
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
max_node_distance = streamlit.slider(label="Max Relation Distance", min_value=1, max_value=10, value=10)

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
