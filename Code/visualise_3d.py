import plotly.graph_objs as go
import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes with multiple attributes (dummy data)
G.add_nodes_from([(1, {'pos': (0, 0, 0), 'info': {'label': 'Node 1', 'color': 'blue'}}),
                  (2, {'pos': (1, 1, 1), 'info': {'label': 'Node 2', 'color': 'red'}}),
                  (3, {'pos': (2, 2, 2), 'info': {'label': 'Node 3', 'color': 'green'}}),
                  (4, {'pos': (3, 3, 3), 'info': {'label': 'Node 4', 'color': 'orange'}})])

# Add edges with attributes
G.add_edges_from([(1, 2, {'weight': 2}),
                  (2, 3, {'weight': 3}),
                  (3, 4, {'weight': 4}),
                  (4, 1, {'weight': 1})])

# Extract node positions
node_pos = nx.get_node_attributes(G, 'pos')

# Create node traces
node_trace = go.Scatter3d(
    x=[pos[0] for pos in node_pos.values()],
    y=[pos[1] for pos in node_pos.values()],
    z=[pos[2] for pos in node_pos.values()],
    mode='markers',
    marker=dict(
        size=8,
        line=dict(color='rgb(0,0,0)', width=1)
    ),
    hoverinfo='text',
    hovertext=[node[1]['info'] for node in G.nodes(data=True)]
)

# Create edge traces
edge_traces = []
for edge in G.edges(data=True):
    start = node_pos[edge[0]]
    end = node_pos[edge[1]]
    weight = edge[2]['weight']
    edge_trace = go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color='rgb(125,125,125)', width=2),
        hoverinfo='text',
        hovertext=f"Start: {edge[0]}, End: {edge[1]}, Weight: {weight}"
    )
    edge_traces.append(edge_trace)

# Create layout
layout = go.Layout(
    title='3D Graph Visualization',
    showlegend=False,
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    hovermode='closest'  # Enables the hover mode closest to the point
)

# Combine traces
fig = go.Figure(data=[node_trace] + edge_traces, layout=layout)

# Update the figure's layout to include a click event handler
fig.update_layout(
    clickmode='event+select'
)

# Plot 3d visualisation
fig.show()