# Model Deployment and Network Analysis

In this final module, you'll deploy your models and perform network analysis to understand semantic relationships and thematic patterns in your data. This stage transforms individual predictions into knowledge graphs.

## Overview

This module covers:

- Building semantic similarity networks
- Graph construction and visualization
- Community detection algorithms
- Centrality measures
- Narrative structure analysis

## What is Network Analysis?

Network analysis examines relationships and connections between entities:

- **Nodes**: Images, labels, captions, or concepts
- **Edges**: Semantic similarities or co-occurrences
- **Communities**: Clusters of related content
- **Structure**: Overall patterns and themes

## Graph Types

### Bipartite Graphs

Connect two different types of nodes (e.g., images and labels):

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create bipartite graph
G = nx.Graph()

# Add nodes with bipartite attribute
images = ['img1', 'img2', 'img3']
labels = ['protest', 'gathering', 'march']

G.add_nodes_from(images, bipartite=0)  # Images
G.add_nodes_from(labels, bipartite=1)  # Labels

# Add edges (image-label associations)
edges = [
    ('img1', 'protest'),
    ('img1', 'gathering'),
    ('img2', 'gathering'),
    ('img3', 'march')
]
G.add_edges_from(edges)

# Visualize
pos = nx.bipartite_layout(G, images)
nx.draw(G, pos, with_labels=True, node_color=['lightblue']*3 + ['lightgreen']*3)
plt.show()
```

### Multipartite Graphs

Connect multiple types of nodes (images, labels, captions):

```python
# Create multipartite graph
G = nx.Graph()

# Add different node types
images = ['img1', 'img2']
labels = ['protest', 'crowd']
captions = ['caption1', 'caption2']

G.add_nodes_from(images, node_type='image')
G.add_nodes_from(labels, node_type='label')
G.add_nodes_from(captions, node_type='caption')

# Add edges between different types
G.add_edges_from([
    ('img1', 'protest'),
    ('img1', 'caption1'),
    ('protest', 'caption1')
])
```

### Semantic Similarity Networks

Connect nodes based on semantic similarity:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
captions = [
    "A peaceful protest in the city square",
    "Demonstrators march through downtown",
    "Quiet evening in the plaza"
]
embeddings = model.encode(captions)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Build graph with similarity threshold
G = nx.Graph()
G.add_nodes_from(range(len(captions)))

threshold = 0.5
for i in range(len(captions)):
    for j in range(i+1, len(captions)):
        if similarity_matrix[i][j] > threshold:
            G.add_edge(i, j, weight=similarity_matrix[i][j])

# Visualize
pos = nx.spring_layout(G)
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw(G, pos, with_labels=True, edge_color=edge_weights,
        edge_cmap=plt.cm.Blues, width=3, node_color='lightblue', node_size=500)
plt.show()
```

## Building Networks

### Step 1: Prepare Data

Organize your generated labels and captions:

```python
import pandas as pd

# Load predictions
data = pd.DataFrame({
    'image_id': ['img1', 'img2', 'img3'],
    'labels': [['protest', 'crowd'], ['gathering'], ['march', 'crowd']],
    'caption': [
        'A large crowd protesting',
        'People gathering peacefully',
        'March through the streets'
    ]
})
```

### Step 2: Calculate Semantic Weights

Compute weights for different data types:

#### Visual Weights

Based on image similarity (using embeddings):

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

# Load pre-trained model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Remove classification layer to get embeddings
model = torch.nn.Sequential(*list(model.children())[:-1])

# Transform for model input
transform = weights.transforms()

def get_image_embedding(image_path):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        embedding = model(img_tensor)

    return embedding.squeeze().numpy()

# Calculate visual similarity
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
embeddings = [get_image_embedding(path) for path in image_paths]

from sklearn.metrics.pairwise import cosine_similarity
visual_similarity = cosine_similarity(embeddings)
```

#### Label Weights

Based on co-occurrence and semantic similarity:

```python
from sklearn.preprocessing import MultiLabelBinarizer

# Create label matrix
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(data['labels'])

# Calculate Jaccard similarity
from sklearn.metrics import jaccard_score

def jaccard_similarity_matrix(label_matrix):
    n = label_matrix.shape[0]
    similarity = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            similarity[i, j] = jaccard_score(label_matrix[i], label_matrix[j])
            similarity[j, i] = similarity[i, j]

    return similarity

label_similarity = jaccard_similarity_matrix(label_matrix)
```

#### Caption Weights

Based on semantic embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
caption_embeddings = model.encode(data['caption'].tolist())
caption_similarity = cosine_similarity(caption_embeddings)
```

### Step 3: Combine Weights

Create composite similarity scores:

```python
# Weighted combination of different similarities
alpha = 0.4  # Visual weight
beta = 0.3   # Label weight
gamma = 0.3  # Caption weight

combined_similarity = (
    alpha * visual_similarity +
    beta * label_similarity +
    gamma * caption_similarity
)

# Build final graph
G = nx.Graph()
n_images = len(data)

# Add nodes
for i, row in data.iterrows():
    G.add_node(i, image_id=row['image_id'], caption=row['caption'])

# Add edges with combined weights
threshold = 0.6
for i in range(n_images):
    for j in range(i+1, n_images):
        if combined_similarity[i, j] > threshold:
            G.add_edge(i, j, weight=combined_similarity[i, j])
```

## Community Detection

Identify clusters of related content:

### Leiden Algorithm

State-of-the-art community detection:

```python
import igraph as ig
import leidenalg

# Convert NetworkX graph to igraph
edges = list(G.edges())
g = ig.Graph(edges=edges)

# Run Leiden algorithm
partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

# Assign communities back to NetworkX graph
for i, community in enumerate(partition):
    for node in community:
        G.nodes[node]['community'] = i

print(f"Found {len(partition)} communities")
print(f"Modularity: {partition.modularity:.4f}")
```

### Louvain Algorithm

Alternative community detection:

```python
from networkx.algorithms import community

# Compute best partition
communities = community.greedy_modularity_communities(G)

# Assign to nodes
for i, comm in enumerate(communities):
    for node in comm:
        G.nodes[node]['community'] = i
```

### Visualize Communities

```python
import matplotlib.pyplot as plt

# Get community assignments
community_map = nx.get_node_attributes(G, 'community')
communities = list(set(community_map.values()))

# Create color map
colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
node_colors = [colors[community_map[node]] for node in G.nodes()]

# Draw graph
pos = nx.spring_layout(G, k=0.5, iterations=50)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, node_color=node_colors, with_labels=True,
        node_size=500, font_size=10, font_weight='bold')

# Add legend
for i, color in enumerate(colors):
    plt.scatter([], [], c=[color], label=f'Community {i}')
plt.legend()
plt.title('Network Communities')
plt.show()
```

## Centrality Measures

Identify important nodes in the network:

### Degree Centrality

Number of connections:

```python
degree_centrality = nx.degree_centrality(G)

# Find most connected nodes
sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
print("Top 5 nodes by degree centrality:")
for node, centrality in sorted_nodes[:5]:
    print(f"Node {node}: {centrality:.4f}")
```

### Betweenness Centrality

Nodes that bridge communities:

```python
betweenness_centrality = nx.betweenness_centrality(G)

sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
print("Top 5 nodes by betweenness centrality:")
for node, centrality in sorted_nodes[:5]:
    print(f"Node {node}: {centrality:.4f}")
```

### Eigenvector Centrality

Importance based on connections to important nodes:

```python
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

sorted_nodes = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
print("Top 5 nodes by eigenvector centrality:")
for node, centrality in sorted_nodes[:5]:
    print(f"Node {node}: {centrality:.4f}")
```

### Visualize Centrality

```python
# Visualize with node size based on centrality
pos = nx.spring_layout(G)
node_sizes = [eigenvector_centrality[node] * 3000 for node in G.nodes()]

plt.figure(figsize=(12, 8))
nx.draw(G, pos, node_size=node_sizes, node_color='lightblue',
        with_labels=True, font_size=10, font_weight='bold')
plt.title('Network with Node Sizes by Eigenvector Centrality')
plt.show()
```

## Advanced Analysis

### ERGM (Exponential Random Graph Models)

Model network formation processes:

```python
# Note: ERGM typically requires specialized software like statnet (R)
# Python implementation example using ergm package

# This is a conceptual example
# Install: pip install ergm

# Define ERGM terms
# ergm_model = ergm(network ~ edges + nodematch('community') + triangle)
```

### Narrative Structure Analysis

Identify story arcs and themes:

```python
# Temporal analysis if timestamps available
def analyze_temporal_patterns(G, timestamps):
    # Group by time periods
    time_groups = {}

    for node, ts in timestamps.items():
        period = ts // 3600  # Hour bins
        if period not in time_groups:
            time_groups[period] = []
        time_groups[period].append(node)

    # Analyze evolution
    for period in sorted(time_groups.keys()):
        subgraph = G.subgraph(time_groups[period])
        print(f"Period {period}: {subgraph.number_of_nodes()} nodes, "
              f"{subgraph.number_of_edges()} edges")

# Theme extraction
def extract_themes(G, captions):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(captions)

    # Get top terms
    feature_names = vectorizer.get_feature_names_out()
    print("Top themes:", feature_names)

    return feature_names
```

## Interactive Visualization

### Using Plotly

```python
import plotly.graph_objects as go

# Get positions
pos = nx.spring_layout(G)

# Extract node positions
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]

# Extract edge positions
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# Create figure
fig = go.Figure()

# Add edges
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                         line=dict(width=0.5, color='#888'),
                         hoverinfo='none'))

# Add nodes
fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers',
                         marker=dict(size=10, color='lightblue'),
                         text=list(G.nodes()),
                         hoverinfo='text'))

fig.update_layout(title='Interactive Network Graph',
                  showlegend=False,
                  hovermode='closest',
                  xaxis=dict(showgrid=False, zeroline=False),
                  yaxis=dict(showgrid=False, zeroline=False))
fig.show()
```

### Using Pyvis

```python
from pyvis.network import Network

# Create interactive network
net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')

# Add nodes
for node in G.nodes():
    net.add_node(node, label=str(node), title=f"Node {node}")

# Add edges
for edge in G.edges(data=True):
    net.add_edge(edge[0], edge[1], value=edge[2].get('weight', 1))

# Save and display
net.show('network.html')
```

## Best Practices

### Graph Construction
1. Choose appropriate similarity thresholds
2. Consider edge weight normalization
3. Handle disconnected components
4. Validate graph structure

### Community Detection
1. Try multiple algorithms
2. Compare results qualitatively
3. Validate with domain knowledge
4. Check community sizes

### Visualization
1. Use layouts appropriate for graph structure
2. Limit node labels for large graphs
3. Use interactive tools for exploration
4. Export high-quality figures

## Outputs

By the end of this module, you should have:

- [ ] Semantic similarity networks
- [ ] Community structure analysis
- [ ] Centrality measures for all nodes
- [ ] Network visualizations
- [ ] Thematic analysis report

## Interpretation

### Analyzing Results

Questions to ask:

1. **Community Structure**: What themes emerge from communities?
2. **Central Nodes**: Which images/concepts are most influential?
3. **Bridges**: Which nodes connect different communities?
4. **Patterns**: Are there temporal or spatial patterns?

### Reporting Findings

Document your analysis:

```markdown
# Network Analysis Report

## Overview
- Total nodes: X
- Total edges: Y
- Number of communities: Z

## Key Findings
1. Community 1: [Theme description]
2. Community 2: [Theme description]

## Central Concepts
- Most connected: [Node X]
- Key bridges: [Nodes Y, Z]

## Narrative Structure
[Description of overall patterns and story]
```

## Additional Resources

- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [igraph Documentation](https://igraph.org/python/)
- [Leiden Algorithm Paper](https://www.nature.com/articles/s41598-019-41695-z)
- [Network Science Book](http://networksciencebook.com/)

## Troubleshooting

### Graph Too Dense
- Increase similarity threshold
- Use sparse graph representations
- Sample edges

### Poor Community Detection
- Adjust resolution parameter
- Try different algorithms
- Check data quality

### Visualization Issues
- Reduce number of nodes shown
- Use hierarchical layouts
- Export to specialized tools (Gephi, Cytoscape)

---

Congratulations on completing the workshop! Check the [CyVerse Learning Center](https://learning.cyverse.org) for more resources.
