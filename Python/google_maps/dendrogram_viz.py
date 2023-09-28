import numpy as np
import plotly.figure_factory as ff

# Sample data
data = np.array([
    [1, 2, 3, 4, 5],
    [1, 1, 2, 3, 4],
    [0, 1, 1, 2, 3],
    [0, 0, 1, 1, 2],
    [0, 0, 0, 1, 1]
])

labels = ['A', 'B', 'C', 'D', 'E']
fig = ff.create_dendrogram(data, orientation='left', labels=labels)

# Customize hover information
for trace in fig.data:
    trace.hoverinfo = 'text'
    trace.text = [f"Label: {label}" for label in labels]

fig.update_layout(width=800, height=500)
fig.show()