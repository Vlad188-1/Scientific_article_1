# Plot graphics
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
from pathlib import Path


def plot_learning_curve(history: dict, path_to_save: Path, num_epochs: int):
    
    acc = history['train_accuracy']
    val_acc = history['val_accuracy']
    loss = history['train_loss']
    val_loss = history['val_loss']

    epochs = np.arange(1, num_epochs+1, 1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=['$\Large\\textbf{Accuracy}$', '$\Large\\textbf{Loss}$'], vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=epochs, y=acc, mode='lines+markers', showlegend=True, name='Training', marker_color='blue'),1,1)
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', showlegend=True, name='Validation', marker_color='red'),1,1)

    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines+markers', showlegend=False, marker_color='blue'),1,2)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', showlegend=False, marker_color='red'),1,2)

    fig.update_layout(plot_bgcolor='white', title=dict(text='<b>Training and validation neural network', x=0.5, font=dict(size=20)), 
                      legend=dict(font=dict(size=14)), height=550, width=1150, showlegend=True)
    fig.update_xaxes(row=1, col=1, nticks=5, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey', title=dict(text='Epochs', font=dict(size=23)))
    fig.update_xaxes(row=1, col=2, nticks=5, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey', title=dict(text='Epochs', font=dict(size=23)))

    fig.update_yaxes(row=1, col=1, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')
    fig.update_yaxes(row=1, col=2, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')
    fig.write_image(path_to_save)