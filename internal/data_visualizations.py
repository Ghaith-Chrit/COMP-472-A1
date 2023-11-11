import math
import os
import graphviz
import imageio as iio
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def create_percentage_plots(data: pd.DataFrame, output_name: str) -> None:
    number_of_plots = data.shape[1]
    number_of_plots_per_row = math.ceil(number_of_plots / 2)
    number_of_axes = 2 * number_of_plots_per_row
    fig, axs = plt.subplots(2, number_of_plots_per_row, figsize=(number_of_plots_per_row * 20, number_of_plots_per_row * 3))
    plt.subplots_adjust(hspace=0.3)  
    axs = axs.flatten()

    for i, col in enumerate(data.columns):
        column_freq: pd.Series = data[col].value_counts(normalize=True).sort_index()
        unique, counts = column_freq.index, column_freq.values*100

        axs[i].set_title(f'Percentage of instances \n in {col} feature')
        axs[i].set_xlabel(col.title())
        axs[i].set_ylabel('Percentage of instances')

        axs[i].scatter(unique, counts)
        
        axs[i].set_ylim(bottom=0)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].yaxis.set_major_formatter(PercentFormatter())

        extent = axs[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{output_name}-temp-{i}.png', bbox_inches=extent.expanded(1.18, 1.215))

    if number_of_axes > data.shape[1]:
        fig.delaxes(axs[-1])
    
    plt.suptitle('Percentage of Instances in Each Feature', fontsize=16, fontweight='500')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(f"{output_name}.png", bbox_inches='tight', dpi=100)

    frames = np.stack([iio.imread(f"{output_name}-temp-{x}.png") for x in range(number_of_plots)], axis=0)
    iio.mimsave(f"{output_name}.gif", frames, format = 'GIF-PIL', duration=5)

    for i in range(number_of_plots):
        os.remove(f"{output_name}-temp-{i}.png")

def visualize_graph(dtc: tree.DecisionTreeClassifier, feature_names, class_names, output: str, max_depth: int | None = None) -> None:
	dot_data = tree.export_graphviz(dtc, 
									out_file=None, 
									feature_names=feature_names,  
									class_names=class_names,  
									filled=True, 
									rounded=True,  
									special_characters=True,
									max_depth=max_depth)  
	graph = graphviz.Source(dot_data) 
	graph.render(output) 
