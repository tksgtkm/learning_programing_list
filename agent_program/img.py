import time
from collections import defaultdict
from inspect import getsource

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from matplotlib import lines

import ipywidgets as widgets

from IPython.display import display

def show_map(graph_data, node_colors=None):
    G = nx.Graph(graph_data['graph_dict'])
    node_colors = node_colors or graph_data['node_colors']
    node_positions = graph_data['node_positions']
    node_label_pos = graph_data['node_label_positions']
    edge_weights = graph_data['edge_weights']

    plt.figure(figsize=(18, 13))
    nx.draw(G, pos={k: node_positions[k] for k in G.nodes()}, node_color=[node_colors[node] for node in G.nodes()], linewidths=0.3, edgecolors='k')

    node_label_handles = nx.draw_networkx_labels(G, pos=node_label_pos, font_size=14)

    [label.set_bbox(dict(facecolor='white', edgecolor='none')) for label in node_label_handles.values()]

    nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_weights, font_size=14)

    white_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="white")
    orange_circle = lines.Line2D([], [], color="orange", marker='o', markersize=15, markerfacecolor="orange")
    red_circle = lines.Line2D([], [], color="red", marker='o', markersize=15, markerfacecolor="red")
    gray_circle = lines.Line2D([], [], color="gray", marker='o', markersize=15, markerfacecolor="gray")
    green_circle = lines.Line2D([], [], color="green", marker='o', markersize=15, markerfacecolor="green")
    plt.legend((white_circle, orange_circle, red_circle, gray_circle, green_circle),
               ('Un-explored', 'Frontier', 'Currently Exploring', 'Explored', 'Final Solution'),
               numpoints=1, prop={'size': 16}, loc=(.8, .75))
    
    plt.show()

def final_path_colors(initial_node_colors, problem, solution):
    final_colors = dict(initial_node_colors)
    final_colors[problem.initial] = "green"
    for node in solution:
        final_colors[node] = "green"
    return final_colors

def display_visual(graph_data, user_input, algorithm=None, problem=None):
    initial_node_colors = graph_data['node_colors']
    if user_input is False:
        def slider_callback(iteration):
            try:
                show_map(graph_data, node_colors=all_node_colors[iteration])
            except:
                pass

        def visualize_callback(visualize):
            if visualize is True:
                button.value = False

                global all_node_colors

                iterations, all_node_colors, node = algorithm(problem)
                solution = node.solution()
                all_node_colors.append(final_path_colors(all_node_colors[0], problem, solution))

                slider.max = len(all_node_colors) - 1

                for i in range(slider.max + 1):
                    slider.value = i

        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration=slider)
        display(slider_visual)

        button = widgets.ToggleButton(value=False)
        button_visual = widgets.interactive(visualize_callback, visualize=button)
        display(button_visual)

    if user_input is True:
        node_colors = dict(initial_node_colors)
        if isinstance(algorithm, dict):
            assert set(algorithm.keys()).issubset(
                {
                    "Breadth First Tree Search",
                    "Depth First Tree Search",
                    "Breadth First Search",
                    "Depth First Graph Search",
                    "Best First Graph Search",
                    "Uniform Cost Search",
                    "Depth Limited Search",
                    "Iterative Deepening Search",
                    "Greedy Best First Search",
                    "A-star Search",
                    "Recursive Best First Search"
                }
            )
            algo_dropdown = widgets.Dropdown(
                description="Search Algorithm",
                options=sorted(list(algorithm.keys())),
                value="Breadth First Tree Search"
            )

            display(algo_dropdown)
        elif algorithm is None:
            print("No Algorithm to run")
            return 0
        
        def slider_callback(iteration):
            try:
                show_map(graph_data, node_colors=all_node_colors[iteration])
            except:
                pass

        def visualize_callback(visualize):
            if visualize is True:
                button.value = False

                global all_node_colors

                iterations, all_node_colors, node = algorithm(problem)
                solution = node.solution()
                all_node_colors.append(final_path_colors(all_node_colors[0], problem, solution))

                slider.max = len(all_node_colors) - 1

                for i in range(slider.max + 1):
                    slider.value = i

        start_dropdown = widgets.Dropdown(description="Start city: ",
                                          options=sorted(list(node_colors.keys())), value="Arad")
        display(start_dropdown)

        end_dropdown = widgets.Dropdown(description="Goal city: ",
                                        options=sorted(list(node_colors.keys())), value="Fagaras")
        display(end_dropdown)

        button = widgets.ToggleButton(value=False)
        button_visual = widgets.interactive(visualize_callback, visualize=button)
        display(button_visual)

        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration=slider)
        display(slider_visual)

def plot_NQueens(solution):
    n = len(solution)
    board = np.array([2 * int((i + j) % 2) for j in range(n) for i in range(n)]).reshape((n, n))
    im = Image.open('images/queen_s.png')
    height = im.size[1]
    im = np.array(im).astype(np.float64) / 255
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title('{} Queens'.format(n))
    plt.imshow(board, cmap='binary', interpolation='nearest')
    if isinstance(solution, dict):
        for (k, v) in solution.items():
            newax = fig.add_axes([0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1)
            newax.imshow(im)
            newax.axis('off')
    elif isinstance(solution, list):
        for (k, v) in enumerate(solution):
            newax = fig.add_axes([0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1)
            newax.imshow(im)
            newax.axis('off')
    fig.tight_layout()
    plt.show()
 
def heatmap(grid, cmap='binary', interpolation='nearest'):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title('Heatmap')
    plt.imshow(grid, cmap=cmap, interpolation=interpolation)
    fig.tight_layout()
    plt.show()