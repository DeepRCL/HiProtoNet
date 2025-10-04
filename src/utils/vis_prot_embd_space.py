import pickle
import torch
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import pandas
import glob
import io
import base64
from PIL import Image
import wandb

import src.utils.embedding_visualizer.hyptorch.pmath as pmath
from src.utils.embedding_visualizer.learning.pca import HoroPCA
from src.utils import lorentz as L
from src.utils.embedding_visualizer.htsne_impl import TSNE as hTSNE  # Assuming this is a local implementation


# Make the working directory the root of the project
sys.path.append(os.path.dirname(os.getcwd()))


def load_pickle(pickle_path, log=print):
    """Loads data from a pickle file."""
    with open(pickle_path, "rb") as handle:
        pickle_data = pickle.load(handle)
    return pickle_data


def get_prot_info_path(model_root_dir, prot_type, epoch_num):
    """Constructs the path to the prototype information pickle file."""
    return f"{model_root_dir}/img/{prot_type}/epoch-{epoch_num}_pushed/prototypes_info.pickle"


def get_hyperbolic_prototypes(prototype_vec, visual_alpha, curv, lift_prototypes=False, device="cpu"):
    """
        Transforms prototypes to hyperbolic space.
        visual_alpha and curv are needed logged! no need to exp them here!
    """
    N, D = prototype_vec.shape[:2]
    prototype_vec = prototype_vec.reshape(N, D)  # shape (num_classes, D)
    if lift_prototypes:
        prototype_vec = L.get_hyperbolic_feats(prototype_vec, visual_alpha, curv, device)  # shape (num_classes, D)
    return prototype_vec


def get_time_component(x_space, curv):
    """Calculates the time component for Lorentz embedding."""
    x_time = torch.sqrt(1 / curv + torch.sum(x_space**2, dim=-1))
    return x_time


def get_space_time(x, visual_alpha, curv, lift_prototypes):
    """Transforms spatial coordinates to space-time coordinates in Lorentz model."""
    x_space = get_hyperbolic_prototypes(x, visual_alpha, curv, lift_prototypes)
    curv = curv.exp()
    x_time = get_time_component(x_space, curv)
    x_full = torch.cat([x_time.unsqueeze(-1), x_space], dim=-1)
    lorentz_constraint = -x_time**2 + torch.sum(x_space**2, axis=1)

    atols = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 2]
    for atol in atols:
        try:
            assert torch.allclose(lorentz_constraint, -1 / curv * torch.ones_like(lorentz_constraint), atol=atol)
            return x_full, atol
        except AssertionError:
            print(f"Lorentz constraint not satisfied within tolerance level {atol}.")
            continue
    raise ValueError("Lorentz constraint not satisfied within acceptable tolerance levels.")


def lorentz_to_poincare_batch(lorentz_coords, curvature):
    """
    Converts a batch of points in the Lorentz model of hyperbolic space to the Poincaré model.

    Args:
        lorentz_coords (torch.Tensor): Tensor of shape (batch_size, N+1), where each row is a point
                                       in the Lorentz model. The first column is the time-like coordinate.
        curvature (float): Curvature of the hyperboloid (negative value for hyperbolic space).

    Returns:
        torch.Tensor: Tensor of shape (batch_size, N), where each row is the corresponding point
                      in the Poincaré model.
    """
    if curvature >= 0:
        raise ValueError("Curvature must be negative for hyperbolic space.")

    # Extract the time-like coordinate (x_0) and spatial coordinates (x_1, ..., x_N)
    x_0 = lorentz_coords[:, 0]  # Shape: (batch_size,)
    x_spatial = lorentz_coords[:, 1:]  # Shape: (batch_size, N)

    # Compute the squared norm of spatial coordinates for each point
    spatial_norm_squared = torch.sum(x_spatial ** 2, dim=1)  # Shape: (batch_size,)

    # Compute the denominator (normalization term)
    denom = x_0 + torch.sqrt(x_0 ** 2 - spatial_norm_squared)

    # Add a small epsilon to avoid division by zero
    eps = 1e-9
    denom = torch.clamp(denom, min=eps)

    # Convert to Poincaré coordinates
    poincare_coords = x_spatial / denom.unsqueeze(1)  # Shape: (batch_size, N)

    return poincare_coords


def horopca_transform(x, n_components=2, lr=0.05, max_steps=500, horopca=None):
    """
    Apply HoroPCA transformation to input data

    Args:
        x: Input tensor of shape (batch_size, dim)
        n_components: Number of components for dimensionality reduction
        lr: Learning rate for optimization
        max_steps: Maximum optimization steps

    Returns:
        embeddings: Transformed embeddings in the ball
        metrics: Computed metrics from HoroPCA
        horopca: HoroPCA model
    """
    train_model = horopca is None
    horopca = horopca or HoroPCA(dim=x.shape[1], n_components=n_components, lr=lr, max_steps=max_steps)

    if torch.cuda.is_available():
        horopca.cuda()
        x = x.cuda()

    if train_model:
        print("horopca model fitting...")
        start_time = time.time()
        horopca.fit(x, iterative=False, optim=True)
        print(f"horopca model fitting took {time.time() - start_time:.2f} seconds")
    else:
        print("horopca model is trained already.")

    metrics = horopca.compute_metrics(x)
    embeddings = horopca.map_to_ball(x).detach().cpu()

    return embeddings, metrics, horopca


def get_horopca_model_path(model_root_dir, model_name, n_components, epoch_num):
    """Constructs the path to the saved HoroPCA model."""
    return f"{model_root_dir}/img/analysis/{model_name}-{n_components}_D/epoch_{epoch_num:02d}.pth"


def save_horopca_model(horopca_model, model_root_dir, model_name, n_components, epoch_num):
    """Saves the HoroPCA model to a file."""
    horopca_model_path = get_horopca_model_path(model_root_dir, model_name, n_components, epoch_num)
    os.makedirs(os.path.dirname(horopca_model_path), exist_ok=True)
    torch.save(horopca_model, horopca_model_path)
    print(f"Horopca model saved at {horopca_model_path}")
    return horopca_model_path


def load_horopca_model(model_root_dir, model_name, n_components, epoch_num):
    """Loads a HoroPCA model from a file."""
    horopca_model_path = get_horopca_model_path(model_root_dir, model_name, n_components, epoch_num)
    horopca_model = torch.load(horopca_model_path)
    print(f"Horopca model loaded from {horopca_model_path}")
    return horopca_model


def run_cosne(embeddings, learning_rate=1.0, learning_rate_for_h_loss=0.0, perplexity=5, early_exaggeration=1, student_t_gamma=1.0):
    """Runs the CoSNE algorithm."""
    print(f"Running with perplexity={perplexity}, learning_rate={learning_rate}, early_exaggeration={early_exaggeration}, student_t_gamma={student_t_gamma}, learning_rate_for_h_loss={learning_rate_for_h_loss}")
    print("Running CoSNE")
    co_sne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True,
                    metric='precomputed', learning_rate_for_h_loss=learning_rate_for_h_loss, student_t_gamma=student_t_gamma, learning_rate=learning_rate, n_iter=1000, perplexity=perplexity, early_exaggeration=early_exaggeration)

    dists = pmath.dist_matrix(embeddings, embeddings, c=1).numpy()

    print("CoSNE model fitting...")
    start_time = time.time()
    CO_SNE_embedding = co_sne.fit_transform(dists, embeddings)
    print(f"CoSNE model fitting took {time.time() - start_time:.2f} seconds")

    return CO_SNE_embedding


def dict_info(data):
    """Prints information about the loaded data."""
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            print(key, val.shape)
        else:
            print(key, val)


def get_fig_path(model_root_dir, fig_name, epoch_num, format='png'):
    file_path = f"{model_root_dir}/img/analysis/{fig_name}/{epoch_num:02d}.{format}"
    #make directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path


def plot_embeddings(ax, points, colors, title, radius=1.05, s=5, alpha=0.6, legend=None):
    """Create a single plot with circle boundary"""
    circle = patches.Circle((0, 0), radius, color='black', fill=False, linestyle='--', alpha=0.3)
    # circlec enter a start gray color
    ax.scatter(0, 0, c='black', marker='*', s=50, alpha=0.3)
    ax.add_patch(circle)
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=s, alpha=alpha)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)

    if legend is not None:
        for key, val in legend['items'].items():
            ax.scatter([], [], c=val, s=20, label=key)
        ax.legend(fontsize=legend['size'])

    # set tik fontsize to 8
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)


# define a function to get color and also indices for the selected classes
def get_color_indices_selected_classes(num_classes, num_locals, num_broads, selected_num_classes):
    num_local_prototypes_per_class = num_locals // num_classes
    num_broad_prototypes_per_class = num_broads // num_classes
    # create the color list, a different color for each category and each class
    # create list of colors, same shape as number of prototypes, blue for local, red for broad
    local_colors = ['blue', 'green', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'brown', 'gray'][:selected_num_classes]
    broad_colors = ['red', 'lime', 'orange', 'teal', 'navy', 'maroon', 'olive', 'coral', 'gold', 'silver'][:selected_num_classes]
    # repeat each color for the number of prototypes per class
    local_colors = [color for color in local_colors for _ in range(num_local_prototypes_per_class)]
    broad_colors = [color for color in broad_colors for _ in range(num_broad_prototypes_per_class)]
    # concatenate the colors
    colors_selected_classes = local_colors + broad_colors

    # find the indices, so we use them to plot the embeddings_2d, and CO_SNE_embeddings
    local_indx_end = num_local_prototypes_per_class*selected_num_classes
    local_prototypes_indices = list(range(0, local_indx_end))
    broad_start = num_local_prototypes_per_class*num_classes
    broad_end = broad_start + num_broad_prototypes_per_class*selected_num_classes
    part_prototypes_indices = list(range(broad_start, broad_end))
    indices_selected_classes = local_prototypes_indices + part_prototypes_indices

    # Create intra class index so each prototype within a class has a unique index
    # intra_class_index = [j for _ in range(selected_num_classes) for j in range(num_local_prototypes_per_class)] + \
    #                 [j for _ in range(selected_num_classes) for j in range(num_broad_prototypes_per_class)]
    # put the indices in sequence, like how they prototypes are visualized!!!
    intra_class_index = [j for j in range(num_local_prototypes_per_class*selected_num_classes)] + \
                    [j for j in range(num_broad_prototypes_per_class*selected_num_classes)]

    return colors_selected_classes, indices_selected_classes, intra_class_index


def image_to_base64(image_path):
    img = Image.open(image_path)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_interactive_scatter(embeddings_data, indices, intra_class_index, colors_selected_classes, title="",
                               image_paths=None):
    """
    Creates an interactive scatter plot from embeddings data

    Args:
        embeddings_data: Tensor containing the embeddings coordinates
        indices: List of indices to select from embeddings
        colors_selected_classes: List of color labels for each point
        title: Title for the plot

    Returns:
        Plotly figure object
    """
    # Create a dataframe with the embeddings coordinates
    df = pandas.DataFrame(embeddings_data[indices], columns=['x', 'y'])

    # # Create a readable color mapping
    # color_map = {
    #     'Class 1 local': 'blue',
    #     'Class 1 broad': 'red',
    #     'Class 2 local': 'green',
    #     'Class 2 broad': 'orange'
    # }
    # Create a readable color mapping based on colors_selected_classes
    unique_colors = []
    color_map = {}
    for color in colors_selected_classes:
        if color not in unique_colors:
            unique_colors.append(color)
    half_len = len(unique_colors) // 2
    for i, color in enumerate(unique_colors):
        class_type = 'local' if i < half_len else 'broad'
        color_map[f'C{(i % half_len) + 1}-{class_type}'] = color

    df['colors'] = colors_selected_classes
    # Map colors to descriptive labels
    df['type'] = df['colors'].map({v: k for k, v in color_map.items()})
    df['intra_class_index'] = intra_class_index
    df['index'] = list(range(len(indices)))

    # Add image data if provided
    if image_paths is not None:
        df['image_base64'] = [image_to_base64(path) for path in image_paths]
        custom_data = ['image_base64', 'intra_class_index']
    else:
        custom_data = ['intra_class_index']

    # Add a new column for marker symbols
    # For example, let's say we want stars for local classes and circles for broad classes
    df['marker_symbol'] = df['type'].apply(lambda x: 'star' if 'local' in x else 'circle')

    # Create the scatter plot with updated title formatting and color mapping
    fig = px.scatter(
        df,
        x='x',
        y='y',
        custom_data=custom_data,
        color='type',
        symbol='marker_symbol',  # Use the new column for marker symbols
        hover_data=['intra_class_index'],
        title=title.replace('\n', '<br>'),
        color_discrete_map=color_map
    )

    if image_paths is not None:
        fig.update_traces(
            hovertemplate='<img src="%{customdata[0]}" width="200"><br>x: %{x}<br>y: %{y}<br>Prototype Index: %{customdata[1]}'
        )
    else:
        fig.update_traces(
            hovertemplate='x: %{x}<br>y: %{y}<br>Prototype Index: %{customdata[0]}'
        )

    # Update layout
    fig.update_layout(
        legend_title_text='Class',
        autosize=True,
        width=600,
        height=500,
        showlegend=True,
        title_x=0.5
    )

    # Explicitly set the symbols for each trace
    for trace in fig.data:
        if 'local' in trace.name:
            trace.marker.symbol = 'star'
        else:
            trace.marker.symbol = 'circle'

    # Update marker size
    fig.update_traces(marker=dict(size=10))

    return fig


def get_prot_image_paths(dirname):
    # return a list of all the images in the directory, sorted based on the index in the filename
    image_paths = glob.glob(os.path.join(dirname, "*.png"))
    image_paths.sort(key=lambda x:  int(os.path.basename(x).split('_')[0]))
    return image_paths

def visualize_prototype_embedding_space(model_root_dir, epoch_num, emb_class_names, log_wandb="disabled"):
    """
    Visualizes the prototype embedding space using HoroPCA and CoSNE.

    Args:
        model_root_dir (str): The root directory where the model is stored.
        epoch_num (int): The epoch number to visualize.
    """
    model_runname = os.path.basename(model_root_dir)

    ###################################################################
    # load the prototypes
    ###################################################################
    # Construct paths to prototype information
    prot_local_path = get_prot_info_path(model_root_dir, "local", epoch_num)
    prot_broad_path = get_prot_info_path(model_root_dir, "part", epoch_num)

    # Load prototype information
    prot_local_info = load_pickle(prot_localpath)
    prot_broad_info = load_pickle(prot_broad_path)

    # Print prototype information
    print("prototypes of | model:{} | epoch:{}".format(model_root_dir, epoch_num))
    print("#" * 50)
    dict_info(prot_localinfo)
    print("#" * 50)
    dict_info(prot_broad_info)
    print("#" * 50)

    # Extract relevant information
    curv = torch.tensor(prot_local_info["curv"])  # it is stored as log! need to convert it to exp when needed
    print("curv after exp", curv.exp())
    visual_alpha = torch.tensor(prot_local_info["visual_alpha"])  # it is stored as log! need to convert it to exp when needed
    print("visual_alpha after exp", visual_alpha.exp())
    lift_pros = prot_local_info["lift_prototypes"]
    num_classes = prot_local_info["prototypes_preds"].shape[1]

    # Extract pre-push and post-push embeddings
    prot_name = "prototypes_features_to_be_replaced" if "prototypes_features_to_be_replaced" in prot_local_info else "prototypes_pre_push"
    local_prototypes = torch.from_numpy(prot_local_info[prot_name])
    part_prototypes = torch.from_numpy(prot_broad_info[prot_name])

    prot_name = "prototypes_src_tracked_features" if "prototypes_src_tracked_features" in prot_local_info else "prototypes_post_push"
    local_prototypes_pushed = torch.from_numpy(prot_local_info[prot_name])
    part_prototypes_pushed = torch.from_numpy(prot_broad_info[prot_name])

    ###################################################################
    #  turn them to N+1: Convert prototypes to space-time representation
    ###################################################################
    part_prototypes_full, atol_part_prototypes = get_space_time(part_prototypes, visual_alpha, curv, lift_pros)
    local_prototypes_full, atol_local_prototypes = get_space_time(local_prototypes, visual_alpha, curv, lift_pros)
    print(part_prototypes_full.shape)
    print(local_prototypes_full.shape)
    # concatenate the local and broad prototypes
    prototypes_full = torch.cat([local_prototypes_full, part_prototypes_full], dim=0)
    print(prototypes_full.shape)

    part_prototypes_pushed_full, atol_part_prototypes_pushed = get_space_time(part_prototypes_pushed, visual_alpha, curv, lift_pros)
    local_prototypes_pushed_full, atol_local_prototypes_pushed = get_space_time(local_prototypes_pushed, visual_alpha, curv, lift_pros)
    print(part_prototypes_pushed_full.shape)
    print(local_prototypes_pushed_full.shape)
    # concatenate the local and broad prototypes
    prototypes_pushed_full = torch.cat([local_prototypes_pushed_full, part_prototypes_pushed_full], dim=0)
    print(prototypes_pushed_full.shape)

    ###################################################################
    # Convert to Poincare embeddings
    ###################################################################
    _curv = -curv.exp()
    prototypes_poincare = lorentz_to_poincare_batch(prototypes_full, _curv)
    prototypes_poincare_pushed = lorentz_to_poincare_batch(prototypes_pushed_full, _curv)

    ###################################################################
    # then we use HoroPCA to 2 and deeper dimensions
    ###################################################################
    # HoroPCA to 2 dimensions (Pre-Push)
    n_components = 2
    max_steps = 500
    lr = 0.05

    try:
        horopca_model_2D = load_horopca_model(model_root_dir, 'horopca', n_components, epoch_num)
        embeddings_2d, metrics, _ = horopca_transform(prototypes_poincare, n_components, lr=lr, max_steps=max_steps, horopca=horopca_model_2D)
    except Exception as e:
        embeddings_2d, metrics, horopca_model_2D = horopca_transform(prototypes_poincare, n_components, lr=lr, max_steps=max_steps)
        save_horopca_model(horopca_model_2D, model_root_dir, 'horopca', n_components, epoch_num)

    # HoroPCA to 2 dimensions (Post-Push)
    try:
        horopca_model_2D_pushed = load_horopca_model(model_root_dir, 'horopca_pushed', n_components, epoch_num)
        embeddings_pushed_2d, metrics_pushed, _ = horopca_transform(prototypes_poincare_pushed, n_components, lr=lr, max_steps=max_steps, horopca=horopca_model_2D_pushed)
    except Exception as e:
        embeddings_pushed_2d, metrics_pushed, horopca_model_2D_pushed = horopca_transform(prototypes_poincare_pushed, n_components, lr=lr, max_steps=max_steps)
        save_horopca_model(horopca_model_2D_pushed, model_root_dir, 'horopca_pushed', n_components, epoch_num)

    # HoroPCA to 20 dimensions (Pre-Push)
    n_components = 20

    try:
        horopca_model_large = load_horopca_model(model_root_dir, 'horopca', n_components, epoch_num)
        embeddings, _, _ = horopca_transform(prototypes_poincare, n_components, lr=lr, max_steps=max_steps, horopca=horopca_model_large)
    except Exception as e:
        embeddings, _, horopca_model_large = horopca_transform(prototypes_poincare, n_components, lr=lr, max_steps=max_steps)
        save_horopca_model(horopca_model_large, model_root_dir, 'horopca', n_components, epoch_num)

    # HoroPCA to 20 dimensions (Post-Push)
    try:
        horopca_model_large_pushed = load_horopca_model(model_root_dir, 'horopca_pushed', n_components, epoch_num)
        embeddings_pushed, _, _ = horopca_transform(prototypes_poincare_pushed, n_components, lr=lr, max_steps=max_steps, horopca=horopca_model_large_pushed)
    except Exception as e:
        embeddings_pushed, _, horopca_model_large_pushed = horopca_transform(prototypes_poincare_pushed, n_components, lr=lr, max_steps=max_steps)
        save_horopca_model(horopca_model_large_pushed, model_root_dir, 'horopca_pushed', n_components, epoch_num)

    # Transform pre-push embeddings with post-push-fitted model
    embeddings_2d_post_fitted_model, _, _ = horopca_transform(prototypes_poincare, horopca=horopca_model_2D_pushed)
    embeddings_post_fitted_model, _, _ = horopca_transform(prototypes_poincare, horopca=horopca_model_large_pushed)

    ###################################################################
    # then we use the deeper embeddings for Co_SNE
    ###################################################################

    # CO-SNE hyperparameters
    learning_rate = 5.0
    learning_rate_for_h_loss = 0.1
    perplexity = 20
    early_exaggeration = 1.0
    student_t_gamma = 0.1
    # # CHAT GPT
    # learning_rate = 3.0  # Reduce to slow down updates
    # learning_rate_for_h_loss = 0.5  # Increase to handle curvature better
    # perplexity = 30  # Increase to emphasize local relationships
    # early_exaggeration = 2.0  # Increase to cluster better initially
    # student_t_gamma = 0.5  # Allow wider distribution of points

    CO_SNE_embeddings = run_cosne(embeddings, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)
    CO_SNE_embeddings_pushed = run_cosne(embeddings_pushed, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)
    CO_SNE_embeddings_post_fitted_model_large = run_cosne(embeddings_post_fitted_model, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)

    ###################################################################
    # For CO_SNE Compute the centroid of all output 2-dim embeddings and recentered embeddings according to this point.
    ###################################################################
    # # computed the centroid of 2-dim CO_SNE_embeddings and recenter embeddings according to this point.
    # centroid = np.mean(CO_SNE_embeddings, axis=0)
    # CO_SNE_embeddings = CO_SNE_embeddings - centroid
    #
    # centroid = np.mean(CO_SNE_embeddings_pushed, axis=0)
    # CO_SNE_embeddings_pushed = CO_SNE_embeddings_pushed - centroid
    #
    # centroid = np.mean(CO_SNE_embeddings_post_fitted_model_large, axis=0)
    # CO_SNE_embeddings_post_fitted_model_large = CO_SNE_embeddings_post_fitted_model_large - centroid


    ###################################################################
    # then we plot both 2d embeddings, for all prototypes
    ###################################################################
    radius = 1.05

    # create list of colors, same shape as number of prototypes, blue for local, red for broad
    local_colors = ['blue'] * local_prototypes_full.shape[0]
    broad_colors = ['red'] * part_prototypes_full.shape[0]
    # concatenate the colors
    colors = local_colors + broad_colors

    legend = {'items': {'local Prototypes': 'blue', 'broad Prototypes': 'red'}, 'size': 8}

    fig_prototypes_all, axs = plt.subplots(2, 3, figsize=(12, 9))
    plot_embeddings(axs[0,0], embeddings_2d, colors, f"HoroPCA\nPre-Push prototypes", radius=radius, s=5, alpha=0.6, legend=legend)
    plot_embeddings(axs[0,1], embeddings_pushed_2d, colors, f"HoroPCA\nPost-Push prototypes", radius=radius, s=5, alpha=0.6, legend=legend)
    plot_embeddings(axs[1,0], CO_SNE_embeddings, colors, f"CO-SNE\nPre-Push prototypes", radius=radius, s=5, alpha=0.6, legend=legend)
    plot_embeddings(axs[1,1], CO_SNE_embeddings_pushed, colors, f"CO-SNE\nPost-Push prototypes", radius=radius, s=5, alpha=0.6, legend=legend)
    ###########################################################################################################
    ####### plot based on embeddings that are reduced in dimensionality with post-push fitted horopca #########
    plot_embeddings(axs[0,2], embeddings_2d_post_fitted_model, colors, f"HoroPCA (fitted on post-push)\nPre-Push prototypes", radius=radius, s=5, alpha=0.6, legend=legend)
    plot_embeddings(axs[1,2], CO_SNE_embeddings_post_fitted_model_large, colors, f"CO-SNE (fitted on post-push)\nPre-Push prototypes", radius=radius, s=5, alpha=0.6, legend=legend)

    plt.suptitle(
        f"Embedding visualization of all prototypes | epoch: {epoch_num}\n"
        f"model: {model_runname}\n"
        f"atols=[{atol_part_prototypes}, {atol_local_prototypes}, {atol_part_prototypes_pushed}, {atol_local_prototypes_pushed}]",
        fontsize=16
    )
    plt.tight_layout()

    # save figure in the same directory as the horopca model
    plt.savefig(get_fig_path(model_root_dir, "0_all_prototypes", epoch_num), bbox_inches='tight', dpi=fig_prototypes_all.dpi)

    ###################################################################
    # then we plot both 2d embeddings, for the first two classes
    ###################################################################
    # choose the first selected_num_classes only, and plot the part and local prototypes for those classes only
    # considering the embeddigns are concatenated of [local, part], grab the prototypes of the first 2 classes from each group. so, [0:num_local_prototypes_per_class] are the local prototypes of the first class, and [num_local_prototypes_per_class:2*num_local_prototypes_per_class] are the local prototypes of the second class
    # and [num_local_prototypes_per_class*num_classes:num_local_prototypes_per_class*num_classes+num_broad_prototypes_per_class] are the broad prototypes of the first class, and [num_local_prototypes_per_class*num_classes+num_broad_prototypes_per_class:num_local_prototypes_per_class*num_classes+2*num_broad_prototypes_per_class] are the broad prototypes of the second class
    selected_num_classes = len(emb_class_names)
    colors_selected_classes, indices, intra_class_index = get_color_indices_selected_classes(num_classes, local_prototypes_full.shape[0], part_prototypes_full.shape[0], selected_num_classes)
    ################################################################################
    # plot the embeddings_2d
    radius = 0.5
    fig_shortlist_classes, axs = plt.subplots(2, 3, figsize=(12, 9))

    # Create a readable color mapping based on colors_selected_classes
    unique_colors = []
    color_map = {}
    for color in colors_selected_classes:
        if color not in unique_colors:
            unique_colors.append(color)
    half_len = len(unique_colors) // 2
    for i, color in enumerate(unique_colors):
        class_type = 'local' if i < half_len else 'broad'
        class_name = emb_class_names[i % half_len]
        color_map[f'C{(i % half_len) + 1}-{class_type}-{class_name}'] = color

    # add a legend for the colors, based on color_map
    legend = {'items': color_map, 'size': 8}

    plot_embeddings(axs[0,0], embeddings_2d[indices], colors_selected_classes, f"HoroPCA\nPre-Push prototypes", radius=radius, s=15, alpha=0.6, legend=legend)
    plot_embeddings(axs[0,1], embeddings_pushed_2d[indices], colors_selected_classes, f"HoroPCA\nPost-Push prototypes", radius=radius, s=15, alpha=0.6, legend=legend)
    plot_embeddings(axs[1,0], CO_SNE_embeddings[indices], colors_selected_classes, f"CO-SNE\nPre-Push prototypes", radius=radius, s=15, alpha=0.6, legend=legend)
    plot_embeddings(axs[1,1], CO_SNE_embeddings_pushed[indices], colors_selected_classes, f"CO-SNE\nPost-Push prototypes", radius=radius, s=15, alpha=0.6, legend=legend)
    ###########################################################################################################
    ####### plot based on embeddings that are reduced in dimensionality with post-push fitted horopca #########
    plot_embeddings(axs[0,2], embeddings_2d_post_fitted_model[indices], colors_selected_classes, f"HoroPCA (fitted on post-push)\nPre-Push prototypes", radius=radius, s=15, alpha=0.6, legend=legend)
    plot_embeddings(axs[1,2], CO_SNE_embeddings_post_fitted_model_large[indices], colors_selected_classes, f"CO-SNE (fitted on post-push)\nPre-Push prototypes", radius=radius, s=15, alpha=0.6, legend=legend)

    plt.suptitle(
        f"Prototypes of shortlist classes | epoch: {epoch_num}\n"
        f"model: {model_runname}\n"
        f"atols=[{atol_part_prototypes}, {atol_local_prototypes}, {atol_part_prototypes_pushed}, {atol_local_prototypes_pushed}]",
        fontsize=16
    )
    plt.tight_layout()


    # save figure
    plt.savefig(get_fig_path(model_root_dir, "1_shortlist_classes", epoch_num), bbox_inches='tight', dpi=fig_shortlist_classes.dpi)

    ###################################################################
    # then we plot the movement of prototypes during the push
    ###################################################################
    radius = 0.5
    # in a single plot, add scatters of embeddings_pushed_2d and embeddings_2d_post_fitted_model with selected indices, and add a line between each corresponding point, with arrow to show the direction of the movement

    # create a figure and axis
    fig_movement, ax = plt.subplots(figsize=(10, 10))

    circle = plt.Circle((0, 0), radius, color='black', fill=False, linestyle='--', alpha=0.3)
    ax.add_patch(circle)

    # add a star to show the origin
    ax.scatter(0, 0, c='gray', marker='*', s=100)

    # plot the embeddings_pushed_2d
    ax.scatter(embeddings_pushed_2d[indices, 0], embeddings_pushed_2d[indices, 1], c=colors_selected_classes, s=15,
               alpha=1)
    # plot the embeddings_2d_post_fitted_model
    ax.scatter(embeddings_2d_post_fitted_model[indices, 0], embeddings_2d_post_fitted_model[indices, 1],
               c=colors_selected_classes, s=15, alpha=1)

    # add lines between each corresponding point, with arrow pointing from embeddings_2d_post_fitted_model to embeddings_pushed_2d
    for i in range(len(indices)):
        ax.annotate("",
                    xy=(embeddings_pushed_2d[indices[i], 0], embeddings_pushed_2d[indices[i], 1]),
                    xytext=(
                    embeddings_2d_post_fitted_model[indices[i], 0], embeddings_2d_post_fitted_model[indices[i], 1]),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color='black'))

    # add legend for colors_selected_classes
    legend = {'items': color_map, 'size': 8}

    for label, color in legend['items'].items():
        ax.scatter([], [], c=color, label=label)
    ax.legend(fontsize=legend['size'])

    # make sure the aspect ratio is equal
    ax.set_aspect('equal')
    # add title
    ax.set_title(
        f"Movement of pushing prototypes | Epoch:{epoch_num} \n"
        f"model: {model_runname} \n"
        f"atols=[{atol_part_prototypes}, {atol_local_prototypes}, {atol_part_prototypes_pushed}, {atol_local_prototypes_pushed}]",
        fontsize=15
    )
    plt.tight_layout()



    # save figure
    plt.savefig(get_fig_path(model_root_dir, "2_push_movement", epoch_num), bbox_inches='tight', dpi=fig_movement.dpi)
    print(prot_local_path)
    plt.show()

    ###################################################################
    # Interactive scatter plot so we can back track it to the actual prototype and see the result!
    ####################################################################
    # Create interactive plots with titles
    fig1 = create_interactive_scatter(embeddings_2d, indices, intra_class_index, colors_selected_classes,
                                      f'HoroPCA <br> Pre-Push prototypes | Epoch:{epoch_num}')
    fig2 = create_interactive_scatter(embeddings_pushed_2d, indices, intra_class_index, colors_selected_classes,
                                      f'HoroPCA <br> Post-Push prototypes | Epoch:{epoch_num}')
    fig3 = create_interactive_scatter(embeddings_2d_post_fitted_model, indices, intra_class_index,
                                      colors_selected_classes,
                                      f'HoroPCA (fitted on post-push) <br> Pre-Push prototypes | Epoch:{epoch_num}')
    fig4 = create_interactive_scatter(CO_SNE_embeddings, indices, intra_class_index, colors_selected_classes,
                                      f'CO-SNE <br> Pre-Push prototypes | Epoch:{epoch_num}')
    fig5 = create_interactive_scatter(CO_SNE_embeddings_pushed, indices, intra_class_index, colors_selected_classes,
                                      f'CO-SNE <br> Post-Push prototypes | Epoch:{epoch_num}')
    fig6 = create_interactive_scatter(CO_SNE_embeddings_post_fitted_model_large, indices, intra_class_index,
                                      colors_selected_classes,
                                      f'CO-SNE (fitted on post-push) <br> Pre-Push prototypes | Epoch:{epoch_num}')
    # Display all plots
    # fig1.show()
    # fig2.show()
    # fig3.show()
    # fig4.show()
    # fig5.show()
    # fig6.show()

    # save the interactive figures so I can open them up in interactive mode
    fig1.write_html(get_fig_path(model_root_dir, "3_interactive_pre_push", epoch_num, format='html'))
    fig2.write_html(get_fig_path(model_root_dir, "4_interactive_post_push", epoch_num, format='html'))
    fig3.write_html(get_fig_path(model_root_dir, "5_interactive_post_fitted_model", epoch_num, format='html'))
    fig4.write_html(get_fig_path(model_root_dir, "6_interactive_cosne_pre_push", epoch_num, format='html'))
    fig5.write_html(get_fig_path(model_root_dir, "7_interactive_cosne_post_push", epoch_num, format='html'))
    fig6.write_html(get_fig_path(model_root_dir, "8_interactive_cosne_post_fitted", epoch_num, format='html'))

    ###################################################################
    # Interactive plot with Image popup as hover happens
    #####################################################################
    # os.path.dirname(prot_broad_path), os.path.dirname(prot_local_path)

    # # get the image paths
    # broad_prot_image_paths = get_prot_image_paths(os.path.dirname(prot_broad_path))
    # local_prot_image_paths = get_prot_image_paths(os.path.dirname(prot_local_path))

    # # # make the list of image_path.
    # image_paths = local_prot_image_paths + broad_prot_image_paths
    # print(len(image_paths))
    # fig2 = create_interactive_scatter(embeddings_pushed_2d, indices, intra_class_index, colors_selected_classes, f'HoroPCA <br> Post-Push prototypes | Epoch:{epoch_num}',
    #                                   image_paths=image_paths)
    # fig2.show()
    # # save the interactive figures so I can open them up in interactive mode
    # fig2.write_html(get_fig_path(model_root_dir, "9_interactive_post_push_with_images", epoch_num, format='html'))

    # # sanity check, check if PIL works with one of the images
    # img = Image.open(image_paths[0])
    # img.show()
    # #sanirty check if the image is in format of base64
    # image_to_base64(image_paths[0])


    ###################################################################
    # Log in wandb
    #####################################################################
    if log_wandb != "disabled":
        # wandb.log({"embedding_visualization/prototypes": wandb.Image(grid), "epoch": epoch_num})
        # wandb.log({"embedding_visualization/prototypes_shortlist": wandb.Image(grid_shortlist), "epoch": epoch_num})

        # # log fig_movement in wandb
        # wandb.log({"embedding_visualization/prototypes_pushing_movement": wandb.Image(fig_movement), "epoch": epoch_num})

        # # log fig
        # wandb.log({"embedding_visualization/prototypes_all": wandb.Image(fig_prototypes_all), "epoch": epoch_num})
        # grid.shape

        # log fig_movement in wandb

        # log fig
        wandb.log({
            "embedding_visualization/prototypes_all": wandb.Image(fig_prototypes_all),
            "embedding_visualization/prototypes_shortlist": wandb.Image(fig_shortlist_classes),
            "embedding_visualization/prototypes_pushing_movement": wandb.Image(fig_movement),
            "embedding_visualization/interactive_HoroPCA_pre_push": wandb.Html(fig1.to_html(full_html=False)),
            "embedding_visualization/interactive_HoroPCA_post_push": wandb.Html(fig2.to_html(full_html=False)),
            "embedding_visualization/interactive_HoroPCA_post_fitted_model": wandb.Html(fig3.to_html(full_html=False)),
            "embedding_visualization/interactive_CoSNE_pre_push": wandb.Html(fig4.to_html(full_html=False)),
            "embedding_visualization/interactive_CoSNE_post_push": wandb.Html(fig5.to_html(full_html=False)),
            "embedding_visualization/interactive_CoSNE_post_fitted_model": wandb.Html(fig6.to_html(full_html=False)),
            "epoch": epoch_num,
        })


def plot_distance_histogram(root_local_distances, root_broad_distances, title, epoch, model_root_dir, plot_name="distance_to_root", save_fig=True):
    fig_dsit_to_root, ax = plt.subplots()
    ax.hist(root_local_distances, bins=20, alpha=0.5, label='local')
    ax.hist(root_broad_distances, bins=40, alpha=0.5, label='broad')
    ax.legend()
    ax.set_xlabel("Lorentz Distance to Origin of Hyperboloid")
    ax.set_ylabel("Number of Prototypes")
    ax.set_title(f"{title} | Epoch:{epoch}")

    plt.tight_layout()
    
    # save the figure
    if save_fig:
        plt.savefig(get_fig_path(model_root_dir, plot_name, epoch), bbox_inches='tight', dpi=fig_dsit_to_root.dpi)

    return fig_dsit_to_root


def plot_distance_histogram_separate(root_local_distances, root_broad_distances, title, epoch, model_root_dir, plot_name="distance_to_root_sep", save_fig=True):
    fig_dsit_to_root_separate, axs = plt.subplots(2, 1)#, figsize=(10, 10))

    # plot histograms of distances of broad prototypes to the root, on the first subplot
    axs[0].hist(root_broad_distances, bins=40, alpha=0.5, label='broad', color='red')
    axs[0].legend()
    axs[0].tick_params(axis='both', which='major')#, labelsize=15)
    axs[0].set_ylabel("Number of Prototypes")#, fontsize=20)
    # plot histograms of distances of local prototypes to the root, on the second subplot
    axs[1].hist(root_local_distances, bins=20, alpha=0.5, label='local', color='blue')
    axs[1].legend()
    axs[1].tick_params(axis='both', which='major')#, labelsize=15)
    axs[1].set_ylabel("Number of Prototypes")#, fontsize=20)

    fig_dsit_to_root_separate.suptitle(f"{title} | Epoch:{epoch}")
    plt.xlabel("Lorentz Distance to Origin of Hyperboloid")#, fontsize=20)

    plt.tight_layout()

    # save the figure
    if save_fig:
        plt.savefig(get_fig_path(model_root_dir, plot_name, epoch), bbox_inches='tight', dpi=fig_dsit_to_root_separate.dpi)

    return fig_dsit_to_root_separate


# Example Usage (replace with your actual values)
if __name__ == '__main__':
    model_runname = '0_Debug_Entailment_Jump-PushGlob_NoReplace-idx_00'
    model_root_dir = f"../../logs/logs_sockeye/pets/Hyperbolic_XProtoNet_e2e/{model_runname}"  # Replace with your model directory
    epoch_num = 35

    emb_class_names = None

    visualize_prototype_embedding_space(model_root_dir, epoch_num, emb_class_names, log_wandb="disabled")