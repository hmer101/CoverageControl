import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
import numpy as np
import time

import plotly.io as pio
import geopandas as gpd
import pandas as pd

#import plotly.graph_objects as go
#from mpl_toolkits.mplot3d import Axes3D  # this activates 3D projection

import pyvista as pv
import pyvistaqt as pvqt
import pymap3d as pm
import vtk


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = gpytorch.means.LinearMean(input_size=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

### UTILITY FUNCTIONS ###
def normalize_data(data_x, data_y):
    """
    Normalize input features (x) and output (y) to [0, 1] range.
    Uses uniform scaling for x inputs to preserve relative spatial scale.

    Parameters
    ----------
    data_x : torch.Tensor
        Input features (N, 2) - either (lon, lat) or (east, north)
    data_y : torch.Tensor
        Output values (N,) - moisture

    Returns
    -------
    data_x_norm : torch.Tensor
        Normalized input features
    data_y_norm : torch.Tensor
        Normalized output values
    normalization_params : dict
        Dictionary containing normalization parameters
    """
    # Get min/max for each x dimension
    x_min = data_x.min(dim=0)[0]
    x_max = data_x.max(dim=0)[0]

    # Use the maximum range across both dimensions for uniform scaling
    # This preserves the aspect ratio of the spatial coordinates
    x_range = x_max - x_min
    x_scale = x_range.max()

    # Normalize x (inputs) using uniform scale
    data_x_norm = (data_x - x_min) / x_scale

    # Normalize y (output) using range
    y_min = data_y.min()
    y_max = data_y.max()
    y_scale = y_max - y_min
    data_y_norm = (data_y - y_min) / y_scale

    # Clamp minimums to be at least 0
    x_min = torch.clamp_min(x_min, 0)
    y_min = torch.clamp_min(y_min, 0)

    # Create normalization parameters dictionary
    normalization_params = {
        'x_min': x_min,
        'x_scale': x_scale,
        'y_min': y_min,
        'y_scale': y_scale
    }

    return data_x_norm, data_y_norm, normalization_params

def denormalize(data_norm, data_min, data_scale):
    """
    Denormalize data from [0, 1] back to original scale.

    Parameters
    ----------
    data_norm : torch.Tensor or np.ndarray
        Normalized data
    data_min : torch.Tensor, np.ndarray, or float
        Minimum values for each dimension
    data_scale : torch.Tensor, np.ndarray, or float
        Scale factor (range) for each dimension

    Returns
    -------
    data : torch.Tensor or np.ndarray
        Denormalized data
    """
    return data_norm * data_scale + data_min

def denormalize_predictions(grid_eval_x, grid_eval_y, mean, lower, upper, variance, stddev,
                           train_x, train_y, test_x, test_y, normalization_params):
    """
    Denormalize all prediction outputs and data from a GP model.

    Parameters
    ----------
    grid_eval_x, grid_eval_y : np.ndarray
        Normalized evaluation grid coordinates
    mean : np.ndarray
        Normalized mean predictions
    lower : np.ndarray
        Normalized lower confidence bounds
    upper : np.ndarray
        Normalized upper confidence bounds
    variance : np.ndarray
        Normalized variance
    stddev : np.ndarray
        Normalized standard deviation
    train_x, train_y : torch.Tensor or np.ndarray
        Normalized training data
    test_x, test_y : torch.Tensor or np.ndarray
        Normalized test data
    normalization_params : dict or None
        Normalization parameters containing x_min, x_max, y_min, y_max

    Returns
    -------
    grid_eval_x, grid_eval_y, mean, lower, upper, variance, stddev, train_x, train_y, test_x, test_y
        All denormalized data
    """
    if normalization_params is None:
        return grid_eval_x, grid_eval_y, mean, lower, upper, variance, stddev, train_x, train_y, test_x, test_y

    # Convert tensors to numpy if needed
    x_min_np = normalization_params['x_min'].cpu().numpy() if torch.is_tensor(normalization_params['x_min']) else normalization_params['x_min']
    x_scale_np = normalization_params['x_scale'].cpu().numpy() if torch.is_tensor(normalization_params['x_scale']) else normalization_params['x_scale']
    y_min_np = normalization_params['y_min'].cpu().numpy() if torch.is_tensor(normalization_params['y_min']) else normalization_params['y_min']
    y_scale_np = normalization_params['y_scale'].cpu().numpy() if torch.is_tensor(normalization_params['y_scale']) else normalization_params['y_scale']

    # Denormalize grid coordinates using uniform x_scale
    grid_eval_x = denormalize(grid_eval_x, x_min_np[0], x_scale_np)
    grid_eval_y = denormalize(grid_eval_y, x_min_np[1], x_scale_np)

    # Denormalize predictions (mean, lower, upper)
    mean = denormalize(mean, y_min_np, y_scale_np)
    lower = denormalize(lower, y_min_np, y_scale_np)
    upper = denormalize(upper, y_min_np, y_scale_np)

    # For variance/stddev, scale by y_scale squared/y_scale
    variance = variance * (y_scale_np ** 2)
    stddev = np.sqrt(variance)

    # Denormalize training data
    if train_x is not None and train_y is not None:
        if torch.is_tensor(train_x):
            train_x = train_x.detach().cpu().numpy()
        else:
            train_x = np.asarray(train_x)
        if torch.is_tensor(train_y):
            train_y = train_y.detach().cpu().numpy()
        else:
            train_y = np.asarray(train_y)

        # Denormalize using uniform x_scale for both dimensions
        train_x_denorm = denormalize(train_x, x_min_np, x_scale_np)
        train_y = denormalize(train_y, y_min_np, y_scale_np)
        train_x = train_x_denorm

    # Denormalize test data
    if test_x is not None and test_y is not None:
        if torch.is_tensor(test_x):
            test_x = test_x.detach().cpu().numpy()
        else:
            test_x = np.asarray(test_x)
        if torch.is_tensor(test_y):
            test_y = test_y.detach().cpu().numpy()
        else:
            test_y = np.asarray(test_y)

        # Denormalize using uniform x_scale for both dimensions
        test_x_denorm = denormalize(test_x, x_min_np, x_scale_np)
        test_y = denormalize(test_y, y_min_np, y_scale_np)
        test_x = test_x_denorm

    return grid_eval_x, grid_eval_y, mean, lower, upper, variance, stddev, train_x, train_y, test_x, test_y

def load_data(data_path, num_samples=-1, percent_train=0.8, normalize=False):
    gdf = gpd.read_file(data_path)
    # print(gdf.head())
    # print(gdf.columns)

    # Extract x (longitude) and y (latitude)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    # Keep only points with valid moisture readings
    gdf = gdf.dropna(subset=["Moisture"])

    # Downsample for quick training if a limit is set
    if num_samples == -1:
        gdf_sample = gdf
    else:
        gdf_sample = gdf.sample(num_samples, random_state=42)

    # Always convert to ENU (meters)
    # Use southwest corner as reference
    lat_ref = gdf_sample["lat"].min()
    lon_ref = gdf_sample["lon"].min()
    alt_ref = 0.0

    # Convert geodetic -> ENU (East, North, Up) in meters
    east, north, up = pm.geodetic2enu(
        gdf_sample["lat"].values,
        gdf_sample["lon"].values,
        np.zeros_like(gdf_sample["lat"].values),
        lat_ref, lon_ref, alt_ref
    )

    # Store ENU coordinates
    gdf_sample["east"] = east
    gdf_sample["north"] = north

    # Extract data in ENU coordinates
    data_x = torch.tensor(np.column_stack([east, north]), dtype=torch.float32)

    data_y = torch.tensor(gdf_sample["Moisture"].values, dtype=torch.float32)

    # Normalize if requested
    normalization_params = None
    if normalize:
        data_x, data_y, normalization_params = normalize_data(data_x, data_y)

    # Split into training and testing sets
    n = len(data_x)
    n_train = int(percent_train * n)
    train_x, test_x = data_x[:n_train], data_x[n_train:]
    train_y, test_y = data_y[:n_train], data_y[n_train:]

    return train_x, train_y, test_x, test_y, normalization_params

### TRAINING ###
def train(model, likelihood, train_x, train_y, lr=0.1, training_iter=50, patience=50, min_delta=1e-4,
          warm_start=False, prev_model=None, plot_loss=True):
    """
    Train a GP model.

    Parameters
    ----------
    model : ExactGPModel
        Model to train
    likelihood : GaussianLikelihood
        Likelihood function
    train_x, train_y : torch.Tensor
        Training data
    lr : float
        Learning rate
    training_iter : int
        Maximum number of training iterations
    patience : int
        Early stopping patience
    min_delta : float
        Minimum improvement for early stopping
    warm_start : bool
        If True, initialize hyperparameters from prev_model
    prev_model : ExactGPModel, optional
        Previous model to copy hyperparameters from (required if warm_start=True)
    plot_loss : bool
        If True, plot the loss curve after training
    """
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Warm start: copy hyperparameters from previous model
    if warm_start and prev_model is not None:
        model.load_state_dict(prev_model.state_dict())

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    best_loss = float('inf')
    patience_counter = 0

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()

        loss_value = loss.item()
        losses.append(loss_value)

        # Print all hyperparameters
        lengthscale = model.covar_module.base_kernel.lengthscale.item() #.detach().cpu().numpy()
        outputscale = model.covar_module.outputscale.item()
        noise = model.likelihood.noise.item()

        if isinstance(model.mean_module, gpytorch.means.ConstantMean):
            mean_param = model.mean_module.constant.item()
        elif isinstance(model.mean_module, gpytorch.means.LinearMean):
            weights = model.mean_module.weights.detach().cpu().numpy()
            bias = model.mean_module.bias.item()
            mean_param = f"bias={bias:.3f}, w={weights}"
        else:
            mean_param = "N/A"

        print(f"Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f} "
            f"  lengthscale: {lengthscale:.3f} "
            f"  outputscale: {outputscale:.3f} "
            f"  noise: {noise:.3f} "
            f"  mean: {mean_param}")
        optimizer.step()

        # Early stopping check
        if loss_value < best_loss - min_delta:
            best_loss = loss_value
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'\nEarly stopping at iteration {i+1}. No improvement for {patience} iterations.')
            break

    # Plot loss curve
    if plot_loss:
        plot_loss_curve(losses)

    return losses

def update_model_with_new_data(prev_model, likelihood, new_x, new_y,
                               train_x, train_y, device,
                               lr=0.01, training_iter=50, patience=50, min_delta=1e-4, plot_loss=True):
    """
    Add new observations to the model and perform a quick hyperparameter update.

    Parameters
    ----------
    prev_model : ExactGPModel
        Current trained model
    likelihood : GaussianLikelihood
        Current likelihood
    new_x, new_y : torch.Tensor
        New observations to add (should be on same device as train_x/train_y)
    train_x, train_y : torch.Tensor
        Current training data
    device : torch.device
        Device to use (CPU or CUDA)
    lr : float
        Learning rate for update (typically smaller than initial training)
    training_iter : int
        Number of optimization iterations (typically much smaller than initial training)
    patience : int
        Early stopping patience
    min_delta : float
        Minimum improvement for early stopping
    plot_loss : bool
        If True, plot the loss curve after training

    Returns
    -------
    updated_model : ExactGPModel
        Model with new data and updated hyperparameters
    train_x, train_y : torch.Tensor
        Updated training data
    """
    # Concatenate new data with existing training data
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])

    # Create new model with updated data
    updated_model = ExactGPModel(train_x, train_y, likelihood).to(device)

    # Train with warm start from previous model
    train(updated_model, likelihood, train_x, train_y,
          lr=lr, training_iter=training_iter, patience=patience, min_delta=min_delta,
          warm_start=True, prev_model=prev_model, plot_loss=plot_loss)

    return updated_model, train_x, train_y


### TESTING / PREDICTION ###
def generate_eval_grid(train_x, test_x, num_points=50):
    """
    Generate a grid of test points for prediction.
    Uses the combined bounds of train_x and test_x to determine grid extent.

    Parameters
    ----------
    train_x : torch.Tensor or np.ndarray
        Training input features (N, 2) - already normalized if normalization was used
    test_x : torch.Tensor or np.ndarray
        Test input features (M, 2) - already normalized if normalization was used
    num_points : int
        Number of points along each dimension

    Returns
    -------
    grid_x : np.ndarray (2D)
        X coordinates of grid (in same space as train_x/test_x)
    grid_y : np.ndarray (2D)
        Y coordinates of grid (in same space as train_x/test_x)
    test_points : torch.Tensor
        Flattened test points as (N, 2) tensor
    """
    # Combine train and test data to get full extent
    if torch.is_tensor(train_x):
        train_x_np = train_x.detach().cpu().numpy()
    else:
        train_x_np = np.asarray(train_x)

    if torch.is_tensor(test_x):
        test_x_np = test_x.detach().cpu().numpy()
    else:
        test_x_np = np.asarray(test_x)

    all_x = np.vstack([train_x_np, test_x_np])

    # Get bounds from combined data
    x_min = all_x[:, 0].min()
    x_max = all_x[:, 0].max()
    y_min = all_x[:, 1].min()
    y_max = all_x[:, 1].max()

    # Generate grid
    x_lin = np.linspace(x_min, x_max, num_points)
    y_lin = np.linspace(y_min, y_max, num_points)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin)

    # Create test points tensor
    test_points = torch.tensor(np.column_stack([grid_x.ravel(), grid_y.ravel()]), dtype=torch.float32)

    return grid_x, grid_y, test_points

### PLOTTING ###
def plot_surface_matplotlib(grid_lon, grid_lat, mean, train_x=None, train_y=None):
    """
    Plot the GP mean surface in 3D.
    grid_lon, grid_lat: 2D numpy arrays from np.meshgrid
    mean: 2D numpy array of predicted mean values reshaped to grid shape
    train_x, train_y: Optional training data for overlaying points
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 3D surface
    surf = ax.plot_surface(
        grid_lon, grid_lat, mean,
        cmap='viridis', linewidth=0, antialiased=False, alpha=0.9
    )

    # Optionally plot training points
    if train_x is not None and train_y is not None:
        ax.scatter(
            train_x[:, 0].numpy(),
            train_x[:, 1].numpy(),
            train_y.numpy(),
            color='k', s=20, label='Training points'
        )

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Soil Moisture (%)')
    ax.set_title('Gaussian Process Surface (3D)')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Predicted Moisture')
    plt.legend()
    plt.show()

def plot_surface(grid_east, grid_north, mean, train_x=None, train_y=None, test_x=None, test_y=None,
                 stddev=None, plot_cis=False, lower=None, upper=None, z_scale_factor=100, clim=None,
                 block=True, title_suffix="", plotter=None):
    """
    Plot the GP mean surface in 3D using PyVista (OpenGL-accelerated).
    Data is assumed to already be in ENU coordinates (meters).

    Parameters
    ----------
    grid_east, grid_north : 2D numpy arrays
        Meshgrid of east/north coordinates in meters.
    mean : 2D numpy array
        GP predicted mean values (same shape as grid_east/grid_north).
    train_x, train_y : torch.Tensors or numpy arrays, optional
        Training data points (east, north, moisture) to overlay as markers.
    test_x, test_y : torch.Tensors or numpy arrays, optional
        Test data points (east, north, moisture) to overlay as red stars.
    stddev : 2D numpy array, optional
        Standard deviation of predictions (for uncertainty visualization).
    lower, upper : 2D numpy arrays, optional
        Lower and upper confidence bounds (for plotting confidence surfaces).
    z_scale_factor : float
        Exaggeration factor for z-axis.
    clim : list, optional
        Color limits [min, max] for consistent color mapping.
    block : bool
        If True (default), blocks execution until window is closed.
        If False, opens window non-blocking for side-by-side viewing.
    title_suffix : str
        Additional text to append to the plot title.
    plotter : pv.Plotter or pvqt.BackgroundPlotter, optional
        Existing plotter to update. If provided, the plotter will be cleared and reused
        instead of creating a new one. Useful for real-time updates.

    Returns
    -------
    plotter : pv.Plotter or pv.BackgroundPlotter
        The plotter object (only returned for non-blocking mode or when reusing plotter)
    """

    # Ensure numpy arrays
    east = np.asarray(grid_east)
    north = np.asarray(grid_north)
    mean = np.asarray(mean)

    # All data is assumed to be already denormalized before calling this function

    # Exaggerate z-axis by scaling mean values
    #z_scale_factor = z_scale_factor
    grid = pv.StructuredGrid(east, north, mean * z_scale_factor)
    # Store original values for color mapping
    #grid["Soil Moisture"] = mean.ravel()
    grid["Soil Moisture"] = mean.ravel(order="F")  # ensure Fortran order (y-major)

    # --- Create or reuse PyVista plotter ---
    cmap = 'RdYlGn'

    # If plotter is provided, clear it and reuse
    if plotter is not None:
        plotter.clear()
    # Otherwise create new plotter
    elif not block:
        plotter = pvqt.BackgroundPlotter(window_size=(900, 700), title=title_suffix if title_suffix else "GP Surface")
    else:
        plotter = pv.Plotter(window_size=[900, 700])
    plotter.add_mesh(
        grid,
        scalars="Soil Moisture",
        cmap=cmap, #"viridis",
        clim=clim,       # fixes the range
        smooth_shading=True,
        scalar_bar_args={"title": "Soil Moisture (%)"},
    )

    # --- Optional: add training points ---
    if train_x is not None and train_y is not None:
        train_x_np = np.asarray(train_x)
        train_y_np = np.asarray(train_y)

        # Apply same z-scaling to training points
        pts = np.c_[train_x_np[:, 0], train_x_np[:, 1], train_y_np * z_scale_factor]
        point_cloud = pv.PolyData(pts)
        plotter.add_points(
            point_cloud,
            color="black",
            point_size=6,
            render_points_as_spheres=True,
            label="Training Points",
        )

    # --- Optional: add test points ---
    if test_x is not None and test_y is not None:
        test_x_np = np.asarray(test_x)
        test_y_np = np.asarray(test_y)

        # Apply same z-scaling to test points
        pts_test = np.c_[test_x_np[:, 0], test_x_np[:, 1], test_y_np * z_scale_factor]
        point_cloud_test = pv.PolyData(pts_test)

        # Use a star glyph for test points
        plotter.add_points(
            point_cloud_test,
            color="blue",
            point_size=6,
            render_points_as_spheres=True,
            label="Test Points",
        )

    # --- Appearance and camera setup ---
    # Show the basic axes marker (not the grid)
    plotter.show_axes()

    # Create custom cube axes with corrected z-axis labels
    cube_axes_actor = vtk.vtkCubeAxesActor()
    cube_axes_actor.SetBounds(grid.bounds)
    cube_axes_actor.SetCamera(plotter.renderer.GetActiveCamera())

    # Customize axis titles
    cube_axes_actor.SetXTitle('East (m)')
    cube_axes_actor.SetYTitle('North (m)')
    cube_axes_actor.SetZTitle('Soil Moisture (%)')

    # Set z-axis range to original values (not scaled)
    cube_axes_actor.SetZAxisRange(mean.min(), mean.max())
    cube_axes_actor.SetXAxisRange(grid.bounds[0], grid.bounds[1])
    cube_axes_actor.SetYAxisRange(grid.bounds[2], grid.bounds[3])

    # Set colors to black
    cube_axes_actor.GetXAxesLinesProperty().SetColor(0, 0, 0)
    cube_axes_actor.GetYAxesLinesProperty().SetColor(0, 0, 0)
    cube_axes_actor.GetZAxesLinesProperty().SetColor(0, 0, 0)

    # Set label and title colors to black and increase font sizes
    for i in range(3):
        title_prop = cube_axes_actor.GetTitleTextProperty(i)
        title_prop.SetColor(0, 0, 0)
        title_prop.SetFontSize(24)
        title_prop.SetBold(1)

        label_prop = cube_axes_actor.GetLabelTextProperty(i)
        label_prop.SetColor(0, 0, 0)
        label_prop.SetFontSize(18)
        label_prop.SetBold(1)

    # Also increase the line width of the axes
    cube_axes_actor.GetXAxesLinesProperty().SetLineWidth(2)
    cube_axes_actor.GetYAxesLinesProperty().SetLineWidth(2)
    cube_axes_actor.GetZAxesLinesProperty().SetLineWidth(2)

    # Turn off grid lines on the cube axes
    #cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_NONE)

    plotter.renderer.AddActor(cube_axes_actor)

    # --- Optional: add confidence interval surfaces ---
    if plot_cis and lower is not None and upper is not None:
        lower_arr = np.asarray(lower)
        upper_arr = np.asarray(upper)

        # Create lower and upper bound surfaces
        grid_lower = pv.StructuredGrid(east, north, lower_arr * z_scale_factor)
        grid_upper = pv.StructuredGrid(east, north, upper_arr * z_scale_factor)

        # Add semi-transparent surfaces for confidence bounds
        plotter.add_mesh(grid_lower, color='lightblue', opacity=0.2, label='95% CI Lower')
        plotter.add_mesh(grid_upper, color='lightblue', opacity=0.2, label='95% CI Upper')

    # Add title with optional suffix
    title_text = "Gaussian Process Soil Moisture Surface"
    if title_suffix:
        title_text += f" - {title_suffix}"
    plotter.add_text(title_text, position="upper_edge", font_size=12)

    # Build legend based on what's plotted
    legend_entries = [["Predicted Surface", "w"]]
    if train_x is not None and train_y is not None:
        legend_entries.append(["Training Points", "k"])
    if test_x is not None and test_y is not None:
        legend_entries.append(["Test Points", "b"])
    if plot_cis and lower is not None and upper is not None:
        legend_entries.append(["95% Confidence Interval", "lightblue"])

    plotter.add_legend(labels=legend_entries)

    # plotter.camera_position = [
    #     (1168.2859048622065, -801.093154352905, 281.152515257392),  # camera position
    #     (201.55433299663719, 336.72302827131705, 16.850579261779785),  # focal point
    #     (-0.11084653681453323, 0.13455236898636372, 0.9846871103433731),  # view up
    # ]

    # Show plot with blocking or non-blocking mode
    if block and plotter is not None:
        plotter.show()
        return None
    else:
        # For non-blocking mode or when reusing plotter: return plotter to keep reference alive
        # BackgroundPlotter from pyvistaqt shows automatically when created
        return plotter

def plot_loss_curve(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.grid(True)
    plt.show(block=False)


def print_scaled_hyperparameters(model, likelihood, normalization_params=None):
    """
    Print hyperparameters scaled back to original data units for interpretability.

    Parameters
    ----------
    model : ExactGPModel
        Trained GP model
    likelihood : GaussianLikelihood
        Trained likelihood
    normalization_params : dict or None
        If provided, scale hyperparameters back to original units
    """
    # Get hyperparameters in normalized space
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    outputscale = model.covar_module.outputscale.item()
    noise = likelihood.noise.item()

    if isinstance(model.mean_module, gpytorch.means.ConstantMean):
        mean_param = model.mean_module.constant.item()
    elif isinstance(model.mean_module, gpytorch.means.LinearMean):
        weights = model.mean_module.weights.detach().cpu().numpy()
        bias = model.mean_module.bias.item()
        mean_param = f"bias={bias:.3f}, w={weights}"
    else:
        mean_param = "N/A"

    print("\n=== Learned Hyperparameters ===")

    if normalization_params is not None:
        # Scale hyperparameters back to original units
        x_scale_np = normalization_params['x_scale'].cpu().numpy() if torch.is_tensor(normalization_params['x_scale']) else normalization_params['x_scale']
        y_min_np = normalization_params['y_min'].cpu().numpy() if torch.is_tensor(normalization_params['y_min']) else normalization_params['y_min']
        y_scale_np = normalization_params['y_scale'].cpu().numpy() if torch.is_tensor(normalization_params['y_scale']) else normalization_params['y_scale']

        # Scale lengthscale by x_scale (spatial units)
        lengthscale_scaled = lengthscale * x_scale_np

        # Scale outputscale and noise by y_scale (moisture units)
        outputscale_scaled = outputscale * y_scale_np
        noise_scaled = noise * y_scale_np

        # Scale mean by y_scale and add y_min
        if isinstance(model.mean_module, gpytorch.means.ConstantMean):
            mean_scaled = mean_param * y_scale_np + y_min_np
            mean_str = f"{mean_scaled:.3f}"
        elif isinstance(model.mean_module, gpytorch.means.LinearMean):
            # Weights need to be scaled by y_scale/x_scale, bias by y_scale + y_min
            weights_scaled = weights * y_scale_np / x_scale_np
            bias_scaled = bias * y_scale_np + y_min_np
            mean_str = f"bias={bias_scaled:.3f}, w={weights_scaled}"
        else:
            mean_str = "N/A"

        # Format lengthscale (may be array for ARD)
        if lengthscale_scaled.size == 1:
            lengthscale_str = f"{lengthscale_scaled.item():.3f} meters"
        else:
            lengthscale_str = f"East={lengthscale_scaled[0]:.3f}m, North={lengthscale_scaled[1]:.3f}m"

        print(f"Lengthscale (original units): {lengthscale_str}")
        print(f"Outputscale (original units): {outputscale_scaled:.3f} (moisture variance)")
        print(f"Noise (original units): {noise_scaled:.6f} (moisture std dev)")
        print(f"Mean (original units): {mean_str}")
        print("\nNormalized space hyperparameters:")
        if lengthscale.size == 1:
            print(f"  Lengthscale: {lengthscale.item():.3f}")
        else:
            print(f"  Lengthscale: East={lengthscale[0]:.3f}, North={lengthscale[1]:.3f}")
        print(f"  Outputscale: {outputscale:.3f}")
        print(f"  Noise: {noise:.6f}")
        print(f"  Mean: {mean_param}")
    else:
        # Format lengthscale (may be array for ARD)
        if lengthscale.size == 1:
            lengthscale_str = f"{lengthscale.item():.3f} meters"
        else:
            lengthscale_str = f"East={lengthscale[0]:.3f}m, North={lengthscale[1]:.3f}m"

        print(f"Lengthscale: {lengthscale_str}")
        print(f"Outputscale: {outputscale:.3f}")
        print(f"Noise: {noise:.6f}")

        # Format mean
        if isinstance(model.mean_module, gpytorch.means.ConstantMean):
            print(f"Mean: {mean_param:.3f}")
        else:
            print(f"Mean: {mean_param}")

def evaluate_model(model, likelihood, test_x, test_y, normalization_params=None, plot_results=True):
    """
    Evaluate model performance on test data.

    Parameters
    ----------
    model : ExactGPModel
        Trained GP model
    likelihood : GaussianLikelihood
        Trained likelihood
    test_x : torch.Tensor
        Test input features (N, 2)
    test_y : torch.Tensor
        True test output values (N,)
    normalization_params : dict or None
        If provided, denormalize predictions for evaluation
    plot_results : bool
        If True, plot true vs predicted scatter plot

    Returns
    -------
    rmse : float
        Root mean squared error
    coverage : float
        Percentage of true values within 95% confidence interval
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        pred = likelihood(model(test_x))
        pred_mean = pred.mean
        lower, upper = pred.confidence_region()

    # Move to CPU and convert to numpy
    pred_mean_np = pred_mean.detach().cpu().numpy()
    test_y_np = test_y.detach().cpu().numpy()
    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()

    # Denormalize if needed
    if normalization_params is not None:
        y_min_np = normalization_params['y_min'].cpu().numpy() if torch.is_tensor(normalization_params['y_min']) else normalization_params['y_min']
        y_scale_np = normalization_params['y_scale'].cpu().numpy() if torch.is_tensor(normalization_params['y_scale']) else normalization_params['y_scale']

        pred_mean_np = denormalize(pred_mean_np, y_min_np, y_scale_np)
        test_y_np = denormalize(test_y_np, y_min_np, y_scale_np)
        lower_np = denormalize(lower_np, y_min_np, y_scale_np)
        upper_np = denormalize(upper_np, y_min_np, y_scale_np)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((pred_mean_np - test_y_np)**2))

    # Calculate coverage (% of points within 95% CI)
    within_ci = np.sum((test_y_np >= lower_np) & (test_y_np <= upper_np))
    coverage = 100.0 * within_ci / len(test_y_np)

    # Plot true vs predicted
    if plot_results:
        plt.figure(figsize=(8, 8))

        # Scatter plot
        plt.scatter(test_y_np, pred_mean_np, alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(test_y_np.min(), pred_mean_np.min())
        max_val = max(test_y_np.max(), pred_mean_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('True Moisture (%)', fontsize=12)
        plt.ylabel('Predicted Moisture (%)', fontsize=12)
        plt.title(f'True vs Predicted Moisture\nRMSE: {rmse:.3f}, Coverage: {coverage:.1f}%', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show(block=False)

    print(f"\n=== Model Evaluation ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"95% CI Coverage: {coverage:.2f}%")
    print(f"Number of test points: {len(test_y_np)}")

    return rmse, coverage


### MAIN FUNCTION ###
def main():
    # PARAMETERS
    # 'Dads' is about 60 acres
    data_path = '/marl_sim_basic/data/soil_moisture/jason_soil_data/2024_North_of_shop_Dads_Export_20250630_1310_soil_moisture/Frantom Farm_Dads_North of sho_Harvest_2024-10-03_00.shp'

    # General options
    normalize_data_flag=True

    #####
    # Training parameters
    #####
    # Where to train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_samples=5000 #-1 (GPU out of memory data is 16,465 long) #500
    percent_train=0.8

    # Initial training parameters (for first 10% of data)
    lr_initial=0.005 #0.1 - not normalized, 0.005 - normalized
    training_iter_initial=4000 #2500 #5000 #500 #5000
    min_delta=1e-4
    patience_initial=1000 #200 #For normalized: 1000, for unnormalized: 200

    #####
    # Update parameters
    #####
    # Update parameters
    lr_update=0.01  # Slightly higher LR for quick updates
    training_iter_update=200  # Much fewer iterations for updates
    patience_update=50

    # Incremental training parameters
    enable_incremental = False #True  # Set to False to use original behavior
    num_chunks = 8  # Train on 10%, 20%, 30%, ..., 80% of data

    # Real-time single-point update mode
    enable_realtime_single_point = True #False  # Set to True to add points one-by-one with live plot updates
    realtime_pause_seconds = 0.0 #0.5  # Pause between adding each point
    realtime_max_points = 500 #100  # Maximum number of points to add in real-time mode
    realtime_update_interval = 1  # Update model every N points (1 = every point, 5 = every 5 points, etc.)


    #####
    # Visualization parameters
    #####
    z_scale_factor=50 #100 # For exaggerating z-axis in 3D plot
    clim = [16.3, 18.4] # color limits for consistent color mapping with John Deere operations center data
    plot_cis=True #False

    # Plot control flags
    plot_training_diagnostics = False  # Set to False to disable loss curves and eval plots during incremental training


    # LOAD DATA
    train_x_full, train_y_full, test_x, test_y, normalization_params = load_data(
        data_path,
        num_samples=num_samples,
        percent_train=percent_train,
        normalize=normalize_data_flag
    )

    # Move test data to device once
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    if enable_realtime_single_point:
        # REAL-TIME SINGLE-POINT UPDATE MODE
        print(f"\n{'='*60}")
        print(f"REAL-TIME SINGLE-POINT UPDATE MODE")
        print(f"Adding up to {realtime_max_points} points one at a time")
        print(f"Update interval: every {realtime_update_interval} point(s)")
        print(f"Pause between updates: {realtime_pause_seconds} seconds")
        print(f"{'='*60}\n")

        # Start with first point
        train_x_current = train_x_full[:1].to(device)
        train_y_current = train_y_full[:1].to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x_current, train_y_current, likelihood).to(device)

        # Initial quick training
        print(f"Initial training with 1 point...")
        train(model, likelihood, train_x_current, train_y_current,
              lr=lr_update, training_iter=50, patience=10, min_delta=min_delta, plot_loss=False)

        # Create single BackgroundPlotter that we'll update
        plotter = pvqt.BackgroundPlotter(window_size=(900, 700), title="Real-time GP Update")

        # Determine how many points to add
        n_points_to_add = min(realtime_max_points, len(train_x_full) - 1)

        for point_idx in range(1, n_points_to_add + 1):
            print(f"\rAdding point {point_idx}/{n_points_to_add}...", end='', flush=True)

            # Add new point
            new_x = train_x_full[point_idx:point_idx+1].to(device)
            new_y = train_y_full[point_idx:point_idx+1].to(device)

            # Update model every N points
            if point_idx % realtime_update_interval == 0 or point_idx == 1:
                model, train_x_current, train_y_current = update_model_with_new_data(
                    prev_model=model,
                    likelihood=likelihood,
                    new_x=new_x,
                    new_y=new_y,
                    train_x=train_x_current,
                    train_y=train_y_current,
                    device=device,
                    lr=lr_update,
                    training_iter=20,  # Quick updates
                    patience=10,
                    min_delta=min_delta,
                    plot_loss=False
                )
            else:
                # Just add data without retraining
                train_x_current = torch.cat([train_x_current, new_x])
                train_y_current = torch.cat([train_y_current, new_y])

            # Generate predictions
            model.eval()
            likelihood.eval()
            grid_eval_x, grid_eval_y, eval_points = generate_eval_grid(train_x_current, test_x, num_points=30)
            eval_points = eval_points.to(device)

            with torch.no_grad():
                pred = model(eval_points)
                mean = pred.mean.detach().cpu().numpy().reshape(grid_eval_x.shape)
                variance = pred.variance.detach().cpu().numpy().reshape(grid_eval_x.shape)
                stddev = np.sqrt(variance)
                lower, upper = pred.confidence_region()
                lower = lower.detach().cpu().numpy().reshape(grid_eval_x.shape)
                upper = upper.detach().cpu().numpy().reshape(grid_eval_x.shape)

            # Denormalize
            grid_eval_x_denorm, grid_eval_y_denorm, mean_denorm, lower_denorm, upper_denorm, variance_denorm, stddev_denorm, train_x_plot, train_y_plot, test_x_plot, test_y_plot = denormalize_predictions(
                grid_eval_x, grid_eval_y, mean, lower, upper, variance, stddev,
                train_x_current, train_y_current, test_x, test_y, normalization_params
            )

            # Update plot using plot_surface (reuses existing plotter)
            plotter = plot_surface(
                grid_eval_x_denorm, grid_eval_y_denorm, mean_denorm,
                train_x=train_x_plot, train_y=train_y_plot,
                test_x=None, test_y=None,
                stddev=stddev_denorm, plot_cis=plot_cis, lower=lower_denorm, upper=upper_denorm,
                z_scale_factor=z_scale_factor,
                clim=clim,
                block=False,
                title_suffix=f"{point_idx} training points",
                plotter=plotter  # Reuse existing plotter
            )

            # Pause
            time.sleep(realtime_pause_seconds)

        # print(f"\n\nReal-time updates complete. Press Enter to close...")
        # input()

    elif enable_incremental:
        # INCREMENTAL TRAINING: Start with 10%, add 10% at a time up to 80%
        n_total = len(train_x_full)
        chunk_size = n_total // num_chunks

        print(f"\n{'='*60}")
        print(f"INCREMENTAL TRAINING")
        print(f"Total training samples: {n_total}")
        print(f"Chunk size (10%): {chunk_size}")
        print(f"Number of chunks: {num_chunks}")
        print(f"{'='*60}\n")

        # List to store plotter references (prevents garbage collection)
        plotters = []

        # Initialize model with first chunk (10% of data)
        train_x_current = train_x_full[:chunk_size].to(device)
        train_y_current = train_y_full[:chunk_size].to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x_current, train_y_current, likelihood).to(device)

        print(f"\n{'='*60}")
        print(f"INITIAL TRAINING: {chunk_size} samples (10%)")
        print(f"{'='*60}")

        train(model, likelihood, train_x_current, train_y_current,
              lr=lr_initial, training_iter=training_iter_initial,
              patience=patience_initial, min_delta=min_delta, plot_loss=plot_training_diagnostics)

        print_scaled_hyperparameters(model, likelihood, normalization_params)

        if len(test_x) > 0:
            rmse, coverage = evaluate_model(model, likelihood, test_x, test_y, normalization_params, plot_results=plot_training_diagnostics)
            #print(f"RMSE: {rmse:.4f}")

        # Generate and plot first surface
        model.eval()
        likelihood.eval()
        grid_eval_x, grid_eval_y, eval_points = generate_eval_grid(train_x_current, test_x, num_points=50)
        eval_points = eval_points.to(device)

        with torch.no_grad():
            pred = model(eval_points)
            mean = pred.mean.detach().cpu().numpy().reshape(grid_eval_x.shape)
            variance = pred.variance.detach().cpu().numpy().reshape(grid_eval_x.shape)
            stddev = np.sqrt(variance)
            lower, upper = pred.confidence_region()
            lower = lower.detach().cpu().numpy().reshape(grid_eval_x.shape)
            upper = upper.detach().cpu().numpy().reshape(grid_eval_x.shape)

        grid_eval_x_denorm, grid_eval_y_denorm, mean_denorm, lower_denorm, upper_denorm, variance_denorm, stddev_denorm, train_x_plot, train_y_plot, test_x_plot, test_y_plot = denormalize_predictions(
            grid_eval_x, grid_eval_y, mean, lower, upper, variance, stddev,
            train_x_current, train_y_current, test_x, test_y, normalization_params
        )

        plotter = plot_surface(
            grid_eval_x_denorm, grid_eval_y_denorm, mean_denorm,
            train_x=train_x_plot, train_y=train_y_plot,
            test_x=test_x_plot, test_y=test_y_plot,
            stddev=stddev_denorm, plot_cis=plot_cis, lower=lower_denorm, upper=upper_denorm,
            z_scale_factor=z_scale_factor,
            clim=clim,
            block=False,
            title_suffix=f"10% data (n={chunk_size})"
        )
        if plotter is not None:
            plotters.append(plotter)
        time.sleep(0.1)  # Give window time to initialize

        # Incrementally add remaining chunks
        for i in range(1, num_chunks):
            pct = (i + 1) * 10
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size

            print(f"\n{'='*60}")
            print(f"UPDATE {i}: Adding samples {start_idx} to {end_idx} ({pct}% total)")
            print(f"{'='*60}")

            # Get next chunk
            new_x = train_x_full[start_idx:end_idx].to(device)
            new_y = train_y_full[start_idx:end_idx].to(device)

            # Update model
            model, train_x_current, train_y_current = update_model_with_new_data(
                prev_model=model,
                likelihood=likelihood,
                new_x=new_x,
                new_y=new_y,
                train_x=train_x_current,
                train_y=train_y_current,
                device=device,
                lr=lr_update,
                training_iter=training_iter_update,
                patience=patience_update,
                min_delta=min_delta,
                plot_loss=plot_training_diagnostics
            )

            print_scaled_hyperparameters(model, likelihood, normalization_params)

            if len(test_x) > 0:
                rmse, coverage = evaluate_model(model, likelihood, test_x, test_y, normalization_params, plot_results=plot_training_diagnostics)
                #print(f"RMSE: {rmse:.4f}")

            # Generate and plot updated surface
            model.eval()
            likelihood.eval()
            grid_eval_x, grid_eval_y, eval_points = generate_eval_grid(train_x_current, test_x, num_points=50)
            eval_points = eval_points.to(device)

            with torch.no_grad():
                pred = model(eval_points)
                mean = pred.mean.detach().cpu().numpy().reshape(grid_eval_x.shape)
                variance = pred.variance.detach().cpu().numpy().reshape(grid_eval_x.shape)
                stddev = np.sqrt(variance)
                lower, upper = pred.confidence_region()
                lower = lower.detach().cpu().numpy().reshape(grid_eval_x.shape)
                upper = upper.detach().cpu().numpy().reshape(grid_eval_x.shape)

            grid_eval_x_denorm, grid_eval_y_denorm, mean_denorm, lower_denorm, upper_denorm, variance_denorm, stddev_denorm, train_x_plot, train_y_plot, test_x_plot, test_y_plot = denormalize_predictions(
                grid_eval_x, grid_eval_y, mean, lower, upper, variance, stddev,
                train_x_current, train_y_current, test_x, test_y, normalization_params
            )

            plotter = plot_surface(
                grid_eval_x_denorm, grid_eval_y_denorm, mean_denorm,
                train_x=train_x_plot, train_y=train_y_plot,
                test_x=test_x_plot, test_y=test_y_plot,
                stddev=stddev_denorm, plot_cis=plot_cis, lower=lower_denorm, upper=upper_denorm,
                z_scale_factor=z_scale_factor,
                clim=clim,
                block=False,
                title_suffix=f"{pct}% data (n={end_idx})"
            )
            if plotter is not None:
                plotters.append(plotter)
            time.sleep(0.1)  # Give window time to initialize

    else:
        # ORIGINAL MODE: Train on all data at once
        train_x_current = train_x_full.to(device)
        train_y_current = train_y_full.to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x_current, train_y_current, likelihood).to(device)

        print(f"\n{'='*60}")
        print(f"STANDARD TRAINING")
        print(f"Training samples: {len(train_x_current)}")
        print(f"{'='*60}")

        train(model, likelihood, train_x_current, train_y_current,
              lr=lr_initial, training_iter=training_iter_initial,
              patience=patience_initial, min_delta=min_delta, plot_loss=plot_training_diagnostics)

        print_scaled_hyperparameters(model, likelihood, normalization_params)

        if len(test_x) > 0:
            rmse, coverage = evaluate_model(model, likelihood, test_x, test_y, normalization_params, plot_results=plot_training_diagnostics)

        # Generate and plot surface
        model.eval()
        likelihood.eval()
        grid_eval_x, grid_eval_y, eval_points = generate_eval_grid(train_x_current, test_x, num_points=50)
        eval_points = eval_points.to(device)

        with torch.no_grad():
            pred = model(eval_points)
            mean = pred.mean.detach().cpu().numpy().reshape(grid_eval_x.shape)
            variance = pred.variance.detach().cpu().numpy().reshape(grid_eval_x.shape)
            stddev = np.sqrt(variance)
            lower, upper = pred.confidence_region()
            lower = lower.detach().cpu().numpy().reshape(grid_eval_x.shape)
            upper = upper.detach().cpu().numpy().reshape(grid_eval_x.shape)

        grid_eval_x_denorm, grid_eval_y_denorm, mean_denorm, lower_denorm, upper_denorm, variance_denorm, stddev_denorm, train_x_plot, train_y_plot, test_x_plot, test_y_plot = denormalize_predictions(
            grid_eval_x, grid_eval_y, mean, lower, upper, variance, stddev,
            train_x_current, train_y_current, test_x, test_y, normalization_params
        )

        plot_surface(
            grid_eval_x_denorm, grid_eval_y_denorm, mean_denorm,
            train_x=train_x_plot, train_y=train_y_plot,
            test_x=test_x_plot, test_y=test_y_plot,
            stddev=stddev_denorm, plot_cis=plot_cis, lower=lower_denorm, upper=upper_denorm,
            z_scale_factor=z_scale_factor,
            clim=clim,
            block=True
        )

    input("Press Enter to close...")  # keeps the program alive
    

if __name__ == "__main__":
    main()