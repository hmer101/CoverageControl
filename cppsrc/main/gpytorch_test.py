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
    def __init__(self, train_x, train_y, likelihood, use_priors=False,
                 lengthscale_prior=None, outputscale_prior=None,
                 noise_prior=None, mean_prior=None, init_at_priors=False):
        """
        Initialize GP model with optional informative priors.

        Parameters
        ----------
        train_x, train_y : torch.Tensor
            Training data
        likelihood : gpytorch.likelihoods.GaussianLikelihood
            Likelihood function
        use_priors : bool
            If True, apply the specified priors
        lengthscale_prior : tuple of (mean, std) or None
            LogNormal prior for lengthscale (mean, std of underlying normal distribution).
            If None, uses GPyTorch default.
        outputscale_prior : tuple of (mean, std) or None
            LogNormal prior for outputscale (mean, std of underlying normal distribution).
            If None, uses GPyTorch default.
        noise_prior : tuple of (mean, std) or None
            LogNormal prior for noise (mean, std of underlying normal distribution).
            If None, uses GPyTorch default.
        mean_prior : tuple of (mean, std) or None
            Normal prior for constant mean. If None, uses GPyTorch default
        init_at_priors : bool
            If True, initialize hyperparameters to prior medians/means.
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = gpytorch.means.LinearMean(input_size=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # Set informative priors for adaptive sampling with few points
        if use_priors:
            # Prior on lengthscale: spatial correlation scale (in normalized space)
            # LogNormal prior ensures positive values for scale parameter
            if lengthscale_prior is not None:
                self.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.LogNormalPrior(
                    lengthscale_prior[0], lengthscale_prior[1])

            # Prior on outputscale: signal variance (in normalized space)
            # LogNormal prior ensures positive values for scale parameter
            if outputscale_prior is not None:
                self.covar_module.outputscale_prior = gpytorch.priors.LogNormalPrior(
                    outputscale_prior[0], outputscale_prior[1])

            # Prior on noise: measurement noise (in normalized space)
            # LogNormal prior ensures positive values for scale parameter
            if noise_prior is not None:
                likelihood.noise_prior = gpytorch.priors.LogNormalPrior(
                    noise_prior[0], noise_prior[1])

            # Prior on mean: expected mean value (in normalized space)
            # Normal prior allows both positive and negative values
            if mean_prior is not None:
                self.mean_module.constant_prior = gpytorch.priors.NormalPrior(
                    mean_prior[0], mean_prior[1])
        
        if init_at_priors:
            if lengthscale_prior is not None:
                # Initialize lengthscale to prior median
                self.covar_module.base_kernel.lengthscale = np.exp(lengthscale_prior[0])
            if outputscale_prior is not None:
                # Initialize outputscale to prior median
                self.covar_module.outputscale = np.exp(outputscale_prior[0])
            if noise_prior is not None:
                # Initialize noise to prior median
                likelihood.noise = np.exp(noise_prior[0])
            if mean_prior is not None:
                # Initialize mean to prior mean
                self.mean_module.constant = mean_prior[0]

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


### UTILITY FUNCTIONS ###
# Data loading and preprocessing
def downsample_by_grid(gdf, grid_size):
    """
    Downsample a GeoDataFrame by selecting one point per grid cell.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame with 'east' and 'north' columns in meters.
    grid_size : float
        The size of the grid cells in meters.

    Returns
    -------
    GeoDataFrame
        Downsampled GeoDataFrame.
    """
    # Create grid cell IDs
    gdf['grid_x'] = (gdf['east'] // grid_size).astype(int)
    gdf['grid_y'] = (gdf['north'] // grid_size).astype(int)

    # For each grid cell, select one random sample
    gdf_sampled = gdf.groupby(['grid_x', 'grid_y']).sample(1, random_state=42)

    return gdf_sampled.drop(columns=['grid_x', 'grid_y'])


def load_data(data_path, num_samples=-1, percent_train=0.8, normalize=False, moisture_trim_range=None, normalization_bounds=None, downsample_grid_size=None):
    gdf = gpd.read_file(data_path)
    # print(gdf.head())
    # print(gdf.columns)

    # Extract x (longitude) and y (latitude)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    # Keep only points with valid moisture readings
    gdf = gdf.dropna(subset=["Moisture"])

    # Trim moisture values outside of the specified range if requested
    if moisture_trim_range is not None:
        min_moisture, max_moisture = moisture_trim_range
        gdf = gdf[(gdf["Moisture"] >= min_moisture) & (gdf["Moisture"] <= max_moisture)]

    # Always convert to ENU (meters) for the full dataset
    lat_ref = gdf["lat"].min()
    lon_ref = gdf["lon"].min()
    alt_ref = 0.0
    east, north, up = pm.geodetic2enu(
        gdf["lat"].values,
        gdf["lon"].values,
        np.zeros_like(gdf["lat"].values),
        lat_ref, lon_ref, alt_ref
    )
    gdf["east"] = east
    gdf["north"] = north

    # Now, perform sampling
    gdf_sample = gdf
    if downsample_grid_size is not None:
        print(f"Downsampling data to one point per {downsample_grid_size}m grid...")
        gdf_sample = downsample_by_grid(gdf, downsample_grid_size)
        print(f"Original points: {len(gdf)}, Downsampled points: {len(gdf_sample)}")
    elif num_samples != -1:
        gdf_sample = gdf.sample(num_samples, random_state=42)

    # Extract data in ENU coordinates
    data_x = torch.tensor(np.column_stack([gdf_sample["east"], gdf_sample["north"]]), dtype=torch.float32)
    data_y = torch.tensor(gdf_sample["Moisture"].values, dtype=torch.float32)

    # Normalize if requested
    normalization_params = None
    if normalize:
        data_x, data_y, normalization_params = normalize_data(data_x, data_y, normalization_bounds=normalization_bounds)

    # Split into training and testing sets
    n = len(data_x)
    n_train = int(percent_train * n)
    train_x, test_x = data_x[:n_train], data_x[n_train:]
    train_y, test_y = data_y[:n_train], data_y[n_train:]

    return train_x, train_y, test_x, test_y, normalization_params

# Normalization
def normalize_data(data_x, data_y, normalization_bounds=None):
    """
    Normalize input features (x) and output (y) to [0, 1] range.
    Uses uniform scaling for x inputs to preserve relative spatial scale.

    Parameters
    ----------
    data_x : torch.Tensor
        Input features (N, 2) - either (lon, lat) or (east, north)
    data_y : torch.Tensor
        Output values (N,) - moisture
    normalization_bounds : dict, optional
        If provided, use these bounds for normalization instead of calculating them from data.
        Expected keys: 'x_min', 'x_max', 'y_min', 'y_max'.

    Returns
    -------
    data_x_norm : torch.Tensor
        Normalized input features
    data_y_norm : torch.Tensor
        Normalized output values
    normalization_params : dict
        Dictionary containing normalization parameters
    """
    if normalization_bounds is not None:
        x_min = normalization_bounds['x_min']
        x_max = normalization_bounds['x_max']
        y_min = normalization_bounds['y_min']
        y_max = normalization_bounds['y_max']

        x_range = x_max - x_min
        x_scale = x_range.max()
        y_scale = y_max - y_min
    else:
        # Get min/max for each x dimension
        x_min = data_x.min(dim=0)[0]
        x_max = data_x.max(dim=0)[0]

        # Use the maximum range across both dimensions for uniform scaling
        # This preserves the aspect ratio of the spatial coordinates
        x_range = x_max - x_min
        x_scale = x_range.max()

        # Normalize y (output) using range
        y_min = data_y.min()
        y_max = data_y.max()
        y_scale = y_max - y_min

    # Normalize x (inputs) using uniform scale
    data_x_norm = (data_x - x_min) / x_scale

    # Normalize y (output) using range
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

def normalize_priors(lengthscale_meters, outputscale_moisture, noise_moisture, mean_moisture,
                                          normalization_params, lengthscale_uncertainty=0.5,
                                          outputscale_uncertainty=0.5, noise_uncertainty=0.5, mean_std_moisture=None):
    """
    Convert physically meaningful prior specifications to normalized LogNormal/Normal prior parameters.

    Parameters
    ----------
    lengthscale_meters : float or tuple
        Spatial lengthscale in meters. Can be a single value (median) or (median, std_factor).
    outputscale_moisture : float or tuple
        Output scale (signal variance) in moisture percentage squared units (%²).
    noise_moisture : float or tuple
        Noise variance in moisture percentage squared units (%²).
    mean_moisture : float
        Expected mean moisture value in percentage.
    normalization_params : dict
        Dictionary containing normalization parameters (x_scale, y_scale, y_min).
    lengthscale_uncertainty : float
        Uncertainty factor for lengthscale (std of log-normal distribution). Default 0.5.
    outputscale_uncertainty : float
        Uncertainty factor for outputscale. Default 0.5.
    noise_uncertainty : float
        Uncertainty factor for noise. Default 0.5.
    mean_std_moisture : float or None
        Standard deviation for mean prior in moisture percentage units. If None, uses outputscale_moisture/2.

    Returns
    -------
    lengthscale_prior : tuple
        (mean, std) for LogNormal prior in normalized space
    outputscale_prior : tuple
        (mean, std) for LogNormal prior in normalized space
    noise_prior : tuple
        (mean, std) for LogNormal prior in normalized space
    mean_prior : tuple
        (mean, std) for Normal prior in normalized space

    Notes
    -----
    For LogNormal distribution with parameters (mu, sigma):
    - median = exp(mu)
    - mean = exp(mu + sigma^2/2)
    - To set median to m: mu = log(m)
    """
    # Extract normalization parameters
    x_scale = normalization_params['x_scale']
    if torch.is_tensor(x_scale):
        x_scale = x_scale.item()

    y_scale = normalization_params['y_scale']
    if torch.is_tensor(y_scale):
        y_scale = y_scale.item()

    y_min = normalization_params['y_min']
    if torch.is_tensor(y_min):
        y_min = y_min.item()

    # Convert lengthscale from meters to normalized space
    lengthscale_norm = lengthscale_meters / x_scale
    # LogNormal: mu = log(median), sigma = uncertainty factor
    lengthscale_prior = (np.log(lengthscale_norm), lengthscale_uncertainty)

    # Convert outputscale from moisture units to normalized space
    # Outputscale is variance, so scale by y_scale (not y_scale^2, since we're in normalized [0,1] space)
    outputscale_norm = outputscale_moisture / y_scale
    outputscale_prior = (np.log(outputscale_norm), outputscale_uncertainty)

    # Convert noise from moisture units to normalized space
    noise_norm = noise_moisture / y_scale
    noise_prior = (np.log(noise_norm), noise_uncertainty)

    # Convert mean from moisture units to normalized space
    # Normal distribution: (mean, std) both in normalized space
    mean_norm = (mean_moisture - y_min) / y_scale
    if mean_std_moisture is None:
        mean_std_moisture = outputscale_moisture / 2.0  # Default: half of output scale
    mean_std_norm = mean_std_moisture / y_scale
    mean_prior = (mean_norm, mean_std_norm)

    print("\n=== Physical Priors Converted to Normalized Space ===")
    print(f"Lengthscale: {lengthscale_meters:.1f}m -> {lengthscale_norm:.4f} (normalized)")
    print(f"  LogNormal prior: mu={lengthscale_prior[0]:.3f}, sigma={lengthscale_prior[1]:.3f}")
    print(f"  Median in normalized space: {np.exp(lengthscale_prior[0]):.4f}")
    print(f"Outputscale: {outputscale_moisture:.2f}%² -> {outputscale_norm:.4f} (normalized)")
    print(f"  LogNormal prior: mu={outputscale_prior[0]:.3f}, sigma={outputscale_prior[1]:.3f}")
    print(f"  Median in normalized space: {np.exp(outputscale_prior[0]):.4f}")
    print(f"Noise: {noise_moisture:.3f}%² -> {noise_norm:.5f} (normalized)")
    print(f"  LogNormal prior: mu={noise_prior[0]:.3f}, sigma={noise_prior[1]:.3f}")
    print(f"  Median in normalized space: {np.exp(noise_prior[0]):.5f}")
    print(f"Mean: {mean_moisture:.1f}% -> {mean_norm:.4f} (normalized)")
    print(f"  Normal prior: mean={mean_prior[0]:.4f}, std={mean_prior[1]:.4f}")
    print("="*60)

    return lengthscale_prior, outputscale_prior, noise_prior, mean_prior

# Denormalization
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

def denormalize_hyperparameters(model, likelihood, normalization_params):
    """
    Denormalize model hyperparameters to physical units.

    Parameters
    ----------
    model : ExactGPModel
        The GP model
    likelihood : GaussianLikelihood
        The likelihood
    normalization_params : dict
        Normalization parameters

    Returns
    -------
    dict : Dictionary with denormalized hyperparameters
    """
    # Get normalized values
    lengthscale_norm = model.covar_module.base_kernel.lengthscale.item()
    outputscale_norm = model.covar_module.outputscale.item()
    noise_norm = likelihood.noise.item()

    if isinstance(model.mean_module, gpytorch.means.ConstantMean):
        mean_norm = model.mean_module.constant.item()
    else:
        mean_norm = None

    # Extract normalization parameters
    x_scale = normalization_params['x_scale'].item() if torch.is_tensor(normalization_params['x_scale']) else normalization_params['x_scale']
    y_min = normalization_params['y_min'].item() if torch.is_tensor(normalization_params['y_min']) else normalization_params['y_min']
    y_scale = normalization_params['y_scale'].item() if torch.is_tensor(normalization_params['y_scale']) else normalization_params['y_scale']

    # Denormalize
    lengthscale_phys = lengthscale_norm * x_scale
    outputscale_phys = outputscale_norm * y_scale
    noise_phys = noise_norm * y_scale
    mean_phys = mean_norm * y_scale + y_min if mean_norm is not None else None

    return {
        'lengthscale': lengthscale_phys,
        'outputscale': outputscale_phys,
        'noise': noise_phys,
        'mean': mean_phys
    }

def denormalize_priors(model, likelihood, normalization_params):
    """
    Denormalize model priors to physical units.

    Parameters
    ----------
    model : ExactGPModel
        The GP model
    likelihood : GaussianLikelihood
        The likelihood
    normalization_params : dict
        Normalization parameters

    Returns
    -------
    dict : Dictionary with denormalized prior parameters
    """
    result = {}

    # Extract normalization parameters
    x_scale = normalization_params['x_scale'].item() if torch.is_tensor(normalization_params['x_scale']) else normalization_params['x_scale']
    y_min = normalization_params['y_min'].item() if torch.is_tensor(normalization_params['y_min']) else normalization_params['y_min']
    y_scale = normalization_params['y_scale'].item() if torch.is_tensor(normalization_params['y_scale']) else normalization_params['y_scale']

    # Lengthscale prior
    if hasattr(model.covar_module.base_kernel, 'lengthscale_prior') and model.covar_module.base_kernel.lengthscale_prior is not None:
        ls_prior = model.covar_module.base_kernel.lengthscale_prior
        ls_loc_norm = ls_prior.loc.item()
        ls_scale_norm = ls_prior.scale.item()
        ls_median_norm = np.exp(ls_loc_norm)
        ls_median_phys = ls_median_norm * x_scale
        result['lengthscale'] = {
            'median': ls_median_phys,
            'scale': ls_scale_norm,
            'median_norm': ls_median_norm
        }

    # Outputscale prior
    if hasattr(model.covar_module, 'outputscale_prior') and model.covar_module.outputscale_prior is not None:
        os_prior = model.covar_module.outputscale_prior
        os_loc_norm = os_prior.loc.item()
        os_scale_norm = os_prior.scale.item()
        os_median_norm = np.exp(os_loc_norm)
        os_median_phys = os_median_norm * y_scale
        result['outputscale'] = {
            'median': os_median_phys,
            'scale': os_scale_norm,
            'median_norm': os_median_norm
        }

    # Noise prior
    if hasattr(likelihood, 'noise_prior') and likelihood.noise_prior is not None:
        n_prior = likelihood.noise_prior
        n_loc_norm = n_prior.loc.item()
        n_scale_norm = n_prior.scale.item()
        n_median_norm = np.exp(n_loc_norm)
        n_median_phys = n_median_norm * y_scale
        result['noise'] = {
            'median': n_median_phys,
            'scale': n_scale_norm,
            'median_norm': n_median_norm
        }

    # Mean prior
    if hasattr(model.mean_module, 'constant_prior') and model.mean_module.constant_prior is not None:
        mean_prior = model.mean_module.constant_prior
        mean_loc_norm = mean_prior.loc.item()
        mean_scale_norm = mean_prior.scale.item()
        mean_loc_phys = mean_loc_norm * y_scale + y_min
        mean_scale_phys = mean_scale_norm * y_scale
        result['mean'] = {
            'loc': mean_loc_phys,
            'scale': mean_scale_phys,
            'loc_norm': mean_loc_norm,
            'scale_norm': mean_scale_norm
        }

    return result

# Printing
def print_scaled_hyperparameters(model, likelihood, normalization_params=None, training_mode=False, training_iter=0, training_iter_max=0, training_loss=0.0):
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
    if not training_mode:
        print("\n=== Learned Hyperparameters ===")

    if normalization_params is not None:
        # Use denormalize_hyperparameters helper to get physical units
        hyper_phys = denormalize_hyperparameters(model, likelihood, normalization_params)

        if training_mode:
            print(f"Iter {training_iter + 1}/{training_iter_max} - Loss: {training_loss:.3f} "
                f"  lengthscale: {hyper_phys['lengthscale']:.2f}m "
                f"  outputscale: {hyper_phys['outputscale']:.3f}%² "
                f"  noise: {hyper_phys['noise']:.3f}%² "
                f"  mean: {hyper_phys['mean']:.2f}%")
        else:
            print(f"Lengthscale: {hyper_phys['lengthscale']:.2f}m")
            print(f"Outputscale: {hyper_phys['outputscale']:.3f}%²")
            print(f"Noise: {hyper_phys['noise']:.3f}%²")
            print(f"Mean: {hyper_phys['mean']:.2f}%")

    else:
        # No normalization - print normalized values
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        outputscale = model.covar_module.outputscale.item()
        noise = likelihood.noise.item()

        if training_mode:
            if isinstance(model.mean_module, gpytorch.means.ConstantMean):
                mean_param = model.mean_module.constant.item()
            elif isinstance(model.mean_module, gpytorch.means.LinearMean):
                weights = model.mean_module.weights.detach().cpu().numpy()
                bias = model.mean_module.bias.item()
                mean_param = f"bias={bias:.3f}, w={weights}"
            else:
                mean_param = "N/A"

            print(f"Iter {training_iter + 1}/{training_iter_max} - Loss: {training_loss:.3f} "
                f"  lengthscale: {lengthscale:.3f} "
                f"  outputscale: {outputscale:.3f} "
                f"  noise: {noise:.3f} "
                f"  mean: {mean_param}")
        else:
            if isinstance(model.mean_module, gpytorch.means.ConstantMean):
                mean_param = model.mean_module.constant.item()
                print(f"Lengthscale: {lengthscale:.3f}")
                print(f"Outputscale: {outputscale:.3f}")
                print(f"Noise: {noise:.6f}")
                print(f"Mean: {mean_param:.3f}")
            elif isinstance(model.mean_module, gpytorch.means.LinearMean):
                weights = model.mean_module.weights.detach().cpu().numpy()
                bias = model.mean_module.bias.item()
                print(f"Lengthscale: {lengthscale:.3f}")
                print(f"Outputscale: {outputscale:.3f}")
                print(f"Noise: {noise:.6f}")
                print(f"Mean: bias={bias:.3f}, w={weights}")
            else:
                print(f"Lengthscale: {lengthscale:.3f}")
                print(f"Outputscale: {outputscale:.3f}")
                print(f"Noise: {noise:.6f}")
                print(f"Mean: N/A")

def print_prior_configuration(model, likelihood, normalization_params=None):
    """
    Print the prior configuration of the model.

    Parameters
    ----------
    model : ExactGPModel
        The GP model
    likelihood : GaussianLikelihood
        The likelihood
    normalization_params : dict or None
        If provided, scale priors back to original units for display
    """
    print(f"\n=== Prior Configuration ===")

    if normalization_params is not None:
        # Use denormalize_priors helper to get physical units
        priors_phys = denormalize_priors(model, likelihood, normalization_params)

        # Mean prior
        if 'mean' in priors_phys:
            mean_info = priors_phys['mean']
            print(f"Mean prior: Normal(loc={mean_info['loc']:.2f}%, scale={mean_info['scale']:.2f}%) "
                  f"[normalized: loc={mean_info['loc_norm']:.4f}, scale={mean_info['scale_norm']:.4f}]")
        else:
            print(f"Mean prior: None")

        # Lengthscale prior
        if 'lengthscale' in priors_phys:
            ls_info = priors_phys['lengthscale']
            print(f"Lengthscale prior: LogNormal(median={ls_info['median']:.2f}m, scale={ls_info['scale']:.2f}) "
                  f"[normalized median: {ls_info['median_norm']:.4f}]")
        else:
            print(f"Lengthscale prior: None")

        # Outputscale prior
        if 'outputscale' in priors_phys:
            os_info = priors_phys['outputscale']
            print(f"Outputscale prior: LogNormal(median={os_info['median']:.3f}%², scale={os_info['scale']:.2f}) "
                  f"[normalized median: {os_info['median_norm']:.4f}]")
        else:
            print(f"Outputscale prior: None")

        # Noise prior
        if 'noise' in priors_phys:
            n_info = priors_phys['noise']
            print(f"Noise prior: LogNormal(median={n_info['median']:.3f}%, scale={n_info['scale']:.2f}) "
                  f"[normalized median: {n_info['median_norm']:.5f}]")
        else:
            print(f"Noise prior: None")
    else:
        # No normalization - print raw normalized values
        # Check mean prior
        if hasattr(model.mean_module, 'constant_prior') and model.mean_module.constant_prior is not None:
            mean_prior = model.mean_module.constant_prior
            mean_loc = mean_prior.loc.item()
            mean_scale = mean_prior.scale.item()
            print(f"Mean prior: Normal(loc={mean_loc:.4f}, scale={mean_scale:.4f})")
        else:
            print(f"Mean prior: None")

        # Check lengthscale prior
        if hasattr(model.covar_module.base_kernel, 'lengthscale_prior') and model.covar_module.base_kernel.lengthscale_prior is not None:
            ls_prior = model.covar_module.base_kernel.lengthscale_prior
            ls_loc = ls_prior.loc.item()
            ls_scale = ls_prior.scale.item()
            print(f"Lengthscale prior: LogNormal(loc={ls_loc:.4f}, scale={ls_scale:.4f})")
        else:
            print(f"Lengthscale prior: None")

        # Check outputscale prior
        if hasattr(model.covar_module, 'outputscale_prior') and model.covar_module.outputscale_prior is not None:
            os_prior = model.covar_module.outputscale_prior
            os_loc = os_prior.loc.item()
            os_scale = os_prior.scale.item()
            print(f"Outputscale prior: LogNormal(loc={os_loc:.4f}, scale={os_scale:.4f})")
        else:
            print(f"Outputscale prior: None")

        # Check noise prior
        if hasattr(likelihood, 'noise_prior') and likelihood.noise_prior is not None:
            n_prior = likelihood.noise_prior
            n_loc = n_prior.loc.item()
            n_scale = n_prior.scale.item()
            print(f"Noise prior: LogNormal(loc={n_loc:.4f}, scale={n_scale:.4f})")
        else:
            print(f"Noise prior: None")

    print("="*60)


### TRAINING ###
def train(model, likelihood, train_x, train_y, lr=0.1, training_iter=50, patience=50, min_delta=1e-4,
          warm_start=False, prev_model=None, plot_loss=True, normalization_params=None):
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

        # Print hyperparameters (in physical units if normalization_params provided)
        print_scaled_hyperparameters(model, likelihood, normalization_params=normalization_params, training_mode=True, training_iter=i, training_iter_max=training_iter, training_loss=loss.item())

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
                               lr=0.01, training_iter=50, patience=50, min_delta=1e-4, plot_loss=True,
                               use_priors=False, lengthscale_prior=None, outputscale_prior=None,
                               noise_prior=None, mean_prior=None, normalization_params=None):
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
    use_priors : bool
        If True, apply informative priors to the model
    lengthscale_prior, outputscale_prior, noise_prior, mean_prior : tuple or None
        Prior parameters (mean, std) for each hyperparameter
    normalization_params : dict or None
        Normalization parameters for denormalizing hyperparameters during training

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

    # Create new model with updated data and priors
    updated_model = ExactGPModel(train_x, train_y, likelihood,
                                 use_priors=use_priors,
                                 lengthscale_prior=lengthscale_prior,
                                 outputscale_prior=outputscale_prior,
                                 noise_prior=noise_prior,
                                 mean_prior=mean_prior).to(device)

    # Train with warm start from previous model
    train(updated_model, likelihood, train_x, train_y,
          lr=lr, training_iter=training_iter, patience=patience, min_delta=min_delta,
          warm_start=True, prev_model=prev_model, plot_loss=plot_loss,
          normalization_params=normalization_params)

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



### MAIN FUNCTION ###
def main():
    # PARAMETERS
    # 'Dads' is about 60 acres
    data_path = '/marl_sim_basic/data/soil_moisture/jason_soil_data/2024_North_of_shop_Dads_Export_20250630_1310_soil_moisture/Frantom Farm_Dads_North of sho_Harvest_2024-10-03_00.shp'

    # General options
    normalize_data_flag=True
    moisture_trim_range = (10, 30) # Set to None to disable

    # Preset normalization parameters (in physical units)
    # Set to None to calculate from data
    # preset_normalization_bounds = {
    #     'x_min': torch.tensor([363000.0, 4305000.0]),  # Example values for Easting, Northing
    #     'x_max': torch.tensor([364000.0, 4306000.0]),
    #     'y_min': torch.tensor(10.0),
    #     'y_max': torch.tensor(30.0)
    # }
    preset_normalization_bounds = None # to disable

    downsample_grid_size = 60 # in meters. Set to None to disable.

    #####
    # Training parameters
    #####
    # Where to train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_samples=5000 #-1 (GPU out of memory data is 16,465 long) #500
    percent_train=0.8 #0.7 #0.8

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
    enable_realtime_single_point = False  # Set to True to add points one-by-one with live plot updates
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
    plot_training_diagnostics = True  # Set to False to disable loss curves and eval plots during incremental training


    #####
    # Prior configuration for adaptive sampling (in physical units)
    #####
    use_priors = True  # Enable informative priors for faster convergence with few samples
    warm_start_from_priors = True # Initialize model at prior medians

    # Physical prior specifications (will be converted to normalized space automatically)
    # Lengthscale: spatial correlation distance in meters
    #   For soil moisture: typically 5-20m depending on field variability
    lengthscale_meters = 10.0  # Median expected lengthscale in meters

    # Outputscale: signal magnitude in moisture percentage
    #   Typical range of soil moisture variation across the field
    outputscale_moisture = 0.5  # Expect ~0.5% typical variation in moisture

    # Noise: measurement noise variance in moisture percentage squared
    #   Soil moisture sensors typically have ~0.5-1% accuracy, so variance is (0.5)²=0.25 to 1²=1
    noise_moisture = 0.1  # Median expected noise variance: 0.1%²

    # Mean: expected mean moisture value in percentage
    #   Center of your typical moisture range
    mean_moisture = 17.0  # Expected mean: 17% (center of [16.3, 18.4] range)

    # Uncertainty factors for LogNormal priors (sigma parameter)
    #   Higher values = more uncertainty. 0.5 is moderate, 1.0 is high uncertainty
    lengthscale_uncertainty = 0.7  # Moderate uncertainty in spatial scale
    outputscale_uncertainty = 0.7  # Moderate uncertainty in signal magnitude
    noise_uncertainty = 0.5  # Lower uncertainty in noise (we know sensor specs)

    # Mean prior standard deviation in moisture percentage
    mean_std_moisture = 0.5  # Allow mean to vary by ±0.5% around expected value


    #####
    # LOAD DATA
    #####
    train_x_full, train_y_full, test_x, test_y, normalization_params = load_data(
        data_path,
        num_samples=num_samples,
        percent_train=percent_train,
        normalize=normalize_data_flag,
        moisture_trim_range=moisture_trim_range,
        normalization_bounds=preset_normalization_bounds,
        downsample_grid_size=downsample_grid_size
    )

    # Convert physical priors to normalized space
    if use_priors and normalization_params is not None:
        lengthscale_prior, outputscale_prior, noise_prior, mean_prior = normalize_priors(
            lengthscale_meters=lengthscale_meters,
            outputscale_moisture=outputscale_moisture,
            noise_moisture=noise_moisture,
            mean_moisture=mean_moisture,
            normalization_params=normalization_params,
            lengthscale_uncertainty=lengthscale_uncertainty,
            outputscale_uncertainty=outputscale_uncertainty,
            noise_uncertainty=noise_uncertainty,
            mean_std_moisture=mean_std_moisture
        )
    else:
        lengthscale_prior = None
        outputscale_prior = None
        noise_prior = None
        mean_prior = None

    # Move test data to device once
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    if enable_realtime_single_point:
        training_iter_single = 20
        patience_single = 10
        
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
        model = ExactGPModel(train_x_current, train_y_current, likelihood,
                             use_priors=use_priors,
                             lengthscale_prior=lengthscale_prior,
                             outputscale_prior=outputscale_prior,
                             noise_prior=noise_prior,
                             mean_prior=mean_prior,
                             init_at_priors=warm_start_from_priors).to(device)

        # Check priors before training
        print_prior_configuration(model, likelihood, normalization_params)

        print(f"\nInitial training with 1 point...")
        train(model, likelihood, train_x_current, train_y_current,
              lr=lr_update, training_iter=training_iter_single, patience=patience_single, min_delta=min_delta, plot_loss=False,
              normalization_params=normalization_params)

        # Check priors after training
        print_prior_configuration(model, likelihood, normalization_params)

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
                    training_iter=training_iter_single,  
                    patience=patience_single, 
                    min_delta=min_delta,
                    plot_loss=False,
                    use_priors=use_priors,
                    lengthscale_prior=lengthscale_prior,
                    outputscale_prior=outputscale_prior,
                    noise_prior=noise_prior,
                    mean_prior=mean_prior,
                    normalization_params=normalization_params
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

        if len(test_x) > 0:
            rmse, coverage = evaluate_model(model, likelihood, test_x, test_y, normalization_params, plot_results=plot_training_diagnostics)

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
        model = ExactGPModel(train_x_current, train_y_current, likelihood,
                             use_priors=use_priors,
                             lengthscale_prior=lengthscale_prior,
                             outputscale_prior=outputscale_prior,
                             noise_prior=noise_prior,
                             mean_prior=mean_prior,
                             init_at_priors=warm_start_from_priors).to(device)

        print(f"\n{'='*60}")
        print(f"INITIAL TRAINING: {chunk_size} samples (10%)")
        print(f"{'='*60}")

        train(model, likelihood, train_x_current, train_y_current,
              lr=lr_initial, training_iter=training_iter_initial,
              patience=patience_initial, min_delta=min_delta, plot_loss=plot_training_diagnostics,
              normalization_params=normalization_params)

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
                plot_loss=plot_training_diagnostics,
                use_priors=use_priors,
                lengthscale_prior=lengthscale_prior,
                outputscale_prior=outputscale_prior,
                noise_prior=noise_prior,
                mean_prior=mean_prior,
                normalization_params=normalization_params
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
        model = ExactGPModel(train_x_current, train_y_current, likelihood,
                             use_priors=use_priors,
                             lengthscale_prior=lengthscale_prior,
                             outputscale_prior=outputscale_prior,
                             noise_prior=noise_prior,
                             mean_prior=mean_prior,
                             init_at_priors=warm_start_from_priors).to(device)

        print(f"\n{'='*60}")
        print(f"STANDARD TRAINING")
        print(f"Training samples: {len(train_x_current)}")
        print(f"{'='*60}")

        train(model, likelihood, train_x_current, train_y_current,
              lr=lr_initial, training_iter=training_iter_initial,
              patience=patience_initial, min_delta=min_delta, plot_loss=plot_training_diagnostics,
              normalization_params=normalization_params)

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