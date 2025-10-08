import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
import numpy as np

import plotly.io as pio
import geopandas as gpd
import pandas as pd

#import plotly.graph_objects as go
#from mpl_toolkits.mplot3d import Axes3D  # this activates 3D projection

import pyvista as pv
import pymap3d as pm
import vtk



# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def load_data(data_path, num_samples=-1, percent_train=0.8):
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

    # Extract data
    data_x = torch.tensor(gdf_sample[["lon", "lat"]].values, dtype=torch.float32)
    data_y = torch.tensor(gdf_sample["Moisture"].values, dtype=torch.float32)

    # Split into training and testing sets
    n = len(data_x)
    n_train = int(percent_train * n)
    train_x, test_x = data_x[:n_train], data_x[n_train:]
    train_y, test_y = data_y[:n_train], data_y[n_train:]

    return train_x, train_y, test_x, test_y, gdf_sample


def train(model, likelihood, train_x, train_y, training_iter=50):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()


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


def plot_surface(grid_lon, grid_lat, mean, train_x=None, train_y=None):
    """
    Plot the GP mean surface in 3D using PyVista (OpenGL-accelerated),
    with geodesic coordinates converted to local ENU coordinates using
    pymap3d.geodetic2enu(). The southwest (bottom-left) corner of the
    data is used as the reference origin (0, 0, 0).

    Parameters
    ----------
    grid_lon, grid_lat : 2D numpy arrays
        Meshgrid of longitude and latitude values.
    mean : 2D numpy array
        GP predicted mean values (same shape as grid_lon/grid_lat).
    train_x, train_y : torch.Tensors or numpy arrays, optional
        Training data points (lon, lat, moisture) to overlay as markers.
    """

    # Ensure numpy arrays
    grid_lon = np.asarray(grid_lon)
    grid_lat = np.asarray(grid_lat)
    mean = np.asarray(mean)

    # --- Use southwest corner as reference ---
    lat_ref = np.min(grid_lat)
    lon_ref = np.min(grid_lon)
    alt_ref = 0.0

    # Convert geodetic -> ENU (East, North, Up) in meters
    east, north, up = pm.geodetic2enu(
        grid_lat, grid_lon, np.zeros_like(grid_lat),  # alt = 0
        lat_ref, lon_ref, alt_ref
    )

    # Exaggerate z-axis by scaling mean values
    z_scale_factor = 100
    grid = pv.StructuredGrid(east, north, mean * z_scale_factor)
    # Store original values for color mapping
    grid["Soil Moisture"] = mean.ravel()

    # --- Create PyVista plotter ---
    plotter = pv.Plotter(window_size=[900, 700])
    plotter.add_mesh(
        grid,
        scalars="Soil Moisture",
        cmap="viridis",
        smooth_shading=True,
        scalar_bar_args={"title": "Soil Moisture (%)"},
    )

    # --- Optional: add training points ---
    if train_x is not None and train_y is not None:
        if torch.is_tensor(train_x):
            train_x = train_x.detach().cpu().numpy()
        if torch.is_tensor(train_y):
            train_y = train_y.detach().cpu().numpy()

        # Convert training (lon, lat) → ENU using same reference
        east_t, north_t, _ = pm.geodetic2enu(
            train_x[:,1], train_x[:,0], np.zeros_like(train_x[:,0]),
            lat_ref, lon_ref, alt_ref
        )

        # Apply same z-scaling to training points
        pts = np.c_[east_t, north_t, train_y * z_scale_factor]
        point_cloud = pv.PolyData(pts)
        plotter.add_points(
            point_cloud,
            color="black",
            point_size=6,
            render_points_as_spheres=True,
            label="Training Points",
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

    plotter.add_text("Gaussian Process Soil Moisture Surface", position="upper_edge", font_size=12)
    plotter.add_legend(labels=[["Predicted Surface", "w"], ["Training Points", "k"]])

    # plotter.camera_position = [
    #     (1168.2859048622065, -801.093154352905, 281.152515257392),  # camera position
    #     (201.55433299663719, 336.72302827131705, 16.850579261779785),  # focal point
    #     (-0.11084653681453323, 0.13455236898636372, 0.9846871103433731),  # view up
    # ]

    plotter.show()
    #print(plotter.camera_position)


def main():
    # PARAMETERS
    data_path = '/marl_sim_basic/data/soil_moisture/jason_soil_data/2024_North_of_shop_Dads_Export_20250630_1310_soil_moisture/Frantom Farm_Dads_North of sho_Harvest_2024-10-03_00.shp'

    num_samples=5000 #-1 (GPU out of memory) #500 #Data is 16,465 long 
    percent_train=1 #0.8
    training_iter=50 #500 #5000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # LOAD DATA
    train_x, train_y, _, _, data_subset = load_data(data_path, num_samples=num_samples, percent_train=percent_train)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # Move model, likelihood, and data to the selected device
    model.to(device)
    likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # TRAIN
    train(model, likelihood, train_x, train_y, training_iter=training_iter)


    # GENERATE TEST POINTS
    model.eval()
    likelihood.eval()
    
    lon_lin = np.linspace(data_subset.lon.min(), data_subset.lon.max(), 50)
    lat_lin = np.linspace(data_subset.lat.min(), data_subset.lat.max(), 50)
    grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)
    test_points = torch.tensor(np.column_stack([grid_lon.ravel(), grid_lat.ravel()]), dtype=torch.float32)

    # Move test points to device
    test_points = test_points.to(device)


    # PREDICT AT TEST POINTS
    with torch.no_grad():
        pred = likelihood(model(test_points))
        #pred = pred.detach().cpu().numpy()  # Move predictions back to CPU for numpy conversion

        mean = pred.mean.detach().cpu().numpy().reshape(grid_lon.shape)
        # lower, upper = pred.confidence_region()
        # lower = lower.numpy().reshape(grid_lon.shape)
        # upper = upper.numpy().reshape(grid_lon.shape)


    # PLOT SURFACE
    plot_surface(grid_lon, grid_lat, mean, train_x=train_x, train_y=train_y)
    

if __name__ == "__main__":
    main()