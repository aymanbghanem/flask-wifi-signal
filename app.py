import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.path as mpltPath
from flask import Flask, request, jsonify
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
app = Flask(__name__)


# Function to calculate polygon area using the shoelace formula
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    return area
# Function to generate the plot and return it as base64-encoded string
def generate_plot(x_coords, y_coords, signal_strengths):
    positions = np.column_stack((x_coords, y_coords))
    
    # Create the polygon path
    polygon_vertices = np.vstack((positions, positions[0]))
    polygon_path = mpltPath.Path(polygon_vertices)
    
    # Calculate the area of the polygon
    area = polygon_area(polygon_vertices)
    
    # Create the interpolated grid
    min_x, min_y = np.min(polygon_vertices, axis=0)
    max_x, max_y = np.max(polygon_vertices, axis=0)
    grid_resolution = 200
    grid_x, grid_y = np.mgrid[min_x:max_x:grid_resolution*1j, min_y:max_y:grid_resolution*1j]
    grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    inside_mask = polygon_path.contains_points(grid_points)
    grid_points_inside = grid_points[inside_mask]
    
    grid_signal = griddata(positions, signal_strengths, grid_points_inside, method='linear')
    grid_signal = np.nan_to_num(grid_signal, nan=np.nanmin(signal_strengths))
    grid_signal_full = np.full(grid_x.shape, np.nan)
    grid_signal_full_flat = grid_signal_full.flatten()
    grid_signal_full_flat[inside_mask] = grid_signal
    grid_signal_full = grid_signal_full_flat.reshape(grid_x.shape)
    
    # Calculate area with signal strength between -30 dBm and -70 dBm
    min_strength = -130
    max_strength = -30
    signal_mask = (grid_signal_full >= min_strength) & (grid_signal_full <= max_strength)
    valid_mask = ~np.isnan(grid_signal_full)
    combined_mask = signal_mask & valid_mask
    
    delta_x = (max_x - min_x) / (grid_resolution - 1)
    delta_y = (max_y - min_y) / (grid_resolution - 1)
    cell_area = delta_x * delta_y
    num_cells = np.count_nonzero(combined_mask)
    area_between_30_and_50 = num_cells * cell_area
    percentage = (area_between_30_and_50 / area) * 100
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = ax.imshow(
        grid_signal_full.T, extent=(min_x, max_x, min_y, max_y),
        origin='lower', cmap='RdYlGn', alpha=0.8, aspect='equal',
        vmin=-30,vmax=-130
    )
    cbar = fig.colorbar(heatmap)
    cbar.set_label('Signal Strength (dBm)')
    ax.plot(polygon_vertices[:, 0], polygon_vertices[:, 1], 'k-', linewidth=2, label='Polygon Boundary')
    ax.scatter(x_coords, y_coords, c='black', edgecolors='white', s=50, label='Measurement Points')
    ax.set_title('Wi-Fi Signal Strength Heat Map within the Polygon')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Convert plot to base64
    img_io = io.BytesIO()
    FigureCanvas(fig).print_png(img_io)
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf8')
    
    plt.close(fig)
    
    # Return text summary and base64 plot
    return {
        "area": f"{area:.2f} square meters",
        "area_between_30_and_50": f"{area_between_30_and_50:.2f} square meters",
        "percentage": f"{percentage:.2f}%",
        "image_base64": img_base64
    }
@app.route('/generate-plot', methods=['POST'])
def generate_plot_api():
    data = request.json
    x_coords = np.array(data['x'])
    y_coords = np.array(data['y'])
    signal_strengths = np.array(data['strength'])
    
    result = generate_plot(x_coords, y_coords, signal_strengths)
    
    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True)
