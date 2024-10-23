import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.path as mpltPath
from scipy.spatial.distance import cdist
from flask import Flask, request, jsonify
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvas

app = Flask(__name__)

# Function to calculate polygon area using the shoelace formula
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    return area

# Function to simulate signal enhancement using a Gaussian function
def apply_signal_enhancement(grid_x, grid_y, grid_signal_full, enhancer_point, enhancer_strength=10, radius=2.0):
    """Simulate signal enhancement by adding a Gaussian bump around the enhancer_point."""
    distance = np.sqrt((grid_x - enhancer_point[0]) ** 2 + (grid_y - enhancer_point[1]) ** 2)
    enhancement = enhancer_strength * np.exp(- (distance ** 2) / (2 * radius ** 2))
    return grid_signal_full + enhancement

# Function to find the best special point for signal enhancement
def find_best_special_point(x_coords, y_coords, signal_strengths, grid_x, grid_y, grid_signal_full, min_strength, max_strength):
    candidate_points = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    best_point = None
    best_improvement = -np.inf

    # Simulate enhancement at each candidate point and evaluate improvement
    for candidate in candidate_points:
        enhanced_signal = apply_signal_enhancement(grid_x, grid_y, grid_signal_full, candidate)
        improvement = np.sum((enhanced_signal > min_strength) & (enhanced_signal < max_strength))  # Improvement metric

        if improvement > best_improvement:
            best_improvement = improvement
            best_point = candidate

    return best_point

# Function to generate the plot and return it as base64-encoded string
def generate_plot_with_best_point(x_coords, y_coords, signal_strengths, min_strength, max_strength):
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
    
    # Find the best special point for signal enhancement
    best_point = find_best_special_point(x_coords, y_coords, signal_strengths, grid_x, grid_y, grid_signal_full, min_strength, max_strength)
    
    # Apply enhancement at the best point
    enhanced_signal_full = apply_signal_enhancement(grid_x, grid_y, grid_signal_full, best_point)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = ax.imshow(
        enhanced_signal_full.T, extent=(min_x, max_x, min_y, max_y),
        origin='lower', cmap='RdYlGn', alpha=0.8, aspect='equal'
    )
    cbar = fig.colorbar(heatmap)
    cbar.set_label('Signal Strength (dBm)')
    ax.plot(polygon_vertices[:, 0], polygon_vertices[:, 1], 'k-', linewidth=2, label='Polygon Boundary')
    ax.scatter(x_coords, y_coords, c='black', edgecolors='white', s=50, label='Measurement Points')
    
    # Plot the best point for signal enhancement
    if best_point is not None:
        ax.scatter([best_point[0]], [best_point[1]], c='blue', s=100, label='Best Signal Enhancer Location', marker='*')
    
    ax.set_title('Wi-Fi Signal Strength Heat Map with Best Enhancer Point')
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
        "best_point": best_point.tolist() if best_point is not None else None,
        "image_base64": img_base64
    }

@app.route('/generate-plot', methods=['POST'])
def generate_plot_api():
    data = request.json
    x_coords = np.array(data['x'])
    y_coords = np.array(data['y'])
    signal_strengths = np.array(data['strength'])
    
    # Get min_strength and max_strength from the request, with defaults
    min_strength = data.get('min_strength', -70)
    max_strength = data.get('max_strength', -30)
    
    result = generate_plot_with_best_point(x_coords, y_coords, signal_strengths, min_strength, max_strength)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
