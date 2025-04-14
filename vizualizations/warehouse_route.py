import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Tuple, Optional
import streamlit as st

class WarehouseRouter:
    """Class to handle warehouse routing visualization and path finding"""
    
    def __init__(self, warehouse_layout: np.ndarray, cell_coords: Dict, coords_to_cell: Dict):
        """
        Initialize the warehouse router
        
        Args:
            warehouse_layout: 2D numpy array representing the warehouse layout
            cell_coords: Dictionary mapping cell IDs to (row, col) coordinates
            coords_to_cell: Dictionary mapping (row, col) coordinates to cell IDs
        """
        self.layout = warehouse_layout
        self.rows, self.cols = warehouse_layout.shape
        self.cell_coords = cell_coords
        self.coords_to_cell = coords_to_cell
        
        # Create a graph representation of the warehouse for path finding
        self.graph = self._create_warehouse_graph()
    
    def _create_warehouse_graph(self) -> nx.Graph:
        """Create a graph representation of the warehouse for path finding"""
        G = nx.Graph()
        
        # Add nodes for all possible positions in the warehouse
        for i in range(self.rows):
            for j in range(self.cols):
                G.add_node((i, j))
        
        # Identify storage cells and aisles
        storage_cells = set(self.cell_coords.values())
        
        # Add edges for horizontal and vertical movements
        # where both cells are valid positions and not storage cells
        for i in range(self.rows):
            for j in range(self.cols):
                current_pos = (i, j)
                cell_value = str(self.layout[i, j]).strip()
                
                # Skip if the cell is an obstacle or invalid
                if cell_value in ['nan', 'None', ''] or pd.isna(self.layout[i, j]):
                    continue
                
                # Determine if this is a storage cell or aisle
                is_storage_cell = current_pos in storage_cells
                
                # If this is a storage cell, we can only connect to adjacent aisles
                # If this is an aisle, we can connect to other aisles or adjacent storage cells
                
                # Skip adding outgoing edges from storage cells - they will only have incoming edges
                # This ensures we never cross through a storage cell but can approach it
                if not is_storage_cell:
                    # This is an aisle, check all four directions
                    directions = [
                        (i, j+1),  # Right
                        (i, j-1),  # Left
                        (i+1, j),  # Down
                        (i-1, j)   # Up
                    ]
                    
                    for next_pos in directions:
                        next_i, next_j = next_pos
                        
                        # Check if the position is valid
                        if 0 <= next_i < self.rows and 0 <= next_j < self.cols:
                            next_value = str(self.layout[next_i, next_j]).strip()
                            
                            # Skip if the cell is an obstacle or invalid
                            if next_value in ['nan', 'None', ''] or pd.isna(self.layout[next_i, next_j]):
                                continue
                            
                            is_next_storage = next_pos in storage_cells
                            
                            # Connect if:
                            # 1. Next position is also an aisle (aisle to aisle movement)
                            # 2. Next position is a storage cell (approach to storage, but not crossing)
                            if not is_next_storage or (is_next_storage and not is_storage_cell):
                                G.add_edge(current_pos, next_pos)
        
        return G
    
    def get_picking_path(self, order_cells: List[str], start_point: Tuple[int, int] = None) -> List[Tuple[int, int]]:
        """
        Find the optimal path to pick items from the specified cells
        
        Args:
            order_cells: List of cell IDs to visit
            start_point: Starting coordinates (row, col), defaults to bottom left
            
        Returns:
            List of coordinates forming the path
        """
        # Default start point is bottom left
        if start_point is None:
            start_point = (self.rows - 1, 0)
        
        # Ensure all cells exist in the warehouse
        valid_cells = [cell for cell in order_cells if cell in self.cell_coords]
        
        # Get coordinates for all the cells
        coordinates = [self.cell_coords[cell] for cell in valid_cells]
        
        if not coordinates:
            return [start_point]  # Return just the start point if no valid cells
        
        # Use a simple greedy algorithm - visit the closest unvisited cell each time
        # For a real warehouse, a more sophisticated algorithm like TSP would be better
        path = [start_point]
        current_pos = start_point
        remaining_coords = coordinates.copy()
        
        while remaining_coords:
            # Find the closest remaining coordinate
            closest_coord = None
            shortest_path = None
            shortest_distance = float('inf')
            
            for coord in remaining_coords:
                try:
                    # Find the shortest path to this coordinate
                    coord_path = nx.shortest_path(self.graph, current_pos, coord)
                    path_length = len(coord_path) - 1  # Exclude the starting point
                    
                    if path_length < shortest_distance:
                        shortest_distance = path_length
                        shortest_path = coord_path
                        closest_coord = coord
                except nx.NetworkXNoPath:
                    # If there's no path, skip this coordinate
                    continue
            
            if closest_coord is None:
                # If we can't reach any more coordinates, we're done
                break
                
            # Add the path to the closest point (excluding the starting point)
            path.extend(shortest_path[1:])
            
            # Update current position and remove the visited coordinate
            current_pos = closest_coord
            remaining_coords.remove(closest_coord)
        
        # Return to the start point
        try:
            return_path = nx.shortest_path(self.graph, current_pos, start_point)
            path.extend(return_path[1:])  # Skip the first element as it's already in the path
        except nx.NetworkXNoPath:
            # If we can't get back to the start, just return what we have
            pass
            
        return path
        
    def visualize_route(self, path: List[Tuple[int, int]], order_cells: List[str]) -> go.Figure:
        """
        Create a visualization of the picking route
        
        Args:
            path: List of coordinates forming the path
            order_cells: List of cell IDs to highlight
            
        Returns:
            Plotly figure object
        """
        # Create figure
        fig = go.Figure()
        
        # Get dimensions
        rows, cols = self.layout.shape
        
        # Identify storage cells and aisles for coloring
        storage_cells = set(self.cell_coords.values())
        
        # Create an array representation of the warehouse for path planning
        # 0 = aisle, 1 = storage cell, 2 = wall/obstacle
        warehouse_map = np.zeros((rows, cols), dtype=int)
        
        # Mark storage cells and obstacles
        for i in range(rows):
            for j in range(cols):
                cell_value = str(self.layout[i, j]).strip()
                
                # Mark as obstacle/wall if the cell is invalid
                if cell_value in ['nan', 'None', ''] or pd.isna(self.layout[i, j]):
                    warehouse_map[i, j] = 2  # Wall/obstacle
                elif (i, j) in storage_cells:
                    warehouse_map[i, j] = 1  # Storage cell
                else:
                    warehouse_map[i, j] = 0  # Aisle
        
        # Add visualization for each cell in the warehouse
        for i in range(rows):
            for j in range(cols):
                cell_value = str(self.layout[i, j]).strip()
                
                # Skip empty cells
                if cell_value in ['nan', 'None', ''] or pd.isna(self.layout[i, j]):
                    continue
                    
                # Check if this is a storage cell
                is_storage_cell = (i, j) in self.coords_to_cell
                
                # Determine cell color
                if is_storage_cell:
                    cell_id = self.coords_to_cell[(i, j)]
                    
                    if cell_id in order_cells:
                        # Target cell
                        color = 'rgba(200, 230, 201, 0.8)'  # Light green
                        border_color = '#388E3C'  # Dark green
                        border_width = 3
                    else:
                        # Regular storage cell
                        color = 'rgba(241, 248, 233, 0.6)'  # Very light green
                        border_color = '#81C784'  # Medium green
                        border_width = 1
                else:
                    # Aisle
                    color = 'rgba(255, 255, 255, 0.3)'  # Nearly transparent white
                    border_color = 'rgba(200, 200, 200, 0.5)'  # Light gray
                    border_width = 0.5
                    
                # Add rectangle for cell
                fig.add_shape(
                    type="rect",
                    x0=j - 0.45,
                    y0=i - 0.45,
                    x1=j + 0.45,
                    y1=i + 0.45,
                    fillcolor=color,
                    line=dict(color=border_color, width=border_width),
                    layer='below'
                )
                
                # Add text label for cell ID if it's a storage cell
                if is_storage_cell:
                    font_color = 'black' if cell_id in order_cells else '#424242'
                    font_size = 12 if cell_id in order_cells else 10
                    font_weight = 'bold' if cell_id in order_cells else 'normal'
                    
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=cell_id,
                        showarrow=False,
                        font=dict(
                            color=font_color,
                            size=font_size,
                            family='Arial',
                            weight=font_weight
                        )
                    )
        
        # Helper function to check if a point is an aisle
        def is_aisle(point):
            row, col = point
            if 0 <= row < rows and 0 <= col < cols:
                return warehouse_map[row, col] == 0
            return False
        
        # Helper function to find a nearby aisle point
        def find_adjacent_aisle(row, col):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
            for dr, dc in directions:
                adj_row, adj_col = row + dr, col + dc
                if is_aisle((adj_row, adj_col)):
                    return (adj_row, adj_col)
            return None
        
        # Helper function to create a path between two aisle points without crossing storage cells
        def create_aisle_path(start, end):
            # Only create path if both start and end are aisles
            if not is_aisle(start) or not is_aisle(end):
                return []
                
            path_segments = []
            current = start
            
            # First move horizontally, then vertically to create a L-shaped path
            # This ensures we never cross a storage cell
            intermediate = (start[0], end[1])
            
            # Check if the intermediate point is an aisle
            if is_aisle(intermediate):
                # We can go directly in L-shape
                # Add horizontal segment
                if start[1] != end[1]:
                    for col in range(min(start[1], end[1]) + 1, max(start[1], end[1]) + 1):
                        test_point = (start[0], col)
                        # Only add if it's an aisle
                        if is_aisle(test_point):
                            path_segments.append(test_point)
                        else:
                            # We've hit a storage cell, need to go around
                            path_segments = []
                            break
                            
                # If we successfully added all horizontal segments
                if start[1] == end[1] or len(path_segments) > 0:
                    # Add vertical segment
                    for row in range(min(start[0], end[0]) + 1, max(start[0], end[0]) + 1):
                        test_point = (row, end[1])
                        # Only add if it's an aisle
                        if is_aisle(test_point):
                            path_segments.append(test_point)
                        else:
                            # We've hit a storage cell, need to go around
                            path_segments = []
                            break
            
            # If the L-shape path failed, try the reverse L-shape
            if not path_segments:
                intermediate = (end[0], start[1])
                
                # Check if the intermediate point is an aisle
                if is_aisle(intermediate):
                    # We can go in reverse L-shape
                    # Add vertical segment first
                    for row in range(min(start[0], end[0]) + 1, max(start[0], end[0]) + 1):
                        test_point = (row, start[1])
                        # Only add if it's an aisle
                        if is_aisle(test_point):
                            path_segments.append(test_point)
                        else:
                            # We've hit a storage cell, need to go around
                            path_segments = []
                            break
                    
                    # If vertical movement was successful
                    if len(path_segments) > 0:
                        # Add horizontal segment
                        for col in range(min(start[1], end[1]) + 1, max(start[1], end[1]) + 1):
                            test_point = (end[0], col)
                            # Only add if it's an aisle
                            if is_aisle(test_point):
                                path_segments.append(test_point)
                            else:
                                # We've hit a storage cell, need to go around
                                path_segments = []
                                break
            
            # If both L-shapes failed, try to find a path with three segments (Z-shape)
            if not path_segments:
                # Find a middle row that has clear aisles
                for mid_row in range(0, rows):
                    if mid_row == start[0] or mid_row == end[0]:
                        continue
                        
                    start_to_mid = create_aisle_path(start, (mid_row, start[1]))
                    if not start_to_mid:
                        continue
                        
                    mid_horizontal = []
                    for col in range(min(start[1], end[1]) + 1, max(start[1], end[1])):
                        test_point = (mid_row, col)
                        if is_aisle(test_point):
                            mid_horizontal.append(test_point)
                        else:
                            mid_horizontal = []
                            break
                            
                    if not mid_horizontal:
                        continue
                        
                    mid_to_end = create_aisle_path((mid_row, end[1]), end)
                    if not mid_to_end:
                        continue
                        
                    # We found a valid path
                    path_segments = start_to_mid + mid_horizontal + mid_to_end
                    break
            
            # If we still don't have a path, try another approach (e.g., A* pathfinding)
            # For simplicity, we'll just return an empty path here
            return path_segments
        
        # Generate path coordinates
        path_x = [coord[1] for coord in path]
        path_y = [coord[0] for coord in path]
        
        # Make sure the path has points to display
        # If path is too short or has only one point, add synthetic points
        if len(path) <= 1:
            # Start with a realistic starting point
            start_point = None
            
            # Find a good starting point - bottom-left aisle
            for i in range(rows-1, -1, -1):
                for j in range(cols):
                    if warehouse_map[i, j] == 0:  # Is aisle
                        start_point = (i, j)
                        break
                if start_point:
                    break
            
            # If no aisle found, use default
            if not start_point:
                start_point = (rows-1, 0)
            
            # If we have target cells, create a path between them
            if order_cells and len(order_cells) > 0:
                # Find adjacent aisle points for each target cell
                target_aisle_points = []
                
                for cell_id in order_cells:
                    if cell_id in self.cell_coords:
                        target_coord = self.cell_coords[cell_id]
                        # Find an adjacent aisle
                        adjacent_aisle = find_adjacent_aisle(target_coord[0], target_coord[1])
                        if adjacent_aisle:
                            target_aisle_points.append(adjacent_aisle)
                
                # Create a path that visits all targets by their adjacent aisles
                if target_aisle_points:
                    # Create a simple path that follows aisles
                    synthetic_path = [start_point]
                    current = start_point
                    
                    # Visit each target
                    for target in target_aisle_points:
                        # Create a path from current position to target
                        segments = create_aisle_path(current, target)
                        
                        # Add the segments to our path
                        if segments:
                            synthetic_path.extend(segments)
                        
                        # Add the target itself
                        synthetic_path.append(target)
                        
                        # Update current position
                        current = target
                    
                    # Return to start
                    end_segments = create_aisle_path(current, start_point)
                    if end_segments:
                        synthetic_path.extend(end_segments)
                    synthetic_path.append(start_point)
                    
                    # Update path coordinates
                    path = synthetic_path
                    path_x = [coord[1] for coord in path]
                    path_y = [coord[0] for coord in path]
        
        # If we still don't have a proper path, ensure we show something
        if len(path) <= 1:
            # Find perimeter aisles
            aisle_coords = []
            
            # Find top, right, bottom, left perimeter aisles
            perimeter_aisles = []
            
            # Top row aisles
            for j in range(cols):
                if warehouse_map[0, j] == 0:  # Is aisle
                    perimeter_aisles.append((0, j))
            
            # Right column aisles
            for i in range(rows):
                if warehouse_map[i, cols-1] == 0:  # Is aisle
                    perimeter_aisles.append((i, cols-1))
            
            # Bottom row aisles
            for j in range(cols-1, -1, -1):
                if warehouse_map[rows-1, j] == 0:  # Is aisle
                    perimeter_aisles.append((rows-1, j))
            
            # Left column aisles
            for i in range(rows-1, -1, -1):
                if warehouse_map[i, 0] == 0:  # Is aisle
                    perimeter_aisles.append((i, 0))
            
            # If we found perimeter aisles, use them for a path
            if perimeter_aisles:
                path = perimeter_aisles
            else:
                # Fallback to just our start point
                path = [start_point]
            
            path_x = [coord[1] for coord in path]
            path_y = [coord[0] for coord in path]
        
        # Remove the background blue line and keep only the main path
        # Add the main path line but make it invisible
        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='none',  # Make the line invisible
            showlegend=False
        ))
        
        # Mark the start and end points, ensuring they are visible even if they overlap
        if path_x and path_y:
            # Start point - large green circle with white border
            fig.add_trace(go.Scatter(
                x=[path_x[0]],
                y=[path_y[0]],
                mode='markers',
                marker=dict(
                    color='#2E7D32',  # Dark green
                    size=20,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                showlegend=False
            ))
            
            # If start and end are the same, offset the end marker slightly
            if path_x[0] == path_x[-1] and path_y[0] == path_y[-1]:
                end_x = path_x[-1] + 0.2
                end_y = path_y[-1] - 0.2
            else:
                end_x = path_x[-1]
                end_y = path_y[-1]
                
            # End point - large red circle with white border
            fig.add_trace(go.Scatter(
                x=[end_x],
                y=[end_y],
                mode='markers',
                marker=dict(
                    color='#C62828',  # Dark red
                    size=20,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                showlegend=False
            ))
        
        # Mark the sequence numbers for cells to visit
        waypoints_x = []
        waypoints_y = []
        waypoint_texts = []
        visit_order = 1
        visited_cells = set()
        
        # First, pre-populate waypoints from target cells
        for cell_id in order_cells:
            if cell_id in self.cell_coords:
                cell_coord = self.cell_coords[cell_id]
                waypoints_x.append(cell_coord[1])
                waypoints_y.append(cell_coord[0])
                waypoint_texts.append(str(visit_order))
                visit_order += 1
                visited_cells.add(cell_id)
        
        # Add the numbered waypoints (sequence)
        if waypoints_x:
            # Add black background markers first for better visibility
            fig.add_trace(go.Scatter(
                x=waypoints_x,
                y=waypoints_y,
                mode='markers',
                marker=dict(
                    size=22,
                    color='black',
                    symbol='diamond',
                ),
                showlegend=False
            ))
            
            # Add orange foreground markers with numbers
            fig.add_trace(go.Scatter(
                x=waypoints_x,
                y=waypoints_y,
                mode='markers+text',
                marker=dict(
                    size=18,
                    color='#E65100',  # Darker orange
                    symbol='diamond',
                    line=dict(color='white', width=1)
                ),
                text=waypoint_texts,
                textposition="middle center",
                textfont=dict(
                    color='white',
                    size=11,
                    family='Arial',
                    weight='bold'
                ),
                showlegend=False
            ))
        
        # Configure layout - more similar to warehouse_ref style
        fig.update_layout(
            title="Маршрут комплектации заказа",
            autosize=True,
            width=800,
            height=600,
            showlegend=False,  # Hide the legend
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                range=[-0.5, cols - 0.5],
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                range=[rows - 0.5, -0.5],  # Reverse Y-axis to match matrix coordinates
                showgrid=False,
                zeroline=False,
                visible=False,
                scaleanchor="x",
                scaleratio=1
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

def visualize_picking_route(warehouse, pallet):
    """
    Create a visualization of the picking route for a pallet
    
    Args:
        warehouse: Warehouse object with layout and coordinate data
        pallet: Pallet data structure containing orders to pick
        
    Returns:
        Plotly figure object
    """
    if warehouse.layout is None:
        st.error("План склада не загружен")
        return None
    
    # Create a router
    router = WarehouseRouter(
        warehouse_layout=warehouse.layout,
        cell_coords=warehouse.cell_coords,
        coords_to_cell=warehouse.coords_to_cell
    )
    
    # Get cells that need to be visited
    cells_to_visit = []
    sku_cells_map = {}  # Maps SKUs to their cell locations
    
    # First build a map of SKUs to cells
    for cell_id, contents in warehouse.cell_contents.items():
        sku = contents.get('sku')
        quantity = contents.get('quantity', 0)
        if sku and quantity > 0:
            if sku not in sku_cells_map:
                sku_cells_map[sku] = []
            sku_cells_map[sku].append(cell_id)
    
    # Now get the cells for each order item
    for order in pallet.get('orders', []):
        order_data = order.get('data', {})
        sku = order_data.get('SKU')
        
        if sku and sku in sku_cells_map:
            # Add all cells containing this SKU
            cells_to_visit.extend(sku_cells_map[sku])
    
    # Remove duplicates
    cells_to_visit = list(set(cells_to_visit))
    
    # Use a logical starting point - check for receiving area or use bottom left
    start_point = (warehouse.rows - 2, 1)  # Near bottom left, but one row up to be in an aisle
    
    # Get the picking path
    path = router.get_picking_path(cells_to_visit, start_point)
    
    # Create the visualization
    fig = router.visualize_route(path, cells_to_visit)
    
    return fig 