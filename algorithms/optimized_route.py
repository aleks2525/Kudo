import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional
import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from collections import defaultdict

class OptimizedWarehouseRouter:
    """
    Advanced warehouse router that optimizes picking routes based on item weights
    and ensures path follows only through aisles.
    """
    
    def __init__(self, warehouse_layout: np.ndarray, cell_coords: Dict, coords_to_cell: Dict, sku_data_path: str = None):
        """
        Initialize the optimized warehouse router
        
        Args:
            warehouse_layout: 2D numpy array representing the warehouse layout
            cell_coords: Dictionary mapping cell IDs to (row, col) coordinates
            coords_to_cell: Dictionary mapping (row, col) coordinates to cell IDs
            sku_data_path: Path to the CSV file containing SKU weight information
        """
        self.layout = warehouse_layout
        self.rows, self.cols = warehouse_layout.shape
        self.cell_coords = cell_coords
        self.coords_to_cell = coords_to_cell
        
        # Load SKU weight data if provided
        self.sku_weights = {}
        if sku_data_path and os.path.exists(sku_data_path):
            self._load_sku_weights(sku_data_path)
        
        # Create maps for path planning
        self.aisle_map = self._create_aisle_map()
        self.graph = self._create_warehouse_graph()
    
    def _load_sku_weights(self, filepath: str):
        """Load SKU weights from CSV file"""
        try:
            df = pd.read_csv(filepath)
            # Extract weight information - assume column names from sample data
            weight_col = 'Вес (г.)'
            sku_col = 'SKU'
            
            if weight_col in df.columns and sku_col in df.columns:
                for _, row in df.iterrows():
                    self.sku_weights[row[sku_col]] = float(row[weight_col])
            else:
                print(f"Warning: Expected columns not found in {filepath}")
        except Exception as e:
            print(f"Error loading SKU weights: {str(e)}")
    
    def _create_aisle_map(self) -> np.ndarray:
        """Create a binary map where 0 = aisle (valid path) and 1 = obstacle/storage cell"""
        aisle_map = np.ones((self.rows, self.cols), dtype=int)
        storage_cells = set(self.cell_coords.values())
        
        for i in range(self.rows):
            for j in range(self.cols):
                cell_value = str(self.layout[i, j]).strip()
                
                # Skip obstacles and invalid cells
                if cell_value in ['nan', 'None', ''] or pd.isna(self.layout[i, j]):
                    aisle_map[i, j] = 1  # Obstacle
                elif (i, j) in storage_cells:
                    aisle_map[i, j] = 1  # Storage cell - cannot be traversed
                else:
                    aisle_map[i, j] = 0  # Aisle - valid path
        
        return aisle_map
    
    def _create_warehouse_graph(self) -> nx.Graph:
        """Create a graph representation of the warehouse where edges only connect aisles"""
        G = nx.Graph()
        
        # Add nodes for all valid positions
        for i in range(self.rows):
            for j in range(self.cols):
                if self.aisle_map[i, j] == 0:  # Only add aisles
                    G.add_node((i, j))
        
        # Add edges between adjacent aisles
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        for i in range(self.rows):
            for j in range(self.cols):
                if self.aisle_map[i, j] == 0:  # Only connect from aisles
                    current = (i, j)
                    
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.rows and 0 <= nj < self.cols and self.aisle_map[ni, nj] == 0:
                            G.add_edge(current, (ni, nj), weight=1)
        
        return G
    
    def _get_nearest_aisle(self, cell_coord: Tuple[int, int]) -> Tuple[int, int]:
        """Find the nearest aisle point to a storage cell"""
        if self.aisle_map[cell_coord[0], cell_coord[1]] == 0:
            return cell_coord  # Already an aisle
        
        # Check adjacent cells in four directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        for di, dj in directions:
            ni, nj = cell_coord[0] + di, cell_coord[1] + dj
            if 0 <= ni < self.rows and 0 <= nj < self.cols and self.aisle_map[ni, nj] == 0:
                return (ni, nj)
        
        # If no adjacent aisle, use BFS to find nearest aisle
        visited = set()
        queue = [(cell_coord[0], cell_coord[1], 0)]  # Row, Col, Distance
        
        while queue:
            i, j, dist = queue.pop(0)
            
            if (i, j) in visited:
                continue
            
            visited.add((i, j))
            
            if self.aisle_map[i, j] == 0:
                return (i, j)
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols and (ni, nj) not in visited:
                    queue.append((ni, nj, dist + 1))
        
        # Fallback to original cell if no aisle found (shouldn't happen in a valid warehouse)
        return cell_coord
    
    def _find_optimal_order(self, cell_weights: Dict[str, float], start_point: Tuple[int, int]) -> List[str]:
        """Use OR-Tools to find an optimal order of cells to visit based on weight (heaviest first)"""
        if not cell_weights:
            return []
        
        # Get cell coordinates and ensure we have valid access points
        cells = list(cell_weights.keys())
        cell_coords = [self.cell_coords[cell] for cell in cells if cell in self.cell_coords]
        
        # Get nearest aisle points for each cell
        aisle_points = [self._get_nearest_aisle(coord) for coord in cell_coords]
        
        # Compute distance matrix between all points
        num_points = len(aisle_points) + 1  # +1 for start point
        distance_matrix = np.zeros((num_points, num_points), dtype=int)
        
        # Add start point to the beginning
        all_points = [start_point] + aisle_points
        
        # Fill distance matrix
        for i in range(num_points):
            for j in range(num_points):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    try:
                        # Use networkx to find shortest path length
                        path_length = nx.shortest_path_length(self.graph, all_points[i], all_points[j])
                        distance_matrix[i, j] = path_length
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        # If no path exists, use a large value
                        distance_matrix[i, j] = 1000000
        
        # Create OR-Tools routing model
        manager = pywrapcp.RoutingIndexManager(
            num_points, 
            1,  # Number of vehicles
            0   # Depot index (start point)
        )
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node, to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add weight-based constraints (heaviest items should be picked first)
        # We'll use penalties to prioritize heavier items early in the route
        for i, cell in enumerate(cells):
            node_index = manager.NodeToIndex(i + 1)  # +1 because start point is at index 0
            
            # Get the weight and convert to a penalty for late pickup
            # Higher weight = lower penalty for early pickup
            weight = cell_weights.get(cell, 0)
            max_weight = max(cell_weights.values()) if cell_weights else 1
            
            # Calculate penalty: items with higher weight should have higher penalty when picked later
            # This will encourage the solver to pick heavier items first
            penalty = int((weight / max_weight) * 10000)
            
            for position in range(1, num_points):
                routing.AddDisjunction([node_index], 0)  # Allow skipping impossible nodes
                routing.AddSoftSameVehicleConstraint([node_index], penalty * position)  # Пенальти за размещение тяжелых предметов в конце маршрута
        
        # Set up the search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 10  # Limit search time
        
        # Solve the routing problem
        solution = routing.SolveWithParameters(search_parameters)
        
        if not solution:
            # Fallback to a simple greedy approach - heaviest first
            return sorted(cells, key=lambda x: -cell_weights.get(x, 0))
        
        # Extract the optimized order of cells
        optimized_order = []
        index = routing.Start(0)
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:  # Skip the start point
                optimized_order.append(cells[node_index - 1])
            index = solution.Value(routing.NextVar(index))
        
        return optimized_order
    
    def _find_path_between_points(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a path between two points using A* pathfinding"""
        try:
            # First try using NetworkX for path finding between aisle points
            return nx.shortest_path(self.graph, start, end)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback to A* if NetworkX fails
            grid = Grid(matrix=1 - self.aisle_map)  # Convert to 1 = walkable, 0 = obstacle
            finder = AStarFinder()
            
            try:
                start_node = grid.node(start[1], start[0])
                end_node = grid.node(end[1], end[0])
                path, _ = finder.find_path(start_node, end_node, grid)
                
                # Convert path from (col, row) to (row, col) format
                return [(p[1], p[0]) for p in path]
            except Exception as e:
                print(f"A* pathfinding error: {str(e)}")
                return []
    
    def get_optimized_route(self, 
                           warehouse_data: Dict[str, Dict], 
                           order_skus: List[str], 
                           start_point: Tuple[int, int] = None) -> Dict:
        """
        Generate an optimized picking route for a given order, prioritizing heavier items first
        
        Args:
            warehouse_data: Dictionary mapping cell IDs to their contents
            order_skus: List of SKUs to be picked
            start_point: Starting coordinates (row, col), defaults to bottom left
            
        Returns:
            Dictionary containing the optimized route information
        """
        # Default start point is bottom left
        if start_point is None:
            start_point = (self.rows - 1, 0)
            
        # Ensure start point is an aisle
        start_point = self._get_nearest_aisle(start_point)
        
        # Find cells containing the ordered SKUs
        sku_to_cells = defaultdict(list)
        cell_weights = {}
        sku_cells = []
        
        for cell_id, contents in warehouse_data.items():
            sku = contents.get('sku')
            quantity = contents.get('quantity', 0)
            
            if sku in order_skus and quantity > 0 and cell_id in self.cell_coords:
                sku_to_cells[sku].append(cell_id)
                weight = self.sku_weights.get(sku, 0)
                cell_weights[cell_id] = weight
                sku_cells.append(cell_id)
        
        # Find optimal order of cells based on weight (heaviest first)
        optimized_cell_order = self._find_optimal_order(cell_weights, start_point)
        
        # If optimization didn't work, fall back to a simple ordering by weight
        if not optimized_cell_order:
            optimized_cell_order = sorted(sku_cells, key=lambda x: -cell_weights.get(x, 0))
        
        # If still no valid cells, return empty route
        if not optimized_cell_order:
            return {
                "path": [start_point],
                "cells": [],
                "order": [],
                "weights": {}
            }
        
        # Build a complete path through all the cells
        complete_path = [start_point]
        current_pos = start_point
        
        # Keep track of the actual order we visit cells
        cell_visit_order = []
        
        # Create a more complete path that approaches each cell
        for cell_id in optimized_cell_order:
            if cell_id in self.cell_coords:
                # Get actual cell coordinates
                cell_coord = self.cell_coords[cell_id]
                
                # Find nearest aisle point to this cell
                aisle_point_to_approach = self._get_nearest_aisle(cell_coord)
                
                # Find path from current position (an aisle point) to the aisle point near the target cell
                path_segment = self._find_path_between_points(current_pos, aisle_point_to_approach)
                
                if path_segment:
                    # Extend the complete path with the new segment (excluding the starting point of the segment)
                    if len(path_segment) > 1:
                        complete_path.extend(path_segment[1:])
                        # Update the current position to the end of the new segment
                        current_pos = path_segment[-1]
                        # Record that we have successfully routed to this cell
                        cell_visit_order.append(cell_id)
                    elif len(path_segment) == 1 and path_segment[0] == current_pos:
                        # Path is just the current point, means we are already at the target aisle point
                        # Still record the cell visit
                        cell_visit_order.append(cell_id)
                    # No else needed, if path_segment is just one point different from current, it means we moved one step
                    # In this case, current_pos is already updated implicitly by being the end of the segment
                    # and we should record the cell visit
                    elif len(path_segment) == 1 and path_segment[0] != current_pos:
                        complete_path.append(path_segment[0]) # Add the single step
                        current_pos = path_segment[0]
                        cell_visit_order.append(cell_id)

                else:
                    # If no path found, skip this cell for now
                    print(f"Warning: No path found from {current_pos} to {aisle_point_to_approach} for cell {cell_id}")
                    continue
        
        # Return to the start point
        path_to_start = self._find_path_between_points(current_pos, start_point)
        if path_to_start and len(path_to_start) > 1:
            complete_path.extend(path_to_start[1:])
        
        # Ensure the path doesn't have any duplicates while preserving order
        unique_path = []
        for point in complete_path:
            if not unique_path or point != unique_path[-1]:
                unique_path.append(point)
        
        # Create a lookup for the order SKUs by weight
        sku_weights = {sku: self.sku_weights.get(sku, 0) for sku in order_skus}
        
        return {
            "path": unique_path, # This path contains ONLY aisle coordinates
            "cells": cell_visit_order, # This contains the target CELL IDs in the order they were visited
            "order": optimized_cell_order, # The original optimized order from OR-Tools
            "weights": sku_weights
        }

def create_warehouse_router(warehouse, sku_data_path='data/sku_dimensions.csv'):
    """
    Create an optimized warehouse router instance
    
    Args:
        warehouse: Warehouse object with layout and coordinate data
        sku_data_path: Path to the SKU dimensions CSV file
    
    Returns:
        OptimizedWarehouseRouter instance
    """
    return OptimizedWarehouseRouter(
        warehouse_layout=warehouse.layout,
        cell_coords=warehouse.cell_coords,
        coords_to_cell=warehouse.coords_to_cell,
        sku_data_path=sku_data_path
    )

def get_optimized_picking_route(warehouse, pallet, sku_data_path='data/sku_dimensions.csv'):
    """
    Generate an optimized picking route for a pallet, prioritizing heavier items first
    
    Args:
        warehouse: Warehouse object with layout and coordinate data
        pallet: Pallet data structure containing orders to pick
        sku_data_path: Path to the SKU dimensions CSV file
    
    Returns:
        Dictionary containing the optimized route information and visualization data
    """
    if warehouse.layout is None:
        return None
    
    # Create the optimized router
    router = create_warehouse_router(warehouse, sku_data_path)
    
    # Get unique SKUs from the orders in the pallet
    order_skus = []
    for order in pallet.get('orders', []):
        order_data = order.get('data', {})
        sku = order_data.get('SKU')
        if sku:
            order_skus.append(sku)
    
    # Remove duplicates
    order_skus = list(set(order_skus))
    
    # Get the optimized route
    route_data = router.get_optimized_route(
        warehouse_data=warehouse.cell_contents,
        order_skus=order_skus,
        start_point=(warehouse.rows - 2, 1)  # Near bottom left in aisle
    )
    
    # Combine with the visualization data
    from vizualizations.warehouse_route import WarehouseRouter
    
    viz_router = WarehouseRouter(
        warehouse_layout=warehouse.layout,
        cell_coords=warehouse.cell_coords,
        coords_to_cell=warehouse.coords_to_cell
    )
    
    # Create visualization figure using the optimized path
    route_data['visualization'] = viz_router.visualize_route(
        path=route_data['path'],
        order_cells=route_data['cells']
    )
    
    return route_data 