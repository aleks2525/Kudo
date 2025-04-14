import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import math
from collections import defaultdict
import os

class PalletOptimizer:
    """
    Advanced pallet optimization algorithm that implements intelligent box stacking.
    
    Features:
    - Layer-based stacking with heavier items at the bottom
    - Optimal space utilization with minimal gaps
    - Cross-layer stacking for stability
    - Box rotation for better fit when allowed
    - Empty space in the center for wrapping
    - Respects maximum pallet height (2m by default)
    """
    
    def __init__(self, pallet_width=80, pallet_length=120, max_height=200, sku_data_path='data/sku_dimensions.csv'):
        """
        Initialize the pallet optimizer
        
        Args:
            pallet_width: Width of pallet in cm
            pallet_length: Length of pallet in cm
            max_height: Maximum height of loaded pallet in cm
            sku_data_path: Path to SKU dimensions CSV file
        """
        self.pallet_width = pallet_width
        self.pallet_length = pallet_length
        self.max_height = max_height
        self.sku_data = self._load_sku_data(sku_data_path)
        
    def _load_sku_data(self, sku_data_path: str) -> Dict[str, Dict]:
        """Load SKU dimensions and properties from CSV file"""
        try:
            # Create a sample data if file doesn't exist
            if not os.path.exists(sku_data_path):
                # Create sample data for testing
                print(f"SKU data file not found: {sku_data_path}. Creating sample data.")
                
                # Create the data directory if it doesn't exist
                os.makedirs(os.path.dirname(sku_data_path), exist_ok=True)
                
                # Create a sample SKU data file
                sample_data = pd.DataFrame({
                    'SKU': ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005'],
                    'Ширина (см.)': [30, 40, 25, 35, 20],
                    'Длина (см.)': [40, 50, 35, 45, 30],
                    'Высота (см.)': [25, 30, 20, 25, 15],
                    'Вес (г.)': [5000, 7000, 3000, 6000, 2000],
                    'Можно вращать': [True, False, True, True, True],
                    'Макс. нагрузка (г.)': [20000, 30000, 15000, 25000, 10000]
                })
                sample_data.to_csv(sku_data_path, index=False)
                
            df = pd.read_csv(sku_data_path)
            sku_data = {}
            
            for _, row in df.iterrows():
                sku = row['SKU']
                sku_data[sku] = {
                    'width': row['Ширина (см.)'],
                    'length': row['Длина (см.)'],
                    'height': row['Высота (см.)'],
                    'weight': row['Вес (г.)'],
                    'can_rotate': bool(row['Можно вращать']),  # Ensure boolean type
                    'max_load': row.get('Макс. нагрузка (г.)', 0)  # Maximum load this box can support
                }
            
            return sku_data
        except Exception as e:
            print(f"Error loading SKU data: {e}")
            
            # Return default data for testing
            return {
                'SKU001': {
                    'width': 30,
                    'length': 40,
                    'height': 25,
                    'weight': 5000,
                    'can_rotate': True,
                    'max_load': 20000
                },
                'SKU002': {
                    'width': 40,
                    'length': 50,
                    'height': 30,
                    'weight': 7000,
                    'can_rotate': False,
                    'max_load': 30000
                }
            }
    
    def optimize_pallet(self, orders: List[Dict], settings: Dict = None) -> Dict:
        """
        Generate optimized pallet layout based on orders
        
        Args:
            orders: List of order dictionaries containing SKU and quantity
            settings: Dictionary of optimization settings (optional)
            
        Returns:
            Dictionary containing the optimized pallet layout
        """
        if not orders or not self.sku_data:
            return {"status": "error", "message": "No orders or SKU data available"}
        
        # Default settings
        default_settings = {
            'center_empty': False,         # Leave center space empty for wrapping
            'prioritize_stability': True,  # Prioritize stability over perfect space utilization
            'max_weight': 1000 * 1000,     # Maximum weight in grams (1000 kg default)
            'stability_factor': 0.8,       # Minimum coverage required for stability (0-1)
            'max_overhang': 3.0,           # Maximum overhang in cm
            'max_height': 200.0            # Maximum pallet height in cm (2 meters)
        }
        
        # Override with provided settings
        if settings:
            default_settings.update(settings)
        
        settings = default_settings
        
        # BUGFIX: Ensure max_height is set and valid
        if 'max_height' not in settings or not settings['max_height']:
            settings['max_height'] = 200.0
        elif float(settings['max_height']) > 200.0:
            print("WARNING: Requested height exceeds 2 meters, capping at 200 cm")
            settings['max_height'] = 200.0
        
        # Apply safety margin to prevent floating point errors
        settings['max_height'] = float(settings['max_height']) * 0.999
        
        # Check if this is a recursive call by looking for 'is_internal_call' flag
        is_internal_call = settings.pop('is_internal_call', False)
        
        # If this is not an internal call, we're in the main function that handles multiple pallets
        if not is_internal_call:
            # Prepare boxes from orders and calculate total volume and weight
            all_boxes = []
            total_volume = 0
            total_weight = 0
            
            for order in orders:
                sku = order.get('data', {}).get('SKU')
                quantity = order.get('data', {}).get('Количество', 1)
                order_id = order.get('id')
                
                if sku in self.sku_data:
                    sku_info = self.sku_data[sku]
                    for i in range(int(quantity)):
                        box = {
                            'sku': sku,
                            'width': sku_info['width'],
                            'length': sku_info['length'],
                            'height': sku_info['height'],
                            'weight': sku_info['weight'],
                            'can_rotate': sku_info['can_rotate'],
                            'max_load': sku_info['max_load'],
                            'id': f"{sku}_{i}",
                            'order_id': order_id,  # Add the order ID to track which order this box belongs to
                            'box_index': i,  # Add box index within the order
                            'rotated': False,
                            'volume': sku_info['width'] * sku_info['length'] * sku_info['height']
                        }
                        all_boxes.append(box)
                        total_volume += box['volume']
                        total_weight += box['weight']
            
            # Calculate number of pallets needed
            pallet_base_area = self.pallet_width * self.pallet_length
            pallet_max_volume = pallet_base_area * settings['max_height']
            
            # Account for practical packing efficiency (approximately 75-85%)
            practical_efficiency = 0.75
            adjusted_pallet_volume = pallet_max_volume * practical_efficiency
            
            # Calculate number of pallets needed based on volume
            pallets_needed_by_volume = max(1, math.ceil(total_volume / adjusted_pallet_volume))
            
            # Calculate number of pallets needed based on weight
            pallets_needed_by_weight = max(1, math.ceil(total_weight / settings['max_weight']))
            
            # Use the greater of the two values
            pallets_needed = max(pallets_needed_by_volume, pallets_needed_by_weight)
            
            # Sort boxes by weight (heavier first) for optimal layered distribution
            all_boxes.sort(key=lambda x: (-x['weight'], -(x['width'] * x['length'])))
            
            # Distribute boxes into pallets with better weight distribution
            pallets = []
            remaining_boxes = all_boxes.copy()
            
            # Create box groups for better distribution
            heavy_boxes = [box for box in all_boxes if box['weight'] > 3000]
            medium_boxes = [box for box in all_boxes if 1000 <= box['weight'] <= 3000]
            light_boxes = [box for box in all_boxes if box['weight'] < 1000]
            
            # Initialize pallets with empty list of boxes
            pallet_boxes = [[] for _ in range(pallets_needed)]
            pallet_weights = [0] * pallets_needed
            pallet_volumes = [0] * pallets_needed
            
            # First, distribute heavy boxes evenly with consideration for volume and height
            for box in heavy_boxes:
                # Find the best pallet for this box - considering both weight and volume constraints
                best_pallet_idx = self._find_best_pallet_for_box(
                    box, pallet_weights, pallet_volumes, pallet_boxes, 
                    settings['max_weight'], pallet_base_area * settings['max_height'] * practical_efficiency
                )
                
                if best_pallet_idx is not None:
                    pallet_boxes[best_pallet_idx].append(box)
                    pallet_weights[best_pallet_idx] += box['weight']
                    pallet_volumes[best_pallet_idx] += box['volume']
                else:
                    # If no suitable pallet found, create a new one
                    pallets_needed += 1
                    pallet_boxes.append([box])
                    pallet_weights.append(box['weight'])
                    pallet_volumes.append(box['volume'])
            
            # Then distribute medium boxes with consideration for volume and height
            for box in medium_boxes:
                best_pallet_idx = self._find_best_pallet_for_box(
                    box, pallet_weights, pallet_volumes, pallet_boxes, 
                    settings['max_weight'], pallet_base_area * settings['max_height'] * practical_efficiency
                )
                
                if best_pallet_idx is not None:
                    pallet_boxes[best_pallet_idx].append(box)
                    pallet_weights[best_pallet_idx] += box['weight']
                    pallet_volumes[best_pallet_idx] += box['volume']
                else:
                    # If no suitable pallet found, create a new one
                    pallets_needed += 1
                    pallet_boxes.append([box])
                    pallet_weights.append(box['weight'])
                    pallet_volumes.append(box['volume'])
            
            # Finally distribute light boxes with consideration for volume and height
            for box in light_boxes:
                best_pallet_idx = self._find_best_pallet_for_box(
                    box, pallet_weights, pallet_volumes, pallet_boxes, 
                    settings['max_weight'], pallet_base_area * settings['max_height'] * practical_efficiency
                )
                
                if best_pallet_idx is not None:
                    pallet_boxes[best_pallet_idx].append(box)
                    pallet_weights[best_pallet_idx] += box['weight']
                    pallet_volumes[best_pallet_idx] += box['volume']
                else:
                    # If no suitable pallet found, create a new one
                    pallets_needed += 1
                    pallet_boxes.append([box])
                    pallet_weights.append(box['weight'])
                    pallet_volumes.append(box['volume'])
            
            # Now optimize each pallet individually
            for pallet_num in range(pallets_needed):
                if not pallet_boxes[pallet_num]:
                    continue
                    
                # Get the boxes for this pallet
                current_boxes = pallet_boxes[pallet_num]
                
                # Sort boxes by weight for layer-based packing (heavy at bottom)
                current_boxes.sort(key=lambda x: (-x['weight'], -(x['width'] * x['length'])))
                
                # Set up settings specific to this pallet
                pallet_settings = settings.copy()
                pallet_settings['max_overhang'] = settings.get('max_overhang', 3.0)
                pallet_settings['is_internal_call'] = True
                
                # Create optimized pallet
                pallet_result = self._optimize_single_pallet(current_boxes, pallet_settings)
                pallets.append(pallet_result)
                
                # Remove the used boxes from remaining_boxes
                used_box_ids = [box['id'] for layer in pallet_result.get('layers', []) for box in layer.get('boxes', [])]
                remaining_boxes = [box for box in remaining_boxes if box['id'] not in used_box_ids]
            
            # If we have pallets with remaining space but boxes still left, try to add more
            if remaining_boxes:
                # Sort remaining boxes by weight
                remaining_boxes.sort(key=lambda x: (-x['weight'], -(x['width'] * x['length'])))
                
                # Make multiple passes to try to place boxes
                for attempt in range(3):  # Try up to 3 times with different strategies
                    if not remaining_boxes:
                        break  # All boxes placed, we're done
                        
                    # Adjust sort order based on attempt
                    if attempt == 0:
                        # First pass: Sort by weight (heavier first)
                        remaining_boxes.sort(key=lambda x: (-x['weight'], -(x['width'] * x['length'])))
                    elif attempt == 1:
                        # Second pass: Sort by volume (smaller first)
                        remaining_boxes.sort(key=lambda x: (x['width'] * x['length'] * x['height']))
                    else:
                        # Third pass: Sort by base area (smaller first)
                        remaining_boxes.sort(key=lambda x: (x['width'] * x['length']))
                    
                    # Try to fit remaining boxes into pallets with available space
                    boxes_processed = []
                    for box_idx, box in enumerate(remaining_boxes):
                        for pallet_idx, pallet in enumerate(pallets):
                            # Check if adding this box would exceed max height or weight
                            if (pallet['total_height'] + box['height'] <= settings['max_height'] and
                                pallet['total_weight'] + box['weight'] <= settings['max_weight']):
                                
                                # Try to add to this pallet
                                pallet_settings = settings.copy()
                                pallet_settings['is_internal_call'] = True
                                
                                # Create a temporary pallet with just this box
                                temp_result = self._optimize_single_pallet([box], pallet_settings)
                                
                                if temp_result.get('remaining_boxes', 0) == 0:
                                    # Box fits, add it to the pallet
                                    for layer in temp_result.get('layers', []):
                                        for temp_box in layer.get('boxes', []):
                                            # Add box to appropriate layer based on height
                                            added = False
                                            for existing_layer in pallet.get('layers', []):
                                                if abs(existing_layer['height'] - layer['height']) < 5:
                                                    existing_layer['boxes'].append(temp_box)
                                                    existing_layer['weight'] += temp_box['weight']
                                                    
                                                    # Update grid
                                                    for pos_idx, pos in enumerate(layer.get('positions', [])):
                                                        if pos_idx < len(layer.get('boxes', [])) and layer['boxes'][pos_idx] == temp_box:
                                                            x, y, w, l = pos
                                                            existing_layer['positions'].append((x, y, w, l))
                                                            # Update grid at this position if within bounds
                                                            try:
                                                                existing_layer['grid'][y:y+l, x:x+w] = 1
                                                            except:
                                                                pass  # Ignore errors from invalid indices
                                                    
                                                    added = True
                                                    break
                                            
                                            if not added:
                                                # Add as new layer
                                                pallet['layers'].append(layer)
                                    
                                    # Update pallet stats
                                    pallet['total_weight'] += box['weight']
                                    pallet['total_height'] = max(pallet['total_height'], 
                                                               pallet['total_height'] + box['height'] if not added else 0)
                                    pallet['box_count'] += 1
                                    
                                    # Mark box as processed
                                    boxes_processed.append(box_idx)
                                    break
                    
                    # Remove processed boxes
                    remaining_boxes = [box for idx, box in enumerate(remaining_boxes) if idx not in boxes_processed]
                    
                    # If no boxes were placed in this pass, continue to next strategy
                    if not boxes_processed:
                        continue
                
                # If we still have remaining boxes after all attempts, consider creating a new pallet
                if remaining_boxes and settings.get('allow_extra_pallets', True):
                    # Create a new pallet with the remaining boxes
                    pallet_settings = settings.copy()
                    pallet_settings['is_internal_call'] = True
                    new_pallet = self._optimize_single_pallet(remaining_boxes, pallet_settings)
                    
                    # Add to pallets list
                    if new_pallet and 'layers' in new_pallet and new_pallet['layers']:
                        pallets.append(new_pallet)
                        remaining_boxes = []  # All boxes placed
            
            # Prepare the final result
            result = {
                'total_pallets': len(pallets),
                'total_volume': total_volume,
                'total_weight': total_weight,
                'pallets': pallets,
                'remaining_boxes': len(remaining_boxes) if remaining_boxes else 0
            }
            
            # If there's a first pallet, use its data for compatibility with the UI code
            if pallets:
                result.update(pallets[0])
                result['additional_pallets'] = pallets[1:] if len(pallets) > 1 else []
            
            return result
        else:
            # This should never be reached now that we're using _optimize_single_pallet
            return {"status": "error", "message": "Internal error: recursive call detected"}
    
    def _optimize_single_pallet(self, boxes: List[Dict], settings: Dict) -> Dict:
        """
        Optimize a single pallet layout
        
        Args:
            boxes: List of box dictionaries
            settings: Dictionary of optimization settings
            
        Returns:
            Dictionary containing the optimized single pallet layout
        """
        # Initialize pallet
        pallet = self._create_empty_pallet()
        
        # Get max height from settings
        max_height = settings.get('max_height', 200.0)
        
        # BUGFIX: Apply safety margin to prevent floating point errors
        max_height = max_height * 0.999  # Reduce by 0.1% as safety margin
        
        # Sort boxes by weight (heavier first) and then by base area for optimal layered distribution
        sorted_boxes = sorted(boxes, key=lambda x: (-x['weight'], -(x['width'] * x['length'])))
        
        # Check if any single box exceeds maximum height
        # If so, mark it as impossible to pack and continue with other boxes
        boxes_to_pack = []
        oversized_boxes = []
        for box in sorted_boxes:
            if float(box['height']) > float(max_height):
                oversized_boxes.append(box)
            else:
                boxes_to_pack.append(box)
        
        # Organize boxes into layers
        layers, remaining_boxes, total_height = self._create_layers(boxes_to_pack, settings)
        
        # If we have remaining boxes, try to place them by rotating
        if remaining_boxes and total_height < max_height:
            # Try rotating boxes that can be rotated to fit in existing or new layers
            rotated_layers, still_remaining, new_height = self._optimize_with_rotation(remaining_boxes, layers, total_height, settings)
            layers = rotated_layers
            remaining_boxes = still_remaining
            total_height = new_height
            
        # Add oversized boxes to remaining_boxes
        remaining_boxes.extend(oversized_boxes)
        
        # Organize cross-layer stacking for stability
        if settings['prioritize_stability'] and len(layers) > 1:
            layers = self._improve_stability(layers, settings)
        
        # Update pallet with the layers
        pallet['layers'] = layers
        pallet['total_height'] = total_height
        pallet['total_weight'] = sum(box['weight'] for layer in layers for box in layer['boxes'])
        pallet['box_count'] = sum(len(layer['boxes']) for layer in layers)
        pallet['utilization'] = self._calculate_utilization(layers)
        pallet['stability_score'] = self._calculate_stability(layers)
        
        # BUGFIX: Final hard check to ensure height doesn't exceed max
        # If somehow we're still over the max height, remove layers from the top
        if float(pallet['total_height']) > float(max_height):
            print(f"WARNING: Height exceeded after optimization: {pallet['total_height']} > {max_height}")
            while layers and float(pallet['total_height']) > float(max_height):
                removed_layer = layers.pop()
                pallet['total_height'] -= removed_layer['height']
                for box in removed_layer['boxes']:
                    remaining_boxes.append(box)
                print(f"Removed layer with height {removed_layer['height']}, new total height: {pallet['total_height']}")
        
        # Add remaining boxes info if any
        if remaining_boxes:
            pallet['remaining_boxes'] = len(remaining_boxes)
            pallet['remaining_weight'] = sum(box['weight'] for box in remaining_boxes)
        else:
            pallet['remaining_boxes'] = 0
            pallet['remaining_weight'] = 0
        
        return pallet
    
    def _create_empty_pallet(self) -> Dict:
        """Initialize an empty pallet structure"""
        return {
            'width': self.pallet_width,
            'length': self.pallet_length,
            'max_height': self.max_height,
            'layers': [],
            'total_height': 0,
            'total_weight': 0,
            'box_count': 0,
            'utilization': 0,
            'stability_score': 0
        }
    
    def _create_layers(self, boxes: List[Dict], settings: Dict) -> Tuple[List[Dict], List[Dict], float]:
        """
        Create efficient layers for the boxes with simplified algorithm for better performance
        
        Args:
            boxes: List of box dictionaries
            settings: Optimization settings
            
        Returns:
            Tuple of (layers, remaining_boxes, total_height)
        """
        layers = []
        remaining_boxes = []
        total_height = 0
        current_boxes = boxes.copy()
        
        # If no boxes, return early
        if not current_boxes:
            return layers, remaining_boxes, total_height
        
        # Get max height from settings (default 200 cm / 2 meters)    
        max_height = settings.get('max_height', 200.0)
        
        # BUGFIX: Apply safety margin to prevent floating point errors
        # Reduce max height by small amount to prevent floating point comparison issues
        max_height = max_height * 0.999  # Reduce by 0.1% as safety margin
        
        # Sort boxes by weight and base area for efficient layering
        current_boxes.sort(key=lambda x: (-x['weight'], -(x['width'] * x['length']), x['height']))
        
        # Keep creating layers until we run out of boxes or reach max height
        while current_boxes and total_height < max_height:
            # Find a box that fits within remaining height
            layer_box_idx = 0
            found_fitting_box = False
            
            for i, box in enumerate(current_boxes):
                # BUGFIX: Stricter height check with explicit float conversion
                if float(total_height) + float(box['height']) <= float(max_height):
                    layer_box_idx = i
                    found_fitting_box = True
                    break
                
            # If no box fits, we're done
            if not found_fitting_box:
                remaining_boxes.extend(current_boxes)
                break
                
            # Take the box for the current layer
            current_box = current_boxes.pop(layer_box_idx)
            
            # Double-check height constraint
            if float(total_height) + float(current_box['height']) > float(max_height):
                # This should never happen with our checks above, but just for safety
                remaining_boxes.append(current_box)
                continue
            
            # Initialize a new layer with the height of the current box
            layer_height = current_box['height']
            current_layer = {
                'height': layer_height,
                'max_load': current_box['max_load'],
                'boxes': [current_box],
                'positions': [(0, 0, current_box['width'], current_box['length'])],  # x, y, width, length
                'grid': np.zeros((self.pallet_length, self.pallet_width)),  # 0 = empty, 1 = filled
                'weight': current_box['weight']
            }
            
            # Mark the position of the first box in the grid
            x, y = 0, 0
            w, l = current_box['width'], current_box['length']
            current_layer['grid'][y:y+l, x:x+w] = 1
            
            # Try to add more boxes to the current layer (single pass)
            # First, identify boxes that have same or similar height
            same_height_boxes = []
            similar_height_boxes = []
            other_boxes = []
            
            for i, box in enumerate(current_boxes[:]):
                # Skip if this box would exceed the max weight
                if current_layer['weight'] + box['weight'] > settings['max_weight']:
                    continue
                    
                if box['height'] == layer_height:
                    same_height_boxes.append((i, box))
                elif abs(box['height'] - layer_height) <= 5:
                    similar_height_boxes.append((i, box))
                else:
                    other_boxes.append((i, box))
            
            # Process boxes in order of priority: same height, similar height, other
            for priority_boxes in [same_height_boxes, similar_height_boxes, other_boxes]:
                # Sort smaller boxes first for better gap filling
                priority_boxes.sort(key=lambda x: (x[1]['width'] * x[1]['length']))
                
                # Try each box
                for box_idx, box in priority_boxes:
                    # Find a position for this box
                    position = self._find_simple_position(box, current_layer, settings)
                    
                    if position:
                        # Position found, add the box to the layer
                        x, y, w, l, rotated = position
                        box['rotated'] = rotated
                        current_layer['boxes'].append(box)
                        current_layer['positions'].append((x, y, w, l))
                        current_layer['grid'][y:y+l, x:x+w] = 1
                        current_layer['weight'] += box['weight']
                        
                        # Remove the box from current_boxes
                        try:
                            idx = current_boxes.index(box)
                            current_boxes.pop(idx)
                        except ValueError:
                            pass  # Box might already be removed
            
            # Add the layer if it has boxes
            if current_layer['boxes']:
                # Try to fill complex gaps in the layer if enabled
                if settings.get('fill_complex_gaps', False):
                    # Attempt to fill complex gaps and update remaining boxes
                    current_boxes = self._fill_complex_gaps(current_layer, current_boxes, settings)
                
                layers.append(current_layer)
                total_height += current_layer['height']
        
        # If we still have boxes that don't fit, add them to remaining_boxes
        remaining_boxes.extend(current_boxes)
        
        # Final check to ensure we don't exceed max height
        if total_height > max_height:
            # Remove layers from the top until we're under max_height
            while layers and total_height > max_height:
                removed_layer = layers.pop()
                total_height -= removed_layer['height']
                remaining_boxes.extend(removed_layer['boxes'])
        
        return layers, remaining_boxes, total_height
    
    def _find_simple_position(self, box: Dict, layer: Dict, settings: Dict) -> Optional[Tuple[int, int, int, int, bool]]:
        """
        Simplified version of position finding for better performance
        
        Args:
            box: Box to place
            layer: Current layer data
            settings: Optimization settings
            
        Returns:
            Tuple of (x, y, width, length, rotated) or None if no position found
        """
        grid = layer['grid']
        max_overhang = settings.get('max_overhang', 3.0)
        
        # Try normal orientation
        width, length = box['width'], box['length']
        
        # First try corners and edges as they're most efficient
        positions_to_try = [
            # Corners
            (0, 0),  # Top-left
            (0, self.pallet_length - length),  # Bottom-left
            (self.pallet_width - width, 0),  # Top-right
            (self.pallet_width - width, self.pallet_length - length),  # Bottom-right
            
            # Edges
            (0, self.pallet_length // 2 - length // 2),  # Left middle
            (self.pallet_width - width, self.pallet_length // 2 - length // 2),  # Right middle
            (self.pallet_width // 2 - width // 2, 0),  # Top middle
            (self.pallet_width // 2 - width // 2, self.pallet_length - length)  # Bottom middle
        ]
        
        # Add some spots along the edges
        for i in range(1, 4):
            positions_to_try.append((i * self.pallet_width // 5, 0))  # Top edge
            positions_to_try.append((i * self.pallet_width // 5, self.pallet_length - length))  # Bottom edge
            positions_to_try.append((0, i * self.pallet_length // 5))  # Left edge
            positions_to_try.append((self.pallet_width - width, i * self.pallet_length // 5))  # Right edge
        
        # Check each position
        for x, y in positions_to_try:
            # Skip invalid positions and positions with overlaps
            if x < -max_overhang or y < -max_overhang or x + width > self.pallet_width + max_overhang or y + length > self.pallet_length + max_overhang:
                continue
                
            # Calculate overhang
            left_overhang = abs(min(0, x))
            right_overhang = abs(max(0, (x + width) - self.pallet_width))
            top_overhang = abs(min(0, y))
            bottom_overhang = abs(max(0, (y + length) - self.pallet_length))
            
            total_overhang = left_overhang + right_overhang + top_overhang + bottom_overhang
            
            # Skip if overhang exceeds allowed maximum
            if total_overhang > max_overhang:
                continue
            
            # Check valid grid area
            valid_x = max(0, x)
            valid_y = max(0, y)
            valid_width = min(width, self.pallet_width - valid_x)
            valid_length = min(length, self.pallet_length - valid_y)
            
            if valid_width <= 0 or valid_length <= 0:
                continue
            
            # Check for overlap
            try:
                if np.any(grid[valid_y:valid_y+valid_length, valid_x:valid_x+valid_width] > 0):
                    continue
            except IndexError:
                continue
                
            # Found valid position
            return (x, y, width, length, False)
        
        # If not found, try simple grid search with bigger steps to save time
        step_size = 5  # Check every 5cm instead of every 1cm
        for y in range(0, self.pallet_length - length + 1, step_size):
            for x in range(0, self.pallet_width - width + 1, step_size):
                # Skip center area if center_empty is True
                if settings['center_empty'] and self._is_center_position(x, y, width, length):
                    continue
                
                # Check for overlap
                try:
                    if np.any(grid[y:y+length, x:x+width] > 0):
                        continue
                except IndexError:
                    continue
                
                # Found valid position
                return (x, y, width, length, False)
        
        # Try rotation if allowed and size differs
        if box['can_rotate'] and width != length:
            width, length = box['length'], box['width']  # Swap dimensions
            
            # Check corners and edges with rotated dimensions
            for x, y in positions_to_try:
                # Skip invalid positions and positions with overlaps
                if x < -max_overhang or y < -max_overhang or x + width > self.pallet_width + max_overhang or y + length > self.pallet_length + max_overhang:
                    continue
                    
                # Calculate overhang
                left_overhang = abs(min(0, x))
                right_overhang = abs(max(0, (x + width) - self.pallet_width))
                top_overhang = abs(min(0, y))
                bottom_overhang = abs(max(0, (y + length) - self.pallet_length))
                
                total_overhang = left_overhang + right_overhang + top_overhang + bottom_overhang
                
                # Skip if overhang exceeds allowed maximum
                if total_overhang > max_overhang:
                    continue
                
                # Check valid grid area
                valid_x = max(0, x)
                valid_y = max(0, y)
                valid_width = min(width, self.pallet_width - valid_x)
                valid_length = min(length, self.pallet_length - valid_y)
                
                if valid_width <= 0 or valid_length <= 0:
                    continue
                
                # Check for overlap
                try:
                    if np.any(grid[valid_y:valid_y+valid_length, valid_x:valid_x+valid_width] > 0):
                        continue
                except IndexError:
                    continue
                    
                # Found valid position with rotation
                return (x, y, width, length, True)
            
            # Try simple grid search with rotated dimensions
            for y in range(0, self.pallet_length - length + 1, step_size):
                for x in range(0, self.pallet_width - width + 1, step_size):
                    # Skip center area if center_empty is True
                    if settings['center_empty'] and self._is_center_position(x, y, width, length):
                        continue
                    
                    # Check for overlap
                    try:
                        if np.any(grid[y:y+length, x:x+width] > 0):
                            continue
                    except IndexError:
                        continue
                    
                    # Found valid position with rotation
                    return (x, y, width, length, True)
        
        # No position found
        return None
    
    def _find_best_position(self, box: Dict, layer: Dict, settings: Dict) -> Optional[Tuple[int, int, int, int, bool]]:
        """
        Использую более простую версию функции _find_simple_position для устранения проблем с синтаксисом
        """
        return self._find_simple_position(box, layer, settings)
    
    def _calculate_improved_position_score(self, x: int, y: int, width: int, length: int, 
                                     grid: np.ndarray, settings: Dict) -> float:
        """
        Calculate an improved score for a position with better heuristics for space utilization
        
        Args:
            x, y: Position coordinates
            width, length: Box dimensions
            grid: Current layer grid
            settings: Optimization settings
            
        Returns:
            Position score (higher is better)
        """
        score = 0
        
        # Bonus for corner positions (highest priority)
        if ((x == 0 or x + width == self.pallet_width) and 
            (y == 0 or y + length == self.pallet_length)):
            score += 10  # Corner position bonus
        
        # Bonus for edge alignment (high priority)
        if x == 0 or x + width == self.pallet_width:
            score += 5  # Align with left/right edge
        if y == 0 or y + length == self.pallet_length:
            score += 5  # Align with top/bottom edge
        
        # Bonus for adjacency to other boxes (better gap filling)
        adjacency_count = 0
        contact_surface = 0
        
        # Check all four sides for adjacency
        valid_x = max(0, x)
        valid_y = max(0, y)
        valid_width = min(width, self.pallet_width - valid_x)
        valid_length = min(length, self.pallet_length - valid_y)
        
        # Check left side
        if valid_x > 0:
            left_contact = np.sum(grid[valid_y:valid_y+valid_length, valid_x-1])
            if left_contact > 0:
                adjacency_count += 1
                contact_surface += left_contact
        
        # Check right side
        if valid_x + valid_width < self.pallet_width:
            right_contact = np.sum(grid[valid_y:valid_y+valid_length, valid_x+valid_width])
            if right_contact > 0:
                adjacency_count += 1
                contact_surface += right_contact
        
        # Check top side
        if valid_y > 0:
            top_contact = np.sum(grid[valid_y-1, valid_x:valid_x+valid_width])
            if top_contact > 0:
                adjacency_count += 1
                contact_surface += top_contact
        
        # Check bottom side
        if valid_y + valid_length < self.pallet_length:
            bottom_contact = np.sum(grid[valid_y+valid_length, valid_x:valid_x+valid_width])
            if bottom_contact > 0:
                adjacency_count += 1
                contact_surface += bottom_contact
        
        # Increased bonus for adjacency
        score += adjacency_count * 6
        
        # Bonus for more contact surface (better stability)
        score += min(contact_surface / 10, 5)  # Cap at 5 to avoid dominating the score
        
        # Penalty for center position if center_empty is True
        if settings['center_empty'] and self._is_close_to_center(x, y, width, length):
            score -= 15
        
        # Penalty for complex shapes that might result in fragmented space
        # This helps avoid creating small, hard-to-fill gaps
        shape_complexity = 0
        if x > 0 and y > 0:
            # Check for potential L-shaped gaps
            if grid[y-1, x-1] == 0 and ((grid[y-1, x] > 0 and grid[y, x-1] > 0) or 
                                        (grid[y-1, x] == 0 and grid[y, x-1] == 0)):
                shape_complexity += 1
        
        score -= shape_complexity * 2
        
        return score
    
    def _is_valid_position(self, grid: np.ndarray, x: int, y: int, width: int, length: int) -> bool:
        """Check if a position is valid (no overlap with existing boxes)"""
        if x + width > self.pallet_width or y + length > self.pallet_length:
            return False
            
        # Check if the position overlaps with any existing box
        if np.any(grid[y:y+length, x:x+width] > 0):
            return False
            
        return True
    
    def _is_center_position(self, x: int, y: int, width: int, length: int) -> bool:
        """Check if a position covers the center of the pallet"""
        center_x = self.pallet_width // 2
        center_y = self.pallet_length // 2
        
        # Check if the box overlaps with the center area (inner 25% of pallet)
        center_width = self.pallet_width // 4
        center_length = self.pallet_length // 4
        
        center_left = center_x - center_width // 2
        center_right = center_x + center_width // 2
        center_top = center_y - center_length // 2
        center_bottom = center_y + center_length // 2
        
        # Check for overlap with center area
        return (x < center_right and x + width > center_left and 
                y < center_bottom and y + length > center_top)
    
    def _calculate_position_score(self, x: int, y: int, width: int, length: int, 
                                 grid: np.ndarray, settings: Dict) -> float:
        """
        Calculate a score for a position (higher is better)
        Scoring factors:
        - Edge alignment (boxes touching the edge of the pallet)
        - Adjacency to other boxes
        - Avoid center if center_empty is True
        """
        score = 0
        
        # Edge alignment
        if x == 0 or x + width == self.pallet_width:
            score += 5  # Bonus for aligning with left/right edge
        if y == 0 or y + length == self.pallet_length:
            score += 5  # Bonus for aligning with top/bottom edge
        
        # Adjacency to other boxes - check all four sides
        # Left side
        if x > 0 and np.any(grid[y:y+length, x-1] > 0):
            score += 3
        # Right side
        if x + width < self.pallet_width and np.any(grid[y:y+length, x+width] > 0):
            score += 3
        # Top side
        if y > 0 and np.any(grid[y-1, x:x+width] > 0):
            score += 3
        # Bottom side
        if y + length < self.pallet_length and np.any(grid[y+length, x:x+width] > 0):
            score += 3
        
        # Corner positioning (even better than edge)
        if (x == 0 and y == 0) or (x == 0 and y + length == self.pallet_length) or \
           (x + width == self.pallet_width and y == 0) or \
           (x + width == self.pallet_width and y + length == self.pallet_length):
            score += 3  # Additional bonus for corner positions
        
        # Penalty for center position if center_empty is True
        if settings['center_empty'] and self._is_close_to_center(x, y, width, length):
            score -= 10
        
        return score
    
    def _is_close_to_center(self, x: int, y: int, width: int, length: int) -> bool:
        """Check if a position is close to the center of the pallet (for penalty)"""
        center_x = self.pallet_width // 2
        center_y = self.pallet_length // 2
        
        box_center_x = x + width // 2
        box_center_y = y + length // 2
        
        # Distance from box center to pallet center
        distance = math.sqrt((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2)
        
        # Close to center if within 33% of the pallet's diagonal
        max_distance = math.sqrt(self.pallet_width ** 2 + self.pallet_length ** 2) / 3
        
        return distance < max_distance
    
    def _optimize_with_rotation(self, boxes: List[Dict], layers: List[Dict], 
                              current_height: float, settings: Dict) -> Tuple[List[Dict], List[Dict], float]:
        """
        Try to optimize the pallet by rotating boxes that can be rotated
        
        Args:
            boxes: Remaining boxes to place
            layers: Current layers
            current_height: Current pallet height
            settings: Optimization settings
            
        Returns:
            Tuple of (updated_layers, still_remaining_boxes, new_height)
        """
        # First try to fit boxes in existing layers by rotation
        remaining_boxes = boxes.copy()
        i = 0
        
        while i < len(remaining_boxes):
            box = remaining_boxes[i]
            
            if not box['can_rotate']:
                i += 1
                continue
            
            placed = False
            
            # Try to place in existing layers
            for layer in layers:
                # Original orientation
                position = self._find_best_position(box, layer, settings)
                
                if position:
                    x, y, w, l, rotated = position
                    box['rotated'] = rotated
                    layer['boxes'].append(box)
                    layer['positions'].append((x, y, w, l))
                    layer['grid'][y:y+l, x:x+w] = 1
                    layer['weight'] += box['weight']
                    remaining_boxes.pop(i)
                    placed = True
                    break
            
            if not placed:
                i += 1
        
        # If we still have remaining boxes, try creating new layers
        if remaining_boxes and current_height < self.max_height:
            new_layers, still_remaining, added_height = self._create_layers(remaining_boxes, settings)
            layers.extend(new_layers)
            current_height += added_height
            return layers, still_remaining, current_height
        
        return layers, remaining_boxes, current_height
    
    def _find_complex_gaps(self, layer: Dict) -> List[Dict]:
        """
        Identify complex-shaped gaps in a layer that might be difficult to fill
        
        Args:
            layer: Layer data containing grid information
            
        Returns:
            List of identified complex gaps with their properties
        """
        # Get the grid from the layer
        grid = layer['grid']
        
        # Make a copy of the grid to avoid modifying the original
        gap_grid = np.copy(grid)
        
        # Find all empty cells (where the grid is 0)
        empty_mask = (gap_grid == 0)
        
        # Identify connected regions of empty space
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(empty_mask)
        
        complex_gaps = []
        
        # For each connected region, analyze its shape
        for region_idx in range(1, num_features + 1):
            region_mask = (labeled_array == region_idx)
            
            # Skip very small regions (less than 4 cells)
            if np.sum(region_mask) < 4:
                continue
                
            # Calculate region properties
            y_indices, x_indices = np.where(region_mask)
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            area = np.sum(region_mask)
            
            # Calculate occupancy ratio (area / bounding box area)
            bounding_box_area = width * height
            occupancy = area / bounding_box_area
            
            # If occupancy is low (complex shape), consider it a complex gap
            if occupancy < 0.7 and area >= 10:  # Threshold can be adjusted
                complex_gaps.append({
                    'x': min_x,
                    'y': min_y,
                    'width': width,
                    'height': height,
                    'area': area,
                    'occupancy': occupancy,
                    'mask': region_mask[min_y:max_y+1, min_x:max_x+1]
                })
        
        return complex_gaps
    
    def _fill_complex_gaps(self, layer: Dict, remaining_boxes: List[Dict], settings: Dict) -> List[Dict]:
        """
        Attempt to fill complex-shaped gaps with smaller boxes
        
        Args:
            layer: Layer data
            remaining_boxes: List of boxes that haven't been placed yet
            settings: Optimization settings
            
        Returns:
            Updated list of remaining boxes after filling gaps
        """
        # Only proceed if fill_complex_gaps is enabled
        if not settings.get('fill_complex_gaps', False):
            return remaining_boxes
            
        # Find complex gaps in this layer
        complex_gaps = self._find_complex_gaps(layer)
        
        if not complex_gaps:
            return remaining_boxes
            
        # Sort boxes by volume (smallest first) for gap filling
        sorted_boxes = sorted(remaining_boxes, key=lambda x: (x['width'] * x['length'] * x['height']))
        
        # Track boxes that were placed
        placed_boxes = []
        
        # For each complex gap, try to fill it with available boxes
        for gap in complex_gaps:
            # Create a sub-grid for this gap
            sub_grid = np.zeros((gap['height'], gap['width']), dtype=int)
            sub_grid[gap['mask']] = 0  # Empty space where the mask is True
            sub_grid[~gap['mask']] = 1  # Occupied space where the mask is False
            
            # Try placing boxes in this gap
            for box_idx, box in enumerate(sorted_boxes):
                if box_idx in placed_boxes:
                    continue  # Skip already placed boxes
                
                # Try normal orientation
                width, length = box['width'], box['length']
                
                # Check if box can fit in this gap
                position_found = False
                
                # Simple grid search within the gap
                for y in range(gap['height'] - length + 1):
                    for x in range(gap['width'] - width + 1):
                        # Check if this position is valid in the sub-grid
                        if np.any(sub_grid[y:y+length, x:x+width] > 0):
                            continue
                        
                        # Map local coordinates to global layer coordinates
                        global_x = gap['x'] + x
                        global_y = gap['y'] + y
                        
                        # Final check in global grid
                        if self._is_valid_position(layer['grid'], global_x, global_y, width, length):
                            # Place the box
                            layer['boxes'].append(box)
                            layer['positions'].append((global_x, global_y, width, length))
                            layer['grid'][global_y:global_y+length, global_x:global_x+width] = 1
                            layer['weight'] += box['weight']
                            
                            # Update sub-grid
                            sub_grid[y:y+length, x:x+width] = 1
                            
                            placed_boxes.append(box_idx)
                            position_found = True
                            break
                    
                    if position_found:
                        break
                
                # If not found, try rotating the box if allowed
                if not position_found and box['can_rotate'] and width != length:
                    width, length = box['length'], box['width']  # Swap dimensions
                    
                    for y in range(gap['height'] - length + 1):
                        for x in range(gap['width'] - width + 1):
                            # Check if this position is valid in the sub-grid
                            if np.any(sub_grid[y:y+length, x:x+width] > 0):
                                continue
                            
                            # Map local coordinates to global layer coordinates
                            global_x = gap['x'] + x
                            global_y = gap['y'] + y
                            
                            # Final check in global grid
                            if self._is_valid_position(layer['grid'], global_x, global_y, width, length):
                                # Place the box
                                box['rotated'] = True
                                layer['boxes'].append(box)
                                layer['positions'].append((global_x, global_y, width, length))
                                layer['grid'][global_y:global_y+length, global_x:global_x+width] = 1
                                layer['weight'] += box['weight']
                                
                                # Update sub-grid
                                sub_grid[y:y+length, x:x+width] = 1
                                
                                placed_boxes.append(box_idx)
                                break
                        
                        if box_idx in placed_boxes:
                            break
        
        # Remove placed boxes from remaining_boxes
        remaining_boxes = [box for idx, box in enumerate(sorted_boxes) if idx not in placed_boxes]
        
        return remaining_boxes
    
    def _improve_stability(self, layers: List[Dict], settings: Dict) -> List[Dict]:
        """
        Improve pallet stability by ensuring boxes in higher layers are supported by boxes below
        
        Args:
            layers: Pallet layers
            settings: Optimization settings
            
        Returns:
            Updated layers with improved stability
        """
        # For each layer except the bottom one
        for i in range(1, len(layers)):
            current_layer = layers[i]
            lower_layer = layers[i-1]
            
            # Create a stability map for the current layer
            stability_map = np.zeros((self.pallet_length, self.pallet_width))
            
            # For each box in the lower layer, mark its area as supporting
            for box_idx, pos in enumerate(lower_layer['positions']):
                x, y, w, l = pos
                box = lower_layer['boxes'][box_idx]
                # Mark the area this box can support (based on its load capacity)
                stability_map[y:y+l, x:x+w] = min(1, box['max_load'] / 10000)  # Normalize to 0-1 range
            
            # For each box in the current layer, check and improve its stability
            for box_idx, pos in enumerate(current_layer['positions']):
                x, y, w, l = pos
                
                # Calculate how much of this box is supported
                box_area = w * l
                supported_area = np.sum(stability_map[y:y+l, x:x+w])
                support_ratio = supported_area / box_area
                
                # If support is insufficient, try to shift the box for better support
                if support_ratio < settings['stability_factor']:
                    # Try different positions to improve stability
                    best_pos = pos
                    best_support = support_ratio
                    
                    # Try shifting slightly in each direction
                    for dx in [-5, -2, 0, 2, 5]:
                        for dy in [-5, -2, 0, 2, 5]:
                            new_x = max(0, min(self.pallet_width - w, x + dx))
                            new_y = max(0, min(self.pallet_length - l, y + dy))
                            
                            # Skip if this position would overlap with other boxes in the current layer
                            if self._would_overlap(new_x, new_y, w, l, current_layer, box_idx):
                                continue
                            
                            # Calculate support at this position
                            new_supported_area = np.sum(stability_map[new_y:new_y+l, new_x:new_x+w])
                            new_support_ratio = new_supported_area / box_area
                            
                            if new_support_ratio > best_support:
                                best_support = new_support_ratio
                                best_pos = (new_x, new_y, w, l)
                    
                    # Update position if we found a better one
                    if best_pos != pos:
                        # Update the grid
                        current_layer['grid'][y:y+l, x:x+w] = 0  # Clear old position
                        current_layer['positions'][box_idx] = best_pos
                        new_x, new_y, w, l = best_pos
                        current_layer['grid'][new_y:new_y+l, new_x:new_x+w] = 1  # Set new position
        
        return layers
    
    def _would_overlap(self, x: int, y: int, width: int, length: int, layer: Dict, exclude_idx: int) -> bool:
        """Check if a position would overlap with any other box in the layer (excluding the one at exclude_idx)"""
        for i, pos in enumerate(layer['positions']):
            if i == exclude_idx:
                continue
                
            px, py, pw, pl = pos
            
            # Check for overlap
            if (x < px + pw and x + width > px and
                y < py + pl and y + length > py):
                return True
                
        return False
    
    def _calculate_utilization(self, layers: List[Dict]) -> float:
        """Calculate space utilization percentage of the pallet"""
        total_volume = self.pallet_width * self.pallet_length * sum(layer['height'] for layer in layers)
        if total_volume == 0:
            return 0
            
        used_volume = sum(
            box['width'] * box['length'] * box['height']
            for layer in layers
            for box in layer['boxes']
        )
        
        return (used_volume / total_volume) * 100
    
    def _calculate_stability(self, layers: List[Dict]) -> float:
        """Calculate stability score of the pallet (0-100)"""
        if not layers:
            return 0
            
        # Base score starts at 100
        stability = 100
        
        # Penalize for each unsupported area in higher layers
        for i in range(1, len(layers)):
            current_layer = layers[i]
            lower_layer = layers[i-1]
            
            # Create a support map for the current layer
            support_map = np.zeros((self.pallet_length, self.pallet_width))
            
            # Mark areas supported by boxes in the lower layer
            for pos in lower_layer['positions']:
                x, y, w, l = pos
                support_map[y:y+l, x:x+w] = 1
            
            # Check support for each box in the current layer
            for box_idx, pos in enumerate(current_layer['positions']):
                x, y, w, l = pos
                box_area = w * l
                supported_area = np.sum(support_map[y:y+l, x:x+w])
                support_ratio = supported_area / box_area
                
                # Penalize for insufficient support
                if support_ratio < 0.8:
                    stability -= (0.8 - support_ratio) * 20
        
        # Penalize for height irregularity
        if len(layers) > 1:
            # Create a height map of the pallet
            height_map = np.zeros((self.pallet_length, self.pallet_width))
            current_height = 0
            
            for layer in layers:
                current_height += layer['height']
                for pos in layer['positions']:
                    x, y, w, l = pos
                    height_map[y:y+l, x:x+w] = current_height
            
            # Calculate height variation
            heights = height_map[height_map > 0]
            if len(heights) > 0:
                height_std = np.std(heights)
                max_acceptable_std = 30  # cm
                
                if height_std > max_acceptable_std:
                    stability -= min(20, (height_std - max_acceptable_std) / 2)
        
        return max(0, stability)

    def _find_best_pallet_for_box(self, box: Dict, pallet_weights: List[float], 
                                pallet_volumes: List[float], pallet_boxes: List[List[Dict]],
                                max_weight: float, max_volume: float) -> Optional[int]:
        """
        Find the best pallet for a box considering both weight and volume constraints
        
        Args:
            box: The box to place
            pallet_weights: List of current pallet weights
            pallet_volumes: List of current pallet volumes
            pallet_boxes: List of boxes assigned to each pallet
            max_weight: Maximum weight capacity per pallet
            max_volume: Maximum volume capacity per pallet
            
        Returns:
            Index of the best pallet or None if no suitable pallet found
        """
        best_pallet_idx = None
        best_score = float('inf')  # Lower is better
        
        # Check if the box can fit in any existing pallet
        for i in range(len(pallet_weights)):
            # Skip if adding this box would exceed weight or volume limits
            if pallet_weights[i] + box['weight'] > max_weight:
                continue
                
            if pallet_volumes[i] + box['volume'] > max_volume:
                continue
                
            # Calculate a score based on various factors
            # For heavy boxes, prefer pallets with less weight to balance loads
            # For lighter boxes, prefer pallets with more weight to maximize utilization
            
            # Base score - start with current weight
            score = pallet_weights[i]
            
            # Adjust score based on box weight
            if box['weight'] > 3000:  # Heavy box
                # Prefer lighter pallets (lower score is better)
                score = pallet_weights[i] / max_weight * 10000
            else:  # Medium or light box
                # Prefer heavier pallets (higher actual weight is better)
                # Invert the score so lower is better
                score = 10000 - (pallet_weights[i] / max_weight * 10000)
                
                # Additional scoring for pallets with heavy items already
                if pallet_boxes[i]:
                    has_heavy_box = any(b['weight'] > 3000 for b in pallet_boxes[i])
                    if has_heavy_box:
                        # Bonus for pallets that already have heavy boxes
                        score -= 2000  # Lower score is better
            
            # Bonus for better volume utilization
            volume_utilization = pallet_volumes[i] / max_volume
            if 0.6 <= volume_utilization <= 0.9:  # Ideal utilization range
                score -= 1000  # Bonus for ideal utilization
            
            # Adjust score to prefer even distribution of boxes
            if len(pallet_boxes) > 1:
                # Calculate average number of boxes per pallet
                avg_boxes = sum(len(boxes) for boxes in pallet_boxes) / len(pallet_boxes)
                # Prefer pallets with fewer boxes if significantly below average
                if len(pallet_boxes[i]) < avg_boxes * 0.7:
                    score -= 1500  # Big bonus for under-filled pallets
            
            # Check if this is the best pallet so far
            if score < best_score:
                best_score = score
                best_pallet_idx = i
        
        return best_pallet_idx

def optimize_pallet(orders, settings=None):
    """
    Generate an optimized pallet layout based on orders
    
    Args:
        orders: List of order dictionaries containing SKU and quantity
        settings: Dictionary of optimization settings (optional)
        
    Returns:
        Dictionary containing the optimized pallet layout
    """
    # Create a new instance of PalletOptimizer and call its optimize_pallet method
    try:
        optimizer = PalletOptimizer()
        return optimizer.optimize_pallet(orders, settings)
    except Exception as e:
        import traceback
        print(f"Error in optimize_pallet: {e}")
        print(traceback.format_exc())
        return {
            "status": "error", 
            "message": f"Optimization failed: {str(e)}",
            "layers": [],
            "total_pallets": 0,
            "total_volume": 0,
            "total_weight": 0
        } 