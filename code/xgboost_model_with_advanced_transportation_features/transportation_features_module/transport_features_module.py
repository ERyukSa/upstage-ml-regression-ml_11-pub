# transport_features_module.py
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import time
import pickle

# Correct Module Structure
# transport_features_module.py - Should contain:
# Basic sequential processing functions
# Core spatial indexing functions
# The main add_transportation_features function

# transport_features_module.py:
# ├── add_transportation_features_efficient() 
# ├── add_transportation_features(
# └── Other helper functions

def create_spatial_index(coords, metric='haversine', use_rtree=False, leaf_size=40): # Transportation Modules Optimizations
    """
    Create optimal spatial index based on data size and available libraries
    
    Args:
        coords: Coordinate array in radians
        metric: Distance metric ('haversine' for geographic coordinates)
        use_rtree: Whether to use R-tree if available (faster for large datasets)
        leaf_size: Leaf size for BallTree (higher values use less memory)
        
    Returns:
        Spatial index object
    """
    if use_rtree and len(coords) > 5000:
        try:
            from rtree import index as rtree_index
            # Create R-tree index (faster for large datasets)
            idx = rtree_index.Index()
            # Fill the R-tree with coordinates
            for i, (lat, lon) in enumerate(coords):
                # R-tree requires a bounding box (minx, miny, maxx, maxy)
                idx.insert(i, (lon, lat, lon, lat))
            return {'type': 'rtree', 'index': idx, 'coords': coords}
        except ImportError:
            print("R-tree not available, falling back to BallTree")
    
    # Use BallTree as fallback
    from sklearn.neighbors import BallTree
    tree = BallTree(coords, metric=metric, leaf_size=leaf_size)
    return {'type': 'balltree', 'index': tree}


# def add_transportation_features_efficient(data, subway_data, bus_data, n_jobs=None):
#     """
#     Add transportation features using spatial indexing for efficiency
    
#     Args:
#         data: DataFrame containing housing data with coordinates
#         subway_data: DataFrame containing subway station information
#         bus_data: DataFrame containing bus stop information
#         n_jobs: Number of parallel jobs to run
        
#     Returns:
#         DataFrame with added transportation features
#     """
#     print("Adding transportation features using spatial indexing...")
#     start_time = time.time()
    
#     # Make a copy to avoid modifying the original data
#     result = data.copy()
    
#     # Define coordinate columns
#     lat_col, lon_col = '좌표Y', '좌표X'  # Housing data
#     subway_lat_col, subway_lon_col = '위도', '경도'  # Subway data
#     bus_lat_col, bus_lon_col = 'Y좌표', 'X좌표'  # Bus data
    
#     # Skip if coordinate columns don't exist
#     if not all(col in result.columns for col in [lat_col, lon_col]):
#         print("Coordinate columns not found. Skipping transportation features.")
#         return result
    
#     # Initialize new features
#     result['nearest_subway_distance'] = np.nan
#     result['nearest_subway_name'] = ''
#     result['nearest_subway_line'] = ''
#     result['subway_stations_500m'] = 0
#     result['subway_stations_1km'] = 0
#     result['subway_stations_3km'] = 0
    
#     result['nearest_bus_distance'] = np.nan
#     result['nearest_bus_name'] = ''
#     result['nearest_bus_type'] = ''
#     result['bus_stops_300m'] = 0
#     result['bus_stops_500m'] = 0
#     result['bus_stops_1km'] = 0
    
#     # Filter out properties with missing coordinates
#     valid_mask = ~(result[lat_col].isna() | result[lon_col].isna())
#     valid_indices = result[valid_mask].index
    
#     if len(valid_indices) == 0:
#         print("No valid coordinates found in housing data")
#         return result
    
#     # Print diagnostic information about the data
#     print(f"Total rows: {len(result)}, Rows with valid coordinates: {len(valid_indices)}")
#     print(f"Subway stations: {len(subway_data)}, Bus stops: {len(bus_data)}")
    
#     # Earth radius for haversine calculations (meters)
#     earth_radius = 6371000
    
#     # Process subway data if available
#     if len(subway_data) > 0:
#         print(f"Building spatial index for {len(subway_data)} subway stations...")
        
#         # Use BallTree for all cases for simplicity and reliability
#         # Convert subway coordinates to radians and build Ball Tree
#         subway_coords = np.radians(subway_data[[subway_lat_col, subway_lon_col]].values)
#         subway_tree = BallTree(subway_coords, metric='haversine')
        
#         # Process in batches to avoid memory issues
#         batch_size = 10000  # Smaller batch size for better memory management
        
#         print(f"Processing nearest subway stations in batches of {batch_size}...")
        
#         for i in range(0, len(valid_indices), batch_size):
#             # Get current batch
#             batch_end = min(i + batch_size, len(valid_indices))
#             batch_indices = valid_indices[i:batch_end]
            
#             # Get coordinates for this batch
#             batch_coords = np.radians(result.loc[batch_indices, [lat_col, lon_col]].values)
            
#             # Find nearest subway station for each property
#             distances, indices = subway_tree.query(batch_coords, k=1)
            
#             # Convert distances from radians to meters
#             distances = distances.flatten() * earth_radius
#             indices = indices.flatten()
            
#             # Assign nearest station results to this batch
#             result.loc[batch_indices, 'nearest_subway_distance'] = distances
            
#             # Extract names and lines directly
#             names = []
#             lines = []
#             for idx in indices:
#                 if idx < 0 or idx >= len(subway_data):
#                     names.append('Unknown')
#                     lines.append('Unknown')
#                 else:
#                     names.append(subway_data.iloc[idx]['역사명'])
#                     lines.append(subway_data.iloc[idx]['호선'])
            
#             # Directly assign to the dataframe without Series creation
#             for j, idx in enumerate(batch_indices):
#                 if j < len(names):  # Safety check
#                     result.loc[idx, 'nearest_subway_name'] = names[j]
#                     result.loc[idx, 'nearest_subway_line'] = lines[j]
            
#             # Count stations within different radiuses for this batch
#             radius_500m = 500 / earth_radius
#             radius_1km = 1000 / earth_radius
#             radius_3km = 3000 / earth_radius
            
#             # Get indices of stations within each radius
#             indices_500m = subway_tree.query_radius(batch_coords, r=radius_500m, count_only=True)
#             indices_1km = subway_tree.query_radius(batch_coords, r=radius_1km, count_only=True)
#             indices_3km = subway_tree.query_radius(batch_coords, r=radius_3km, count_only=True)
            
#             # Assign values directly to batch
#             result.loc[batch_indices, 'subway_stations_500m'] = indices_500m
#             result.loc[batch_indices, 'subway_stations_1km'] = indices_1km
#             result.loc[batch_indices, 'subway_stations_3km'] = indices_3km
            
#             # Report progress for large datasets
#             if i % (batch_size * 10) == 0 and i > 0:
#                 print(f"Processed {i} of {len(valid_indices)} rows...")
def add_transportation_features_efficient(data, subway_data, bus_data, n_jobs=None, subway_tree=None, bus_tree=None):
    """Add transportation features using spatial indexing with enhanced vectorization."""
    print("Adding transportation features using vectorized operations...")
    start_time = time.time()
    
    # Make a copy to avoid modifying the original data
    result = data.copy()
    
    # Define coordinate columns
    lat_col, lon_col = '좌표Y', '좌표X'  # Housing data
    subway_lat_col, subway_lon_col = '위도', '경도'  # Subway data
    bus_lat_col, bus_lon_col = 'Y좌표', 'X좌표'  # Bus data
    
    # Skip if coordinate columns don't exist
    if not all(col in result.columns for col in [lat_col, lon_col]):
        print("Coordinate columns not found. Skipping transportation features.")
        return result
    
    # Initialize all new columns at once with appropriate types
    new_float_columns = ['nearest_subway_distance', 'nearest_bus_distance']
    new_int_columns = ['subway_stations_500m', 'subway_stations_1km', 'subway_stations_3km',
                       'bus_stops_300m', 'bus_stops_500m', 'bus_stops_1km', 
                       'transit_score', 'near_subway', 'near_bus', 'near_public_transit']
    new_str_columns = ['nearest_subway_name', 'nearest_subway_line', 'nearest_bus_name', 
                       'nearest_bus_type', 'transit_quality']
    
    # Initialize all at once
    for col in new_float_columns:
        result[col] = np.nan
    for col in new_int_columns:
        result[col] = 0
    for col in new_str_columns:
        result[col] = ''
    
    # Filter out properties with missing coordinates
    valid_mask = ~(result[lat_col].isna() | result[lon_col].isna())
    valid_indices = result[valid_mask].index
    
    if len(valid_indices) == 0:
        print("No valid coordinates found in housing data")
        return result
    
    # Print diagnostic information
    print(f"Total rows: {len(result)}, Rows with valid coordinates: {len(valid_indices)}")
    print(f"Subway stations: {len(subway_data)}, Bus stops: {len(bus_data)}")
    
    # Earth radius for haversine calculations (meters)
    earth_radius = 6371000
    
    # Process subway data using vectorized operations
    if len(subway_data) > 0:
        # Use provided tree or build a new one
        if subway_tree is None:
            print(f"Building spatial index for {len(subway_data)} subway stations...")
            subway_coords = np.radians(subway_data[[subway_lat_col, subway_lon_col]].values)
            subway_tree = BallTree(subway_coords, metric='haversine')
        
        # Get coordinates for valid properties - all at once
        all_property_coords = np.radians(result.loc[valid_indices, [lat_col, lon_col]].values)
        
        # Calculate batch size based on data size and available memory
        batch_size = min(10000, len(valid_indices))
        
        # Process in batches to avoid memory issues
        for i in range(0, len(valid_indices), batch_size):
            batch_end = min(i + batch_size, len(valid_indices))
            batch_indices = valid_indices[i:batch_end]
            batch_coords = all_property_coords[i:batch_end]
            
            # Find nearest subway station for each property
            distances, indices = subway_tree.query(batch_coords, k=1)
            
            # Convert distances from radians to meters - vectorized operation
            distances = distances.flatten() * earth_radius
            
            # Assign nearest station results to this batch - vectorized assignment
            result.loc[batch_indices, 'nearest_subway_distance'] = distances
            
            # Create arrays for station names and lines - vectorized string lookup
            station_names = subway_data.iloc[indices.flatten()]['역사명'].values
            station_lines = subway_data.iloc[indices.flatten()]['호선'].values
            
            # Assign in bulk operations
            result.loc[batch_indices, 'nearest_subway_name'] = station_names
            result.loc[batch_indices, 'nearest_subway_line'] = station_lines
            
            # Count stations within different radiuses - vectorized radius queries
            radius_500m = 500 / earth_radius
            radius_1km = 1000 / earth_radius
            radius_3km = 3000 / earth_radius
            
            # Get counts for each radius - vectorized operations
            count_500m = subway_tree.query_radius(batch_coords, r=radius_500m, count_only=True)
            count_1km = subway_tree.query_radius(batch_coords, r=radius_1km, count_only=True)
            count_3km = subway_tree.query_radius(batch_coords, r=radius_3km, count_only=True)
            
            # Assign values in bulk
            result.loc[batch_indices, 'subway_stations_500m'] = count_500m
            result.loc[batch_indices, 'subway_stations_1km'] = count_1km
            result.loc[batch_indices, 'subway_stations_3km'] = count_3km
            
            # Report progress for large datasets
            if i % (batch_size * 10) == 0 and i > 0:
                print(f"Processed {i}/{len(valid_indices)} rows...")
    
    # Repeat similar vectorized approach for bus data
    # ...
    # Process bus data using vectorized operations
    if len(bus_data) > 0:
        # Use provided tree or build a new one
        if bus_tree is None:
            print(f"Building spatial index for {len(bus_data)} bus stops...")
            bus_coords = np.radians(bus_data[[bus_lat_col, bus_lon_col]].values)
            bus_tree = BallTree(bus_coords, metric='haversine')
        
        # Get coordinates for valid properties - all at once
        all_property_coords = np.radians(result.loc[valid_indices, [lat_col, lon_col]].values)
        
        # Calculate batch size based on data size and available memory
        batch_size = min(10000, len(valid_indices))
        
        # Process in batches to avoid memory issues
        for i in range(0, len(valid_indices), batch_size):
            batch_end = min(i + batch_size, len(valid_indices))
            batch_indices = valid_indices[i:batch_end]
            batch_coords = all_property_coords[i:batch_end]
            
            # Find nearest bus stop for each property
            distances, indices = bus_tree.query(batch_coords, k=1)
            
            # Convert distances from radians to meters - vectorized operation
            distances = distances.flatten() * earth_radius
            
            # Assign nearest bus stop results to this batch - vectorized assignment
            result.loc[batch_indices, 'nearest_bus_distance'] = distances
            
            # Create arrays for bus stop names and types - vectorized string lookup
            stop_names = bus_data.iloc[indices.flatten()]['정류소명'].values
            stop_types = bus_data.iloc[indices.flatten()]['정류소 타입'].values
            
            # Assign in bulk operations
            result.loc[batch_indices, 'nearest_bus_name'] = stop_names
            result.loc[batch_indices, 'nearest_bus_type'] = stop_types
            
            # Count bus stops within different radiuses - vectorized radius queries
            radius_300m = 300 / earth_radius
            radius_500m = 500 / earth_radius
            radius_1km = 1000 / earth_radius
            
            # Get counts for each radius - vectorized operations
            count_300m = bus_tree.query_radius(batch_coords, r=radius_300m, count_only=True)
            count_500m = bus_tree.query_radius(batch_coords, r=radius_500m, count_only=True)
            count_1km = bus_tree.query_radius(batch_coords, r=radius_1km, count_only=True)
            
            # Assign values in bulk
            result.loc[batch_indices, 'bus_stops_300m'] = count_300m
            result.loc[batch_indices, 'bus_stops_500m'] = count_500m
            result.loc[batch_indices, 'bus_stops_1km'] = count_1km
            
            # Report progress for large datasets
            if i % (batch_size * 10) == 0 and i > 0:
                print(f"Processed {i}/{len(valid_indices)} rows for bus data...")
                
    # Calculate transit score using vectorized operations
    print("Calculating transit scores using vectorized operations...")
    
    # Initialize scores with zeros
    result['transit_score'] = 0
    
    # Compute all masks at once
    valid_subway = ~result['nearest_subway_distance'].isna()
    subway_dist_lt_500 = valid_subway & (result['nearest_subway_distance'] < 500)
    subway_dist_500_1000 = valid_subway & (result['nearest_subway_distance'] >= 500) & (result['nearest_subway_distance'] < 1000)
    subway_dist_1000_2000 = valid_subway & (result['nearest_subway_distance'] >= 1000) & (result['nearest_subway_distance'] < 2000)
    subway_dist_2000_3000 = valid_subway & (result['nearest_subway_distance'] >= 2000) & (result['nearest_subway_distance'] < 3000)
    
    # Apply scores in bulk - vectorized conditional assignment
    result.loc[subway_dist_lt_500, 'transit_score'] += 5
    result.loc[subway_dist_500_1000, 'transit_score'] += 4
    result.loc[subway_dist_1000_2000, 'transit_score'] += 2
    result.loc[subway_dist_2000_3000, 'transit_score'] += 1
    
    # Add bonus for multiple stations in bulk
    result.loc[valid_subway, 'transit_score'] += result.loc[valid_subway, 'subway_stations_1km'].clip(upper=3)
    
    # Similar vectorized approach for bus scores
    # ...
    # Calculate bus scores using vectorized operations
    valid_bus = ~result['nearest_bus_distance'].isna()
    bus_dist_lt_200 = valid_bus & (result['nearest_bus_distance'] < 200)
    bus_dist_200_400 = valid_bus & (result['nearest_bus_distance'] >= 200) & (result['nearest_bus_distance'] < 400)
    bus_dist_400_800 = valid_bus & (result['nearest_bus_distance'] >= 400) & (result['nearest_bus_distance'] < 800)

    # Apply scores in bulk - vectorized conditional assignment
    result.loc[bus_dist_lt_200, 'transit_score'] += 3
    result.loc[bus_dist_200_400, 'transit_score'] += 2
    result.loc[bus_dist_400_800, 'transit_score'] += 1

    # Add bonus for multiple bus stops in bulk
    result.loc[valid_bus, 'transit_score'] += result.loc[valid_bus, 'bus_stops_500m'].clip(upper=2)

    # Create transit quality categories using vectorized binning
    result['transit_quality'] = pd.cut(
        result['transit_score'], 
        bins=[-1, 2, 5, 8, 20], 
        labels=['poor', 'average', 'good', 'excellent']
    )


    # Create binary features using vectorized operations
    result['near_subway'] = (valid_subway & (result['nearest_subway_distance'] < 1000)).astype(int)
    result['near_bus'] = (valid_bus & (result['nearest_bus_distance'] < 400)).astype(int)
    result['near_public_transit'] = ((result['near_subway'] == 1) | (result['near_bus'] == 1)).astype(int)
    
    # Process time tracking
    elapsed_time = time.time() - start_time
    print(f"Transportation features added successfully in {elapsed_time:.2f} seconds.")
    
    return result
    
    # UNVECTORIZED VERSION(SLOW)
    # # Process bus data if available - using the same batched approach
    # if len(bus_data) > 0:
    #     print(f"Building spatial index for {len(bus_data)} bus stops...")
        
    #     # Convert bus coordinates to radians and build Ball Tree
    #     bus_coords = np.radians(bus_data[[bus_lat_col, bus_lon_col]].values)
    #     bus_tree = BallTree(bus_coords, metric='haversine')
        
    #     # Process in batches to avoid memory issues
    #     batch_size = 10000  # Smaller batch size for better memory management
        
    #     print(f"Processing nearest bus stops in batches of {batch_size}...")
        
    #     for i in range(0, len(valid_indices), batch_size):
    #         # Get current batch
    #         batch_end = min(i + batch_size, len(valid_indices))
    #         batch_indices = valid_indices[i:batch_end]
            
    #         # Get coordinates for this batch
    #         batch_coords = np.radians(result.loc[batch_indices, [lat_col, lon_col]].values)
            
    #         # Find nearest bus stop for each property
    #         distances, indices = bus_tree.query(batch_coords, k=1)
            
    #         # Convert distances from radians to meters
    #         distances = distances.flatten() * earth_radius
    #         indices = indices.flatten()
            
    #         # Assign nearest bus stop results to this batch
    #         result.loc[batch_indices, 'nearest_bus_distance'] = distances
            
    #         # Extract names and types directly
    #         names = []
    #         types = []
    #         for idx in indices:
    #             if idx < 0 or idx >= len(bus_data):
    #                 names.append('Unknown')
    #                 types.append('Unknown')
    #             else:
    #                 names.append(bus_data.iloc[idx]['정류소명'])
    #                 types.append(bus_data.iloc[idx]['정류소 타입'])
            
    #         # Directly assign to the dataframe without Series creation
    #         for j, idx in enumerate(batch_indices):
    #             if j < len(names):  # Safety check
    #                 result.loc[idx, 'nearest_bus_name'] = names[j]
    #                 result.loc[idx, 'nearest_bus_type'] = types[j]
            
    #         # Count bus stops within different radiuses for this batch
    #         radius_300m = 300 / earth_radius
    #         radius_500m = 500 / earth_radius
    #         radius_1km = 1000 / earth_radius
            
    #         # Get indices of stops within each radius
    #         indices_300m = bus_tree.query_radius(batch_coords, r=radius_300m, count_only=True)
    #         indices_500m = bus_tree.query_radius(batch_coords, r=radius_500m, count_only=True)
    #         indices_1km = bus_tree.query_radius(batch_coords, r=radius_1km, count_only=True)
            
    #         # Assign values directly to batch
    #         result.loc[batch_indices, 'bus_stops_300m'] = indices_300m
    #         result.loc[batch_indices, 'bus_stops_500m'] = indices_500m
    #         result.loc[batch_indices, 'bus_stops_1km'] = indices_1km
            
    #         # Report progress for large datasets
    #         if i % (batch_size * 10) == 0 and i > 0:
    #             print(f"Processed {i} of {len(valid_indices)} rows...")
    
    # # Calculate transit score 
    # print("Calculating transit scores...")
    
    # # Initialize transit score
    # result['transit_score'] = 0
    
    # # Add subway proximity scores
    # valid_subway = ~result['nearest_subway_distance'].isna()
    
    # # Add points based on subway proximity
    # mask_s1 = valid_subway & (result['nearest_subway_distance'] < 500)
    # mask_s2 = valid_subway & (result['nearest_subway_distance'] >= 500) & (result['nearest_subway_distance'] < 1000)
    # mask_s3 = valid_subway & (result['nearest_subway_distance'] >= 1000) & (result['nearest_subway_distance'] < 2000)
    # mask_s4 = valid_subway & (result['nearest_subway_distance'] >= 2000) & (result['nearest_subway_distance'] < 3000)

    # result.loc[mask_s1, 'transit_score'] += 5
    # result.loc[mask_s2, 'transit_score'] += 4
    # result.loc[mask_s3, 'transit_score'] += 2
    # result.loc[mask_s4, 'transit_score'] += 1
    
    # # Add points for multiple subway stations
    # result.loc[valid_subway, 'transit_score'] += result.loc[valid_subway, 'subway_stations_1km'].clip(upper=3)
    
    # # Add bus proximity scores
    # valid_bus = ~result['nearest_bus_distance'].isna()
    
    # # Vectorized bus score calculation
    # mask_b1 = valid_bus & (result['nearest_bus_distance'] < 200)
    # mask_b2 = valid_bus & (result['nearest_bus_distance'] >= 200) & (result['nearest_bus_distance'] < 400)
    # mask_b3 = valid_bus & (result['nearest_bus_distance'] >= 400) & (result['nearest_bus_distance'] < 800)

    # # Create a temporary Series for the bus score
    # bus_score = pd.Series(0, index=result.index)
    # bus_score.loc[mask_b1] = 3
    # bus_score.loc[mask_b2] = 2
    # bus_score.loc[mask_b3] = 1

    # # Add to transit score
    # result.loc[valid_bus, 'transit_score'] += bus_score[valid_bus]

    # # Vectorized calculation for multiple bus stops
    # bus_stops_contrib = result.loc[valid_bus, 'bus_stops_500m'].clip(upper=2)
    # result.loc[valid_bus, 'transit_score'] += bus_stops_contrib
    
    # # Create transit quality categories
    # result['transit_quality'] = pd.cut(
    #     result['transit_score'], 
    #     bins=[-1, 2, 5, 8, 20], 
    #     labels=['poor', 'average', 'good', 'excellent']
    # )
    
    # # Create binary features
    # mask_near_subway = valid_subway & (result['nearest_subway_distance'] < 1000)
    # mask_near_bus = valid_bus & (result['nearest_bus_distance'] < 400)

    # # Create binary features in one go
    # result['near_subway'] = 0
    # result['near_bus'] = 0 
    # result['near_public_transit'] = 0

    # # Assign in bulk operations
    # result.loc[mask_near_subway, 'near_subway'] = 1
    # result.loc[mask_near_bus, 'near_bus'] = 1
    # result.loc[mask_near_subway | mask_near_bus, 'near_public_transit'] = 1
    
    # # Process time tracking
    # elapsed_time = time.time() - start_time
    # print(f"Transportation features added successfully in {elapsed_time:.2f} seconds.")
    # print(f"New features: {[col for col in result.columns if col not in data.columns]}")
    
    # return result

def process_parallel_chunk(chunk_args):
    """Process a single chunk of data with transportation features
    
    Args:
        chunk_args: Tuple containing (chunk_data, subway_data, bus_data, chunk_id, n_chunks)
        
    Returns:
        Processed chunk with transportation features
    """
    chunk_data, subway_data, bus_data, chunk_id, n_chunks = chunk_args
    
    try:
        print(f"Processing chunk {chunk_id}/{n_chunks}")
        # Process the chunk directly without further batching
        result = add_transportation_features_efficient(chunk_data, subway_data, bus_data)
        print(f"Chunk {chunk_id}/{n_chunks} completed successfully")
        return result
    except Exception as e:
        import traceback
        print(f"Error in chunk {chunk_id}/{n_chunks}: {str(e)}")
        print(traceback.format_exc())
        # Return the original chunk to avoid data loss
        return chunk_data



def add_transportation_features(data, subway_data, bus_data, use_parallel=None, use_cache=True, cache_dir='./cache/transportation_features'):
    """
    Main function to add transportation features to the dataset.
    Automatically determines whether to use parallel processing based on dataset size.
    
    Args:
        data: DataFrame containing housing data with coordinates
        subway_data: DataFrame containing subway station information
        bus_data: DataFrame containing bus stop information
        use_parallel: Force parallel processing (True) or sequential (False), or auto-detect (None)
        
    Returns:
        DataFrame with added transportation features
    """


    """Add transportation features with improved caching"""
    # Create caches directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache filenames
    subway_cache = os.path.join(cache_dir, 'subway_tree.pkl')
    bus_cache = os.path.join(cache_dir, 'bus_tree.pkl')
    results_cache = os.path.join(cache_dir, f'transport_features_rows{len(data)}.pkl')
    
    # Try loading cached results first (fastest path)
    if use_cache and os.path.exists(results_cache):
        print(f"Loading cached final results for {len(data)} rows")
        try:
            with open(results_cache, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cached results: {e}. Will recompute.")

    
    try:
        # Check if we have the data needed
        if '좌표X' not in data.columns or '좌표Y' not in data.columns:
            print("Coordinate columns not found. Skipping transportation features.")
            return data

        # Build or load spatial indices
        subway_tree = None
        bus_tree = None
        
        # Try to load cached indices if available
        if use_cache:
            # Create caches directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            if os.path.exists(subway_cache):
                print(f"Loading cached subway spatial index from {subway_cache}")
                try:
                    with open(subway_cache, 'rb') as f:
                        subway_tree = pickle.load(f)
                except Exception as e:
                    print(f"Error loading cached subway index: {e}. Will rebuild.")
            
            if os.path.exists(bus_cache):
                print(f"Loading cached bus spatial index from {bus_cache}")
                try:
                    with open(bus_cache, 'rb') as f:
                        bus_tree = pickle.load(f)
                except Exception as e:
                    print(f"Error loading cached bus index: {e}. Will rebuild.")
        
        # Build indices if not loaded from cache
        if subway_tree is None and len(subway_data) > 0:
            print(f"Building spatial index for {len(subway_data)} subway stations...")
            subway_coords = np.radians(subway_data[['위도', '경도']].values)
            subway_tree = BallTree(subway_coords, metric='haversine')
            
            # Cache the index
            if use_cache:
                print(f"Saving subway spatial index to {subway_cache}")
                with open(subway_cache, 'wb') as f:
                    pickle.dump(subway_tree, f)
        
        if bus_tree is None and len(bus_data) > 0:
            print(f"Building spatial index for {len(bus_data)} bus stops...")
            bus_coords = np.radians(bus_data[['Y좌표', 'X좌표']].values)
            bus_tree = BallTree(bus_coords, metric='haversine')
            
            # Cache the index
            if use_cache:
                print(f"Saving bus spatial index to {bus_cache}")
                with open(bus_cache, 'wb') as f:
                    pickle.dump(bus_tree, f)
        

        # Process a sample to verify functionality
        print("Processing a sample to verify functionality...")
        sample_size = min(1000, len(data))
        sample_data = data.sample(sample_size, random_state=42).copy()
        
        # Test with sequential processing first to avoid parallel issues
        try:
            sample_result = add_transportation_features_efficient(sample_data, subway_data, bus_data)
            print("Sample processing successful. Proceeding with full dataset.")
        except Exception as e:
            print(f"Error in sample processing: {str(e)}")
            print("Unable to add transportation features. Returning original data.")
            return data
        
        # For large datasets (over 100,000 rows), use sequential for stability
        if len(data) > 100000:
            print(f"Processing {len(data)} rows sequentially for stability...")
            return add_transportation_features_efficient(data, subway_data, bus_data)
        
        # For smaller datasets, can use parallel if requested
        if use_parallel:
            print(f"Using parallel processing for {len(data)} rows.")
            # Import at function level to avoid circular imports
            from transport_features_parallel import process_in_parallel
            return process_in_parallel(data, subway_data, bus_data)
        else:
            print(f"Using sequential processing for {len(data)} rows.")
            return add_transportation_features_efficient(data, subway_data, bus_data)
            
    except Exception as e:
        import traceback
        print(f"Error processing transportation features: {str(e)}")
        print(traceback.format_exc())
        print("Falling back to basic features only")


        # Check for cached results first
        if use_cache and os.path.exists(results_cache):
            print(f"Loading cached transportation features for {len(data)} rows")
            with open(results_cache, 'rb') as f:
                return pickle.load(f)
            
        # Save results before returning
        if use_cache:
            with open(results_cache, 'wb') as f:
                pickle.dump(data, f) # result


        return data