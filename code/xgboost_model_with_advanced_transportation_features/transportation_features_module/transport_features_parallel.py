import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import time



# 2. transport_features_parallel.py - Should contain:
# Parallel processing functions including process_in_parallel
# Chunking strategies
# The main add_transportation_features_parallel function

# transport_features_parallel.py:
# ├── process_chunk()
# ├── process_in_parallel()  <-- Should be here, not in module.py
# ├── process_in_parallel_with_chunking()  <-- Add this new function
# └── add_transportation_features_parallel()


def build_and_save_indices(subway_data, bus_data, subway_lat_col='위도', subway_lon_col='경도', 
                          bus_lat_col='Y좌표', bus_lon_col='X좌표', cache_dir='./cache'):
    """Build and save spatial indices for reuse"""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    subway_index_path = os.path.join(cache_dir, 'subway_index.pkl')
    bus_index_path = os.path.join(cache_dir, 'bus_index.pkl')
    
    # Build and save subway index
    print(f"Building spatial index for {len(subway_data)} subway stations...")
    subway_coords = np.radians(subway_data[[subway_lat_col, subway_lon_col]].values)
    subway_tree = BallTree(subway_coords, metric='haversine')
    
    with open(subway_index_path, 'wb') as f:
        pickle.dump((subway_tree, subway_data[['역사명', '호선']]), f)
    
    # Build and save bus index
    print(f"Building spatial index for {len(bus_data)} bus stops...")
    bus_coords = np.radians(bus_data[[bus_lat_col, bus_lon_col]].values)
    bus_tree = BallTree(bus_coords, metric='haversine')
    
    with open(bus_index_path, 'wb') as f:
        pickle.dump((bus_tree, bus_data[['정류소명', '정류소 타입']]), f)
    
    print(f"Indices saved to {cache_dir}")
    return subway_index_path, bus_index_path

def load_indices(cache_dir='./cache'):
    """Load precomputed spatial indices"""
    subway_index_path = os.path.join(cache_dir, 'subway_index.pkl')
    bus_index_path = os.path.join(cache_dir, 'bus_index.pkl')
    
    if os.path.exists(subway_index_path) and os.path.exists(bus_index_path):
        print("Loading precomputed spatial indices...")
        with open(subway_index_path, 'rb') as f:
            subway_tree, subway_info = pickle.load(f)
        
        with open(bus_index_path, 'rb') as f:
            bus_tree, bus_info = pickle.load(f)
        
        return (subway_tree, subway_info), (bus_tree, bus_info)
    else:
        print("Precomputed indices not found")
        return None, None
    

   

def process_chunk(chunk_data, subway_tree, subway_info, bus_tree, bus_info, lat_col='좌표Y', lon_col='좌표X'):
    """Process a chunk of properties using precomputed spatial indices"""
    # Make a copy to avoid modifying the original data
    data = chunk_data.copy()
    
    # Add columns for results
    data['nearest_subway_distance'] = np.nan
    data['nearest_subway_name'] = ''
    data['nearest_subway_line'] = ''
    data['subway_stations_500m'] = 0
    data['subway_stations_1km'] = 0
    data['subway_stations_3km'] = 0
    
    data['nearest_bus_distance'] = np.nan
    data['nearest_bus_name'] = ''
    data['nearest_bus_type'] = ''
    data['bus_stops_300m'] = 0
    data['bus_stops_500m'] = 0
    data['bus_stops_1km'] = 0
    
    # Filter out properties with missing coordinates
    valid_mask = ~(data[lat_col].isna() | data[lon_col].isna())
    valid_indices = data[valid_mask].index
    
    if len(valid_indices) == 0:
        return data
    
    # Earth radius in meters (for converting between radians and meters)
    earth_radius = 6371000
    
    # Get coordinates for valid properties
    property_coords = np.radians(data.loc[valid_indices, [lat_col, lon_col]].values)
    
    # Process subway data
    if subway_tree is not None:
        # Find nearest subway station
        distances, indices = subway_tree.query(property_coords, k=1)
        
        # Convert distances from radians to meters
        distances = distances * earth_radius
        
        # Assign nearest station results
        data.loc[valid_indices, 'nearest_subway_distance'] = distances.flatten()
        data.loc[valid_indices, 'nearest_subway_name'] = subway_info.iloc[indices.flatten(), 0].values
        data.loc[valid_indices, 'nearest_subway_line'] = subway_info.iloc[indices.flatten(), 1].values
        
        # Count stations within different radiuses
        radius_500m = 500 / earth_radius
        radius_1km = 1000 / earth_radius
        radius_3km = 3000 / earth_radius
        
        count_500m = subway_tree.query_radius(property_coords, r=radius_500m, count_only=True)
        count_1km = subway_tree.query_radius(property_coords, r=radius_1km, count_only=True)
        count_3km = subway_tree.query_radius(property_coords, r=radius_3km, count_only=True)
        
        data.loc[valid_indices, 'subway_stations_500m'] = count_500m
        data.loc[valid_indices, 'subway_stations_1km'] = count_1km
        data.loc[valid_indices, 'subway_stations_3km'] = count_3km
    
    # Process bus data
    if bus_tree is not None:
        # Find nearest bus stop
        distances, indices = bus_tree.query(property_coords, k=1)
        
        # Convert distances from radians to meters
        distances = distances * earth_radius
        
        # Assign nearest bus stop results
        data.loc[valid_indices, 'nearest_bus_distance'] = distances.flatten()
        data.loc[valid_indices, 'nearest_bus_name'] = bus_info.iloc[indices.flatten(), 0].values
        data.loc[valid_indices, 'nearest_bus_type'] = bus_info.iloc[indices.flatten(), 1].values
        
        # Count bus stops within different radiuses
        radius_300m = 300 / earth_radius
        radius_500m = 500 / earth_radius
        radius_1km = 1000 / earth_radius
        
        count_300m = bus_tree.query_radius(property_coords, r=radius_300m, count_only=True)
        count_500m = bus_tree.query_radius(property_coords, r=radius_500m, count_only=True)
        count_1km = bus_tree.query_radius(property_coords, r=radius_1km, count_only=True)
        
        data.loc[valid_indices, 'bus_stops_300m'] = count_300m
        data.loc[valid_indices, 'bus_stops_500m'] = count_500m
        data.loc[valid_indices, 'bus_stops_1km'] = count_1km
    
    # Calculate transit score
    data['transit_score'] = 0
    
    # Add points based on subway proximity
    valid_subway = ~data['nearest_subway_distance'].isna()
    data.loc[valid_subway & (data['nearest_subway_distance'] < 500), 'transit_score'] += 5
    data.loc[valid_subway & (data['nearest_subway_distance'] >= 500) & (data['nearest_subway_distance'] < 1000), 'transit_score'] += 4
    data.loc[valid_subway & (data['nearest_subway_distance'] >= 1000) & (data['nearest_subway_distance'] < 2000), 'transit_score'] += 2
    data.loc[valid_subway & (data['nearest_subway_distance'] >= 2000) & (data['nearest_subway_distance'] < 3000), 'transit_score'] += 1
    
    # Add points for multiple subway stations
    data.loc[valid_subway, 'transit_score'] += data.loc[valid_subway, 'subway_stations_1km'].clip(upper=3)
    
    # Add points based on bus proximity
    valid_bus = ~data['nearest_bus_distance'].isna()
    data.loc[valid_bus & (data['nearest_bus_distance'] < 200), 'transit_score'] += 3
    data.loc[valid_bus & (data['nearest_bus_distance'] >= 200) & (data['nearest_bus_distance'] < 400), 'transit_score'] += 2
    data.loc[valid_bus & (data['nearest_bus_distance'] >= 400) & (data['nearest_bus_distance'] < 800), 'transit_score'] += 1
    
    # Add points for multiple bus stops
    data.loc[valid_bus, 'transit_score'] += data.loc[valid_bus, 'bus_stops_500m'].clip(upper=2)
    
    # Create transit quality categories
    data['transit_quality'] = pd.cut(
        data['transit_score'], 
        bins=[-1, 2, 5, 8, 20], 
        labels=['poor', 'average', 'good', 'excellent']
    )
    
    # Create binary features
    data['near_subway'] = (valid_subway & (data['nearest_subway_distance'] < 1000)).astype(int)
    data['near_bus'] = (valid_bus & (data['nearest_bus_distance'] < 400)).astype(int)
    data['near_public_transit'] = ((data['near_subway'] == 1) | (data['near_bus'] == 1)).astype(int)
    
    return data

# def process_in_parallel(combined_data, subway_data, bus_data, n_jobs=None):
#     """Process transportation features using parallel computing with improved error handling
    
#     Args:
#         combined_data: DataFrame containing housing data
#         subway_data: DataFrame containing subway station information
#         bus_data: DataFrame containing bus stop information
#         n_jobs: Number of parallel jobs to run
        
#     Returns:
#         DataFrame with added transportation features
#     """
#     print("Adding transportation features with parallel processing...")
#     start_time = time.time()
    
#     # Determine number of CPU cores to use
#     if n_jobs is None:
#         n_jobs = max(1, os.cpu_count() - 1)  # Use all cores except one
    
#     print(f"Using {n_jobs} CPU cores")
    
#     # Make a copy of the data
#     result = combined_data.copy()
    
#     # Skip if coordinate columns don't exist
#     lat_col, lon_col = '좌표Y', '좌표X'
#     if not all(col in result.columns for col in [lat_col, lon_col]):
#         print("Coordinate columns not found. Skipping transportation features.")
#         return result
    
#     # Filter out properties with missing coordinates
#     valid_mask = ~(result[lat_col].isna() | result[lon_col].isna())
#     valid_data = result[valid_mask].copy()  # Create a copy to ensure index integrity
    
#     if len(valid_data) == 0:
#         print("No valid coordinates found in housing data")
#         return result
    
#     # Determine ideal chunk size for better load balancing
#     n_chunks = min(n_jobs * 2, 20)  # Cap at 20 chunks to avoid too much overhead
#     if len(valid_data) < n_chunks:
#         n_chunks = 1
    
#     # Split data into chunks with equal size but potentially different index ranges
#     chunk_size = len(valid_data) // n_chunks
#     chunks = []
    
#     for i in range(n_chunks):
#         start_idx = i * chunk_size
#         end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(valid_data)
#         chunk = valid_data.iloc[start_idx:end_idx].copy()  # Create a copy with reset index
#         chunks.append(chunk)
    
#     print(f"Processing data in {n_chunks} chunks of approximately {chunk_size} rows each")
    
#     # Prepare arguments for each chunk
#     chunk_args = [(chunk, subway_data, bus_data, i+1, n_chunks) for i, chunk in enumerate(chunks)]
    
#     # Process chunks in parallel
#     processed_chunks = []
    
#     with ProcessPoolExecutor(max_workers=n_jobs) as executor:
#         # Import the function here instead of at the module level
#         from transport_features_module import process_parallel_chunk
#         results = list(executor.map(process_parallel_chunk, chunk_args))
#         processed_chunks = [chunk for chunk in results if chunk is not None]
    
#     # Combine processed chunks
#     if processed_chunks:
#         try:
#             # First identify the new transportation features
#             sample_chunk = processed_chunks[0]
#             transport_features = [col for col in sample_chunk.columns if col not in combined_data.columns]
            
#             print(f"Found {len(transport_features)} transportation features to combine")
            
#             # Initialize new columns in the result DataFrame (before any assignment)
#             for col in transport_features:
#                 if col not in result.columns:
#                     # Initialize with appropriate default based on column type
#                     if sample_chunk[col].dtype == 'object' or sample_chunk[col].dtype.name == 'category':
#                         result[col] = ''  # For string/categorical columns
#                     else:
#                         result[col] = 0   # For numeric columns
            
#             # Now process one chunk at a time to avoid memory issues
#             print("Merging chunk results back to main DataFrame...")
#             for i, chunk in enumerate(processed_chunks):
#                 print(f"Processing chunk {i+1}/{len(processed_chunks)}")
                
#                 # Get the indices from this chunk
#                 chunk_indices = chunk.index
                
#                 # Process one column at a time
#                 for col in transport_features:
#                     # Skip columns that don't exist in this chunk
#                     if col not in chunk.columns:
#                         continue
                        
#                     # For categorical/string columns, use direct assignment with fillna
#                     if chunk[col].dtype == 'object' or chunk[col].dtype.name == 'category':
#                         result.loc[chunk_indices, col] = chunk[col].fillna('')
#                     else:
#                         # For numeric columns, use direct assignment with fillna(0)
#                         result.loc[chunk_indices, col] = chunk[col].fillna(0)
            
#             print("Successfully merged all chunks")
            
#             # Fill any remaining missing values for consistency
#             # For categorical columns
#             for col in ['nearest_subway_name', 'nearest_subway_line', 'nearest_bus_name', 'nearest_bus_type', 'transit_quality']:
#                 if col in result.columns:
#                     result[col] = result[col].fillna('')
            
#             # For numeric columns
#             for col in ['subway_stations_500m', 'subway_stations_1km', 'subway_stations_3km', 
#                          'bus_stops_300m', 'bus_stops_500m', 'bus_stops_1km', 
#                          'transit_score', 'near_subway', 'near_bus', 'near_public_transit']:
#                 if col in result.columns:
#                     result[col] = result[col].fillna(0)
                        
#         except Exception as e:
#             import traceback
#             print(f"Error combining processed chunks: {str(e)}")
#             print(traceback.format_exc())
#             print("Falling back to sequential processing...")
            
#             # Fall back to sequential processing
#             from transport_features_module import add_transportation_features_efficient
#             result = add_transportation_features_efficient(combined_data, subway_data, bus_data)
    
    
#     elapsed_time = time.time() - start_time
#     print(f"Transportation features added successfully in {elapsed_time:.2f} seconds")
    
#     return result    

def process_in_parallel(combined_data, subway_data, bus_data, n_jobs=None, cache_dir='./cache/transportation_features'):
    """Process transportation features using optimized parallel computing."""
    print("Adding transportation features with optimized parallel processing...")
    start_time = time.time()
    
    # Determine number of CPU cores to use
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)  # Use all cores except one
    
    print(f"Using {n_jobs} CPU cores")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache paths for each chunk
    chunks_cache_dir = os.path.join(cache_dir, 'chunks')
    os.makedirs(chunks_cache_dir, exist_ok=True)
    
    # Make a copy of the data
    result = combined_data.copy()
    
    # Skip if coordinate columns don't exist
    lat_col, lon_col = '좌표Y', '좌표X'
    if not all(col in result.columns for col in [lat_col, lon_col]):
        print("Coordinate columns not found. Skipping transportation features.")
        return result
    
    # Filter out properties with missing coordinates
    valid_mask = ~(result[lat_col].isna() | result[lon_col].isna())
    valid_data = result[valid_mask].copy()  # Create a copy to ensure index integrity
    
    if len(valid_data) == 0:
        print("No valid coordinates found in housing data")
        return result
    
    # Build or load spatial indices once for all chunks
    from transport_features_module import create_spatial_index
    
    # Try to load precomputed indices
    subway_cache = os.path.join(cache_dir, 'subway_tree.pkl')
    bus_cache = os.path.join(cache_dir, 'bus_tree.pkl')
    
    subway_tree = None
    bus_tree = None
    
    # Try to load cached indices
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
        print(f"Saving subway spatial index to {subway_cache}")
        with open(subway_cache, 'wb') as f:
            pickle.dump(subway_tree, f)
    
    if bus_tree is None and len(bus_data) > 0:
        print(f"Building spatial index for {len(bus_data)} bus stops...")
        bus_coords = np.radians(bus_data[['Y좌표', 'X좌표']].values)
        bus_tree = BallTree(bus_coords, metric='haversine')
        
        # Cache the index
        print(f"Saving bus spatial index to {bus_cache}")
        with open(bus_cache, 'wb') as f:
            pickle.dump(bus_tree, f)
    
    # Optimize chunk size based on data size
    if len(valid_data) > 1000000:
        chunk_size = 50000  # Smaller chunks for very large datasets
    elif len(valid_data) > 500000:
        chunk_size = 25000  # Medium chunks for large datasets
    else:
        chunk_size = max(5000, len(valid_data) // (n_jobs * 4))  # Balanced chunks for smaller datasets
    
    # Calculate number of chunks
    n_chunks = (len(valid_data) + chunk_size - 1) // chunk_size
    
    print(f"Processing {len(valid_data)} rows in {n_chunks} chunks of approximately {chunk_size} rows each")
    
    # Create chunks with more balanced distribution
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(valid_data))
        chunk = valid_data.iloc[start_idx:end_idx].copy()
        chunks.append(chunk)

    # Define parallel processing function with caching
    def process_chunk_with_cache(chunk_data, chunk_id):
        """Process a chunk with caching support"""
        chunk_cache_path = os.path.join(chunks_cache_dir, f'chunk_{chunk_id}.pkl')
        
        # Try to load cached chunk result
        if os.path.exists(chunk_cache_path):
            try:
                with open(chunk_cache_path, 'rb') as f:
                    print(f"Loading cached result for chunk {chunk_id}")
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cached chunk {chunk_id}: {e}. Will reprocess.")
        
        try:
            print(f"Processing chunk {chunk_id}/{n_chunks}")
            # Process the chunk using the prebuilt spatial indices
            from transport_features_module import add_transportation_features_efficient
            result = add_transportation_features_efficient(
                chunk_data, 
                subway_data, 
                bus_data, 
                subway_tree=subway_tree, 
                bus_tree=bus_tree
            )
            
            # Cache the result
            with open(chunk_cache_path, 'wb') as f:
                pickle.dump(result, f)
                
            print(f"Chunk {chunk_id}/{n_chunks} completed and cached")
            return result
        except Exception as e:
            import traceback
            print(f"Error in chunk {chunk_id}/{n_chunks}: {str(e)}")
            print(traceback.format_exc())
            # Return the original chunk to avoid data loss
            return chunk_data
    
    # Process chunks in parallel with caching
    processed_chunks = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_chunk_with_cache, chunk, i+1): i 
            for i, chunk in enumerate(chunks)
        }
        
        # Process results as they complete
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_result = future.result()
                processed_chunks.append(chunk_result)
                print(f"Chunk {chunk_idx+1}/{n_chunks} successfully processed")
            except Exception as e:
                print(f"Chunk {chunk_idx+1} processing failed: {e}")
    
    # Combine processed chunks
    if processed_chunks:
        try:
            # Identify the new transportation features
            sample_chunk = processed_chunks[0]
            transport_features = [col for col in sample_chunk.columns if col not in combined_data.columns]
            
            print(f"Found {len(transport_features)} transportation features to combine")
            
            # Initialize new columns
            for col in transport_features:
                if col not in result.columns:
                    # Initialize with appropriate default based on column type
                    if sample_chunk[col].dtype == 'object' or sample_chunk[col].dtype.name == 'category':
                        result[col] = ''  # For string/categorical columns
                    else:
                        result[col] = 0   # For numeric columns
            
            # Merge chunks back - use concat which is more memory efficient
            all_processed = pd.concat(processed_chunks)
            
            # Now merge back to the main dataframe
            for col in transport_features:
                # Use bulk assignment by index
                result.loc[all_processed.index, col] = all_processed[col]
            
            # Cache the final combined result
            results_cache = os.path.join(cache_dir, f'transport_features_rows{len(combined_data)}.pkl')
            print(f"Saving combined results to {results_cache}")
            result.to_pickle(results_cache)
            
        except Exception as e:
            import traceback
            print(f"Error combining processed chunks: {str(e)}")
            print(traceback.format_exc())
            print("Falling back to sequential processing...")
            
            # Fall back to sequential processing
            from transport_features_module import add_transportation_features_efficient
            result = add_transportation_features_efficient(combined_data, subway_data, bus_data)
    
    elapsed_time = time.time() - start_time
    print(f"Transportation features added successfully in {elapsed_time:.2f} seconds")
    
    return result


def process_in_parallel_with_chunking(combined_data, subway_data, bus_data, n_jobs=None, chunk_size=None):
    """
    Process transportation features using parallel computing with chunking for memory efficiency
    """
    print("Adding transportation features with parallel processing and chunking...")
    start_time = time.time()
    
    # Determine number of CPU cores to use
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)
    
    print(f"Using {n_jobs} CPU cores")
    
    # Determine chunk size based on data size
    if chunk_size is None:
        if len(combined_data) > 1000000:
            chunk_size = 200000
        elif len(combined_data) > 500000:
            chunk_size = 100000
        else:
            chunk_size = 50000
    
    print(f"Processing in chunks of {chunk_size} rows")
    
    # Calculate number of chunks
    n_chunks = (len(combined_data) + chunk_size - 1) // chunk_size
    
    # Process chunks sequentially to avoid memory problems
    result_chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(combined_data))
        
        chunk = combined_data.iloc[start_idx:end_idx].copy()
        print(f"Processing chunk {i+1}/{n_chunks} (rows {start_idx} to {end_idx})")
        
        # Process this chunk in parallel
        chunk_result = process_in_parallel(chunk, subway_data, bus_data, n_jobs)
        result_chunks.append(chunk_result)
    
    # Combine results
    result = pd.concat(result_chunks)
    
    elapsed_time = time.time() - start_time
    print(f"Transportation features added successfully in {elapsed_time:.2f} seconds")
    
    return result

def add_transportation_features_parallel(combined_data, subway_data, bus_data, n_jobs=None, 
                                        lat_col='좌표Y', lon_col='좌표X',
                                        subway_lat_col='위도', subway_lon_col='경도',
                                        bus_lat_col='Y좌표', bus_lon_col='X좌표',
                                        use_cache=True, cache_dir='./cache'):
    """Add transportation features using parallel processing and precomputed indices"""
    print("Adding transportation features with parallel processing...")
    start_time = time.time()
    
    # Determine number of CPU cores to use
    if n_jobs is None:
        n_jobs = os.cpu_count() - 1 or 1  # Use all cores except one
    
    print(f"Using {n_jobs} CPU cores")
    
    # Try to load precomputed indices
    if use_cache:
        subway_index, bus_index = load_indices(cache_dir)
        
        # If indices not found, build and save them
        if subway_index is None or bus_index is None:
            subway_index_path, bus_index_path = build_and_save_indices(
                subway_data, bus_data, 
                subway_lat_col, subway_lon_col, 
                bus_lat_col, bus_lon_col,
                cache_dir
            )
            subway_index, bus_index = load_indices(cache_dir)
    else:
        # Build indices but don't save
        print(f"Building spatial index for {len(subway_data)} subway stations...")
        subway_coords = np.radians(subway_data[[subway_lat_col, subway_lon_col]].values)
        subway_tree = BallTree(subway_coords, metric='haversine')
        
        print(f"Building spatial index for {len(bus_data)} bus stops...")
        bus_coords = np.radians(bus_data[[bus_lat_col, bus_lon_col]].values)
        bus_tree = BallTree(bus_coords, metric='haversine')
        
        subway_index = (subway_tree, subway_data[['역사명', '호선']])
        bus_index = (bus_tree, bus_data[['정류소명', '정류소 타입']])
    
    # Extract trees and info
    subway_tree, subway_info = subway_index
    bus_tree, bus_info = bus_index
    
    # Split data into chunks for parallel processing
    n_chunks = n_jobs * 4  # Create more chunks than workers for better load balancing
    chunks = np.array_split(combined_data, n_chunks)
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(
            process_chunk,
            chunks,
            [subway_tree] * n_chunks,
            [subway_info] * n_chunks,
            [bus_tree] * n_chunks,
            [bus_info] * n_chunks,
            [lat_col] * n_chunks,
            [lon_col] * n_chunks
        ))
    
    # Combine results
    result_data = pd.concat(results)
    
    elapsed_time = time.time() - start_time
    print(f"Transportation features added successfully in {elapsed_time:.2f} seconds")
    
    return result_data

def process_in_parallel_with_chunking(combined_data, subway_data, bus_data, n_jobs=None, chunk_size=None):
    """
    Process transportation features using parallel computing with chunking for memory efficiency
    """
    print("Adding transportation features with parallel processing and chunking...")
    start_time = time.time()
    
    # Determine number of CPU cores to use
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)
    
    print(f"Using {n_jobs} CPU cores")
    
    # Determine chunk size based on data size
    if chunk_size is None:
        if len(combined_data) > 1000000:
            chunk_size = 200000
        elif len(combined_data) > 500000:
            chunk_size = 100000
        else:
            chunk_size = 50000
    
    print(f"Processing in chunks of {chunk_size} rows")
    
    # Calculate number of chunks
    n_chunks = (len(combined_data) + chunk_size - 1) // chunk_size
    
    # Process chunks sequentially to avoid memory problems
    result_chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(combined_data))
        
        chunk = combined_data.iloc[start_idx:end_idx].copy()
        print(f"Processing chunk {i+1}/{n_chunks} (rows {start_idx} to {end_idx})")
        
        # Process this chunk in parallel
        chunk_result = process_in_parallel(chunk, subway_data, bus_data, n_jobs)
        result_chunks.append(chunk_result)
    
    # Combine results
    result = pd.concat(result_chunks)
    
    elapsed_time = time.time() - start_time
    print(f"Transportation features added successfully in {elapsed_time:.2f} seconds")
    
    return result
    