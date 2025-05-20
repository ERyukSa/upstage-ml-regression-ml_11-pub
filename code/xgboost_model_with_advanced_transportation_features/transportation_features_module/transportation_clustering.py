
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians
from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial import ConvexHull


def calculate_hub_area(hub_points, buffer_distance=100):
    """Calculate approximate area of a hub using convex hull"""
    # Need at least 3 points for convex hull
    if len(hub_points) < 3:
        # Use buffer around points instead
        return len(hub_points) * np.pi * (buffer_distance ** 2)
    
    try:
        # Calculate convex hull
        hull = ConvexHull(hub_points)
        return hull.volume  # Actually area in 2D
    except:
        # Fallback if convex hull fails
        return len(hub_points) * np.pi * (buffer_distance ** 2)

def identify_transport_hubs(subway_data, bus_data, eps=500, min_samples=5, cache_dir='./cache/transportation_features', use_cache=True):
    """
    Identify transportation hubs by clustering subway stations and bus stops
    
    Args:
        subway_data (DataFrame): Subway station data with locations
        bus_data (DataFrame): Bus stop data with locations
        eps (float): Maximum distance (in meters) between two samples for them to be considered in the same cluster
        min_samples (int): Minimum number of samples in a neighborhood for a point to be considered a core point
        
    Returns:
        tuple: (clustered_transport_data, hub_centers)
    """
    print("\n===== IDENTIFYING TRANSPORTATION HUBS =====")
    
    # Combine subway and bus coordinates
    subway_coords = subway_data[['위도', '경도']].rename(columns={'위도': 'latitude', '경도': 'longitude'})
    subway_coords['type'] = 'subway'
    subway_coords['name'] = subway_data['역사명']
    subway_coords['line'] = subway_data['호선']
    
    # FIXED: Bus data coordinate mapping
    # In most Korean coordinate systems, X좌표 is longitude (east-west) and Y좌표 is latitude (north-south)
    bus_coords = bus_data[['Y좌표', 'X좌표']].rename(columns={'Y좌표': 'latitude', 'X좌표': 'longitude'})
    bus_coords['type'] = 'bus'
    bus_coords['name'] = bus_data['정류소명']
    bus_coords['line'] = bus_data['정류소 타입']
    
    # Check coordinate ranges to detect potential swapping
    print("\nCoordinate range check:")
    print(f"Subway latitude range: {subway_coords['latitude'].min():.6f} to {subway_coords['latitude'].max():.6f}")
    print(f"Subway longitude range: {subway_coords['longitude'].min():.6f} to {subway_coords['longitude'].max():.6f}")
    print(f"Bus latitude range: {bus_coords['latitude'].min():.6f} to {bus_coords['latitude'].max():.6f}")
    print(f"Bus longitude range: {bus_coords['longitude'].min():.6f} to {bus_coords['longitude'].max():.6f}")
    
    # Detect if coordinates appear to be swapped
    lat_range_subway = (36.5, 38.5)  # Expected range for Seoul/Incheon latitude
    lon_range_seoul = (126.5, 127.5)  # Expected range for Seoul/Incheon longitude
    
    # Check if bus coordinates might be swapped
    if (bus_coords['latitude'].mean() > 100 or 
        bus_coords['longitude'].mean() < 100 or
        bus_coords['latitude'].mean() > subway_coords['latitude'].mean() + 50):
        print("\nWARNING: Bus coordinates appear to be swapped!")
        print("Automatically fixing the coordinate mapping...")
        
        # Swap the coordinates
        bus_coords = bus_data[['Y좌표', 'X좌표']].rename(columns={'X좌표': 'latitude', 'Y좌표': 'longitude'})
        bus_coords['type'] = 'bus'
        bus_coords['name'] = bus_data['정류소명']
        bus_coords['line'] = bus_data['정류소 타입']
        
        print("\nAfter correction:")
        print(f"Bus latitude range: {bus_coords['latitude'].min():.6f} to {bus_coords['latitude'].max():.6f}")
        print(f"Bus longitude range: {bus_coords['longitude'].min():.6f} to {bus_coords['longitude'].max():.6f}")
    
    # Combine all coordinates
    all_coords = pd.concat([subway_coords, bus_coords], ignore_index=True)
    
    # Convert to radians for clustering
    X = np.radians(all_coords[['latitude', 'longitude']].values)
    
    # Apply DBSCAN clustering
    # Note: We use haversine metric with Earth radius for geographic clustering
    # eps is converted from meters to radians (divide by Earth's radius)
    earth_radius = 6371000  # meters
    eps_rad = eps / earth_radius  # Convert eps from meters to radians
    
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine', algorithm='ball_tree')
    all_coords['cluster'] = db.fit_predict(X)
    
    # Count clusters (excluding noise which is labeled as -1)
    n_clusters = len(set(all_coords['cluster'])) - (1 if -1 in all_coords['cluster'] else 0)
    n_noise = list(all_coords['cluster']).count(-1)
    
    print(f"Identified {n_clusters} transportation hubs")
    print(f"Noise points (isolated stations/stops): {n_noise}")
    
    # Analyze the clusters
    cluster_summary = all_coords[all_coords['cluster'] >= 0].groupby('cluster').agg({
        'type': ['count', lambda x: sum(x == 'subway'), lambda x: sum(x == 'bus')],
        'latitude': 'mean',
        'longitude': 'mean'
    })
    
    cluster_summary.columns = ['total_stations', 'subway_count', 'bus_count', 'center_lat', 'center_lon']
    cluster_summary = cluster_summary.sort_values('total_stations', ascending=False)
    
    # Print with proper formatting to avoid confusion
    print("\nTop 5 transportation hubs:")
    print(cluster_summary.head(5).to_string(float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))
    
    # Create hub centers DataFrame
    hub_centers = pd.DataFrame({
        'hub_id': cluster_summary.index,
        'latitude': cluster_summary['center_lat'],
        'longitude': cluster_summary['center_lon'],
        'total_stations': cluster_summary['total_stations'],
        'subway_count': cluster_summary['subway_count'],
        'bus_count': cluster_summary['bus_count']
    })
    
    # Add hub size categorization
    hub_centers['hub_size'] = pd.cut(
        hub_centers['total_stations'],
        bins=[0, 5, 10, 20, 100],
        labels=['small', 'medium', 'large', 'major']
    )
    
    # Add hub type (subway-dominant, bus-dominant, mixed)
    hub_centers['subway_ratio'] = hub_centers['subway_count'] / hub_centers['total_stations']
    
    conditions = [
        (hub_centers['subway_ratio'] > 0.7),
        (hub_centers['subway_ratio'] < 0.3),
        (True)  # catchall
    ]
    choices = ['subway-dominant', 'bus-dominant', 'mixed']
    hub_centers['hub_type'] = np.select(conditions, choices, default='mixed')
    
    # Visualize the clusters (top 20)
    plt.figure(figsize=(12, 10))
    
    # Plot noise points
    noise = all_coords[all_coords['cluster'] == -1]
    plt.scatter(noise['longitude'], noise['latitude'], s=10, c='lightgray', label='Isolated stations')
    
    # Plot clusters
    clusters = all_coords[all_coords['cluster'] >= 0]
    top_clusters = clusters[clusters['cluster'].isin(cluster_summary.head(20).index)]
    
    # Plot each cluster with a different color
    for cluster_id, group in top_clusters.groupby('cluster'):
        plt.scatter(group['longitude'], group['latitude'], s=30, 
                   label=f'Hub {cluster_id} ({len(group)} stations)')
    
    # Annotate hub centers
    for idx, row in cluster_summary.head(10).iterrows():
        plt.annotate(f'Hub {idx}', 
                    (row['center_lon'], row['center_lat']),
                    fontsize=12,
                    xytext=(10, 5),
                    textcoords='offset points')
    
    plt.title('Transportation Hubs (Clustered Subway Stations and Bus Stops)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print("\n=== ANALYZING HUB DENSITY AND DISTRIBUTION ===")

    # Analyze each hub's density
    hub_densities = []
    for hub_id in hub_centers['hub_id'].unique():
        # Get stations in this hub
        hub_stations = all_coords[all_coords['cluster'] == hub_id]
        
        # Calculate approximate area in square km
        points = hub_stations[['longitude', 'latitude']].values
        area_sqm = calculate_hub_area(points)
        area_sqkm = area_sqm / 1_000_000  # Convert to sq km
        
        # Calculate density
        stations_per_sqkm = len(hub_stations) / max(0.01, area_sqkm)  # Avoid division by zero
        
        hub_densities.append({
            'hub_id': hub_id,
            'stations': len(hub_stations),
            'subway_stations': sum(hub_stations['type'] == 'subway'),
            'bus_stops': sum(hub_stations['type'] == 'bus'),
            'area_sqkm': area_sqkm,
            'density': stations_per_sqkm
        })

    hub_density_df = pd.DataFrame(hub_densities)
    print("\nHub Density Analysis:")
    print(hub_density_df.sort_values('density', ascending=False))

    # Check for duplicate stations
    print("\nChecking for potential duplicates in Hub 0...")
    largest_hub_id = hub_density_df.sort_values('stations', ascending=False).iloc[0]['hub_id']
    largest_hub_stations = all_coords[all_coords['cluster'] == largest_hub_id]
    subway_counts = largest_hub_stations[largest_hub_stations['type'] == 'subway']['name'].value_counts()
    bus_counts = largest_hub_stations[largest_hub_stations['type'] == 'bus']['name'].value_counts()

    print(f"Largest hub (ID: {largest_hub_id}) statistics:")
    print(f"- Number of uniquely named subway stations: {len(subway_counts)}")
    print(f"- Max occurrences of a single subway station name: {subway_counts.max() if not subway_counts.empty else 0}")
    print(f"- Number of uniquely named bus stops: {len(bus_counts)}")
    print(f"- Max occurrences of a single bus stop name: {bus_counts.max() if not bus_counts.empty else 0}")

    # Plot the distribution of stations for the largest hub
    plt.figure(figsize=(12, 10))
    plt.scatter(largest_hub_stations[largest_hub_stations['type'] == 'bus']['longitude'], 
                largest_hub_stations[largest_hub_stations['type'] == 'bus']['latitude'], 
                alpha=0.3, s=5, c='blue', label='Bus Stops')
    plt.scatter(largest_hub_stations[largest_hub_stations['type'] == 'subway']['longitude'], 
                largest_hub_stations[largest_hub_stations['type'] == 'subway']['latitude'], 
                alpha=1, s=20, c='red', label='Subway Stations')
    plt.title(f'Geographic Distribution of Stations in Largest Hub (ID: {largest_hub_id})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('largest_hub_distribution.png')
    plt.show()

    # Provide recommendations
    print("\nRecommendations based on analysis:")
    print("1. If the largest hub covers too large an area, try smaller eps value in DBSCAN (e.g., 300m instead of 500m)")
    print("2. Check for duplicate records in your transportation data")
    print("3. Consider using different parameters for subway vs. bus stations, as they typically have different spacing patterns")
    print("4. For more balanced clusters, try hierarchical clustering or HDBSCAN instead of DBSCAN")
    
    return all_coords, hub_centers




def add_hub_proximity_features(combined_data, hub_centers):
    """
    Add transportation hub proximity features to housing data
    
    Args:
        combined_data (DataFrame): Combined housing dataset
        hub_centers (DataFrame): Transportation hub centers
        
    Returns:
        DataFrame: Enhanced dataset with hub proximity features
    """
    print("\n===== ADDING HUB PROXIMITY FEATURES =====")
    
    # Make a copy to avoid modifying the original data
    data = combined_data.copy()
    
    # Check if we have location data in the housing dataset
    if not all(col in data.columns for col in ['위도', '경도']):
        print("Warning: Housing dataset doesn't have latitude and longitude. Cannot calculate hub distances.")
        return data
    
    # Add columns for nearest hub
    data['nearest_hub_id'] = np.nan
    data['nearest_hub_distance'] = np.nan
    data['nearest_hub_size'] = ''
    data['nearest_hub_type'] = ''
    data['nearest_major_hub_id'] = np.nan
    data['nearest_major_hub_distance'] = np.nan
    data['nearby_hub_count'] = 0
    
    # Convert hub centers to radians for haversine distance
    hub_coords = np.radians(hub_centers[['latitude', 'longitude']].values)
    
    # Get indices of major hubs (for separate calculation)
    major_hubs = hub_centers[hub_centers['hub_size'] == 'major'].index
    major_hub_indices = [hub_centers.index.get_loc(idx) for idx in major_hubs]
    major_hub_coords = hub_coords[major_hub_indices]
    
    # Calculate for each property
    for idx, row in data.iterrows():
        if pd.isna(row['위도']) or pd.isna(row['경도']):
            continue
            
        # Property coordinates
        property_coords = np.array([[row['위도'], row['경도']]])
        property_coords_rad = np.radians(property_coords)
        
        # Calculate distances to all hubs
        distances = haversine_distances(property_coords_rad, hub_coords) * 6371000  # Earth radius in meters
        distances = distances[0]  # Flatten to 1D array
        
        # Find nearest hub
        if len(distances) > 0:
            nearest_idx = np.argmin(distances)
            hub_id = hub_centers.iloc[nearest_idx]['hub_id']
            
            data.at[idx, 'nearest_hub_id'] = hub_id
            data.at[idx, 'nearest_hub_distance'] = distances[nearest_idx]
            data.at[idx, 'nearest_hub_size'] = hub_centers.iloc[nearest_idx]['hub_size']
            data.at[idx, 'nearest_hub_type'] = hub_centers.iloc[nearest_idx]['hub_type']
            
            # Count nearby hubs (within 2km)
            data.at[idx, 'nearby_hub_count'] = np.sum(distances < 2000)
            
            # Nearest major hub (if there are any)
            if len(major_hub_indices) > 0:
                # Calculate distances to major hubs only
                major_distances = haversine_distances(property_coords_rad, major_hub_coords) * 6371000
                major_distances = major_distances[0]
                
                if len(major_distances) > 0:
                    nearest_major_idx = np.argmin(major_distances)
                    major_hub_id = hub_centers.iloc[major_hub_indices[nearest_major_idx]]['hub_id']
                    
                    data.at[idx, 'nearest_major_hub_id'] = major_hub_id
                    data.at[idx, 'nearest_major_hub_distance'] = major_distances[nearest_major_idx]
    
    # Create hub proximity score
    data['hub_proximity_score'] = 0
    
    # Calculate score based on proximity to hubs and their sizes
    for idx, row in data.iterrows():
        score = 0
        
        # Add component based on nearest hub
        if not pd.isna(row['nearest_hub_distance']):
            # Size factor (larger hubs have more impact)
            size_factor = {
                'small': 1,
                'medium': 2,
                'large': 3,
                'major': 4
            }.get(row['nearest_hub_size'], 0)
            
            # Distance factor (closer is better)
            if row['nearest_hub_distance'] < 500:
                distance_factor = 3  # Very close
            elif row['nearest_hub_distance'] < 1000:
                distance_factor = 2  # Walking distance
            elif row['nearest_hub_distance'] < 2000:
                distance_factor = 1  # Moderate distance
            else:
                distance_factor = 0  # Too far
            
            # Hub type factor
            type_factor = {
                'subway-dominant': 1.5,  # Subway hubs often have better connectivity
                'mixed': 1.3,
                'bus-dominant': 1.0
            }.get(row['nearest_hub_type'], 1.0)
            
            # Nearby hub count bonus
            nearby_bonus = min(row['nearby_hub_count'], 3)  # Cap at 3 to avoid dominating the score
            
            # Calculate hub score
            score = size_factor * distance_factor * type_factor + nearby_bonus
        
        # Assign final score
        data.at[idx, 'hub_proximity_score'] = score
    
    # Create hub quality categories
    data['hub_quality'] = pd.cut(
        data['hub_proximity_score'], 
        bins=[-0.1, 2, 5, 8, 20], 
        labels=['poor', 'average', 'good', 'excellent']
    )
    
    # Create binary features
    data['near_major_hub'] = (data['nearest_major_hub_distance'] < 1500).astype(int)
    data['multiple_hubs_nearby'] = (data['nearby_hub_count'] > 1).astype(int)
    data['subway_dominant_area'] = (data['nearest_hub_type'] == 'subway-dominant').astype(int)
    
    print("Hub proximity features added successfully.")
    print(f"New features: {[col for col in data.columns if 'hub' in col]}")
    
    return data