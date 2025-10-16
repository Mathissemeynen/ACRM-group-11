import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

def load_all_data():
    """Load all necessary data files"""
    print("Loading all data files...")

    # Load trips data
    trips_folder = "data/Trips"
    trips_files = glob.glob(os.path.join(trips_folder, '*.csv'))

    all_trips = []
    for file in trips_files:
        df = pd.read_csv(file, sep=';')
        df['file_source'] = os.path.basename(file)
        all_trips.append(df)

    trips = pd.concat(all_trips, ignore_index=True)
    print(f"Loaded {len(trips):,} trip records")

    # Load other datasets
    stations = pd.read_csv("data/stations.csv")
    travelers = pd.read_csv("data/travelers.csv")
    travelers = fix_travelers_data()
    incidents = pd.read_csv("data/incidents.csv")

    print(f"Loaded {len(stations)} stations, {len(travelers)} traveler records, {len(incidents)} incidents")

    # Debug: Check column names
    print(f"Travelers columns: {list(travelers.columns)}")
    print(f"Stations columns: {list(stations.columns)}")

    return trips, stations, travelers, incidents

def preprocess_data(trips, stations, travelers, incidents):
    """Preprocess all datasets"""
    print("Preprocessing data...")

    # Preprocess trips
    trips['planned_departure_datetime'] = pd.to_datetime(
        trips['Planned departure date'] + ' ' + trips['Planned departure time'], errors='coerce'
    )
    trips['departure_hour'] = trips['planned_departure_datetime'].dt.hour
    trips['departure_dayofweek'] = trips['planned_departure_datetime'].dt.day_name()
    trips['departure_delay'] = trips['Delay at departure'] / 60.0  # Convert seconds to minutes

    # Define peak hours (based on our analysis)
    morning_peak = [7, 8, 9]  # 7-9 AM
    evening_peak = [16, 17, 18]  # 4-6 PM

    trips['is_peak'] = trips['departure_hour'].isin(morning_peak + evening_peak)
    trips['is_weekday'] = ~trips['planned_departure_datetime'].dt.dayofweek.isin([5, 6])

    print("Data preprocessing complete")
    return trips, morning_peak, evening_peak

def calculate_basic_delay_metrics(trips):
    """Calculate basic delay metrics per station"""
    print("Calculating basic delay metrics...")

    # Filter for peak hours on weekdays (when bottlenecks matter most)
    peak_weekday_trips = trips[(trips['is_peak']) & (trips['is_weekday'])]

    delay_metrics = peak_weekday_trips.groupby('Stopping place').agg({
        'departure_delay': ['count', 'sum', 'mean'],
        'Train number': 'nunique'
    }).round(2)

    # Flatten column names
    delay_metrics.columns = ['total_trains', 'total_delay_minutes', 'avg_delay_minutes', 'unique_trains']

    # Calculate severe delays
    severe_delays_5min = peak_weekday_trips[peak_weekday_trips['departure_delay'] > 5].groupby('Stopping place').size()
    severe_delays_15min = peak_weekday_trips[peak_weekday_trips['departure_delay'] > 15].groupby('Stopping place').size()

    delay_metrics['delays_above_5min'] = severe_delays_5min
    delay_metrics['delays_above_15min'] = severe_delays_15min
    delay_metrics['pct_delays_above_5min'] = (delay_metrics['delays_above_5min'] / delay_metrics['total_trains'] * 100).round(2)
    delay_metrics['pct_delays_above_15min'] = (delay_metrics['delays_above_15min'] / delay_metrics['total_trains'] * 100).round(2)

    # Fill NaN values with 0
    delay_metrics = delay_metrics.fillna(0)

    print(f"Calculated metrics for {len(delay_metrics)} stations")
    return delay_metrics

def normalize_with_travelers(delay_metrics, travelers):
    """Normalize delay metrics by passenger traffic"""
    print("Normalizing with traveler data...")

    # Reset index to make 'Stopping place' a column
    delay_metrics = delay_metrics.reset_index()

    # Check the actual column name in travelers
    traveler_station_col = 'STATION' if 'STATION' in travelers.columns else 'station'
    if traveler_station_col not in travelers.columns:
        # Try to find the station column
        possible_names = ['STATION', 'station', 'Station', 'NAME', 'name']
        for name in possible_names:
            if name in travelers.columns:
                traveler_station_col = name
                break
        else:
            print("Could not find station column in travelers data")
            print(f"Travelers columns: {list(travelers.columns)}")
            return delay_metrics

    print(f"Using '{traveler_station_col}' as station column in travelers data")

    # Simple merge - we might need to handle name variations later
    merged = pd.merge(delay_metrics, travelers, left_on='Stopping place', right_on=traveler_station_col, how='left')

    # Check which traveler column to use for normalization
    traveler_cols = [col for col in travelers.columns if 'TRAVELER' in col.upper() or 'PASSENGER' in col.upper()]
    if traveler_cols:
        traveler_col = traveler_cols[0]
        print(f"Using '{traveler_col}' for passenger count")

        # Calculate delay per traveler
        merged['avg_weekday_travelers'] = merged[traveler_col]
        merged['delay_per_1000_travelers'] = (merged['total_delay_minutes'] / merged['avg_weekday_travelers'] * 1000).round(2)
        merged['trains_per_1000_travelers'] = (merged['total_trains'] / merged['avg_weekday_travelers'] * 1000).round(2)
    else:
        print("Could not find traveler count column")
        merged['avg_weekday_travelers'] = 0
        merged['delay_per_1000_travelers'] = 0
        merged['trains_per_1000_travelers'] = 0

    # Handle infinite values from division by zero
    merged = merged.replace([np.inf, -np.inf], 0)
    merged = merged.fillna(0)

    print("Normalization complete")
    return merged

def build_route_graph(trips):
    """Build a directed graph of train routes"""
    print("Building route graph...")

    # For efficiency, sample the data if it's too large
    if len(trips) > 100000:
        sample_trips = trips.sample(50000, random_state=42)
        print("Sampling 50,000 trips for graph construction")
    else:
        sample_trips = trips

    # Group by train number and date to get sequences of stops
    trip_sequences = sample_trips.groupby(['Train number', 'Date of departure']).apply(
        lambda x: x.sort_values('planned_departure_datetime')['Stopping place'].tolist()
    ).reset_index(name='station_sequence')

    # Create edges between consecutive stations
    edges = []
    for sequence in trip_sequences['station_sequence']:
        for i in range(len(sequence) - 1):
            edges.append((sequence[i], sequence[i+1]))

    # Count frequency of each edge
    from collections import Counter
    edge_counts = Counter(edges)

    print(f"Created graph with {len(edge_counts)} unique edges")
    return edge_counts

def calculate_centrality(edge_counts):
    """Calculate betweenness centrality for stations"""
    print("Calculating centrality metrics...")

    try:
        import networkx as nx

        # Create directed graph
        G = nx.DiGraph()

        # Add weighted edges
        for (source, target), weight in edge_counts.items():
            G.add_edge(source, target, weight=weight)

        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(G, weight='weight')

        print(f"Calculated centrality for {len(centrality)} stations")
        return centrality

    except ImportError:
        print("NetworkX not available, using simple degree centrality")
        # Fallback: use simple degree centrality
        station_degrees = {}
        for (source, target), weight in edge_counts.items():
            station_degrees[source] = station_degrees.get(source, 0) + weight
            station_degrees[target] = station_degrees.get(target, 0) + weight

        # Normalize
        max_degree = max(station_degrees.values()) if station_degrees else 1
        centrality = {station: degree/max_degree for station, degree in station_degrees.items()}

        print(f"Calculated fallback centrality for {len(centrality)} stations")
        return centrality

def main():
    """Main analysis function"""
    print("Starting bottleneck analysis...")

    # Step 1: Load data
    trips, stations, travelers, incidents = load_all_data()

    # Step 2: Preprocess
    trips, morning_peak, evening_peak = preprocess_data(trips, stations, travelers, incidents)

    # Step 3: Basic delay metrics
    delay_metrics = calculate_basic_delay_metrics(trips)

    # Step 4: Normalize with travelers
    normalized_metrics = normalize_with_travelers(delay_metrics, travelers)

    # Step 5: Build graph and calculate centrality
    edge_counts = build_route_graph(trips)
    centrality = calculate_centrality(edge_counts)

    # Add centrality to metrics
    centrality_df = pd.DataFrame(list(centrality.items()), columns=['Stopping place', 'betweenness_centrality'])
    final_metrics = pd.merge(normalized_metrics, centrality_df, on='Stopping place', how='left').fillna(0)

    # Display preliminary results
    print("\n" + "="*50)
    print("PRELIMINARY BOTTLENECK ANALYSIS RESULTS")
    print("="*50)

    # Top 10 stations by total delay
    print("\nTop 10 stations by total delay (peak hours, weekdays):")
    top_delay = final_metrics.nlargest(10, 'total_delay_minutes')[['Stopping place', 'total_delay_minutes', 'avg_delay_minutes', 'total_trains']]
    print(top_delay.to_string(index=False))

    # Top 10 stations by delay per traveler (if we have traveler data)
    if 'delay_per_1000_travelers' in final_metrics.columns and final_metrics['delay_per_1000_travelers'].sum() > 0:
        print("\nTop 10 stations by delay per 1000 travelers:")
        top_delay_per_traveler = final_metrics[final_metrics['avg_weekday_travelers'] > 0].nlargest(10, 'delay_per_1000_travelers')[['Stopping place', 'delay_per_1000_travelers', 'avg_weekday_travelers', 'total_delay_minutes']]
        print(top_delay_per_traveler.to_string(index=False))

    # Top 10 stations by betweenness centrality
    print("\nTop 10 stations by betweenness centrality:")
    top_centrality = final_metrics.nlargest(10, 'betweenness_centrality')[['Stopping place', 'betweenness_centrality', 'total_trains']]
    print(top_centrality.to_string(index=False))

    # Save results for next steps
    final_metrics.to_csv('bottleneck_metrics_step1.csv', index=False)
    print(f"\nDetailed metrics saved to 'bottleneck_metrics_step1.csv'")

    print(f"\nPeak hours defined as: Morning {morning_peak[0]}-{morning_peak[-1]+1}:00, Evening {evening_peak[0]}-{evening_peak[-1]+1}:00")
    print("Next steps: Z-score normalization, propagation filtering, and incident filtering")

    return final_metrics

def fix_travelers_data():
    """Fix the malformed travelers.csv file"""
    with open("data/travelers.csv", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # The first line contains all column names separated by semicolons
    # Let's split them properly and rewrite the file
    header_line = lines[0].strip().strip(';')
    headers = [h.strip() for h in header_line.split(';') if h.strip()]

    # Create proper CSV content
    new_lines = [','.join(headers) + '\n']  # New header with commas

    # Process the data lines
    for line in lines[1:]:
        data_line = line.strip().strip(';')
        data_values = [v.strip() for v in data_line.split(';') if v.strip()]
        if len(data_values) == len(headers):
            new_lines.append(','.join(data_values) + '\n')

    # Write the fixed file
    with open("data/travelers_fixed.csv", 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Fixed travelers data: {len(headers)} columns, {len(new_lines)-1} stations")
    return pd.read_csv("data/travelers_fixed.csv")

if __name__ == "__main__":
    results = main()