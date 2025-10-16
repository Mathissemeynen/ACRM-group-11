import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from collections import Counter
from scipy import stats

# =============================================================================
# CONFIGURATION CONSTANTS - EASY TWEAKING
# =============================================================================

# Step 2: Basic Delay Metrics
PEAK_HOURS_MORNING = [7, 8, 9]      # 7-9 AM
PEAK_HOURS_EVENING = [16, 17, 18]   # 4-6 PM

# Step 4: Network Analysis
GRAPH_SAMPLE_SIZE = 50000           # For large datasets

# Step 6: Propagation Filter
MIN_ROUTE_TRIPS = 3                 # Lower from 5 to 3 (catch more routes)
MIN_STATION_OBSERVATIONS = 2        # Lower from 3 to 2 (more station-route pairs)
SYSTEMIC_Z_THRESHOLD = 1.2          # Increase from 1.0 to 1.2 (catch more systemic routes)
SYSTEMIC_DELAY_THRESHOLD = 1.5      # Lower from 2.0 to 1.5 (catch routes with moderate delays)
SIGNIFICANT_DELAY_Z_THRESHOLD = 1.2 # Lower from 1.5 to 1.2 (more stations considered "responsible")
RESPONSIBILITY_ADJUST_THRESHOLD = 0.9 # Increase from 0.8 to 0.9 (adjust more stations)
MIN_RESPONSIBILITY_FACTOR = 0.5     # Increase from 0.3 to 0.5 (stations keep more delay)
HUB_PROTECTION_FACTOR = 0.8         # Increase from 0.6 to 0.8 (major hubs keep even more)


# Step 5: Bottleneck Scoring (metrics to include)
BOTTLENECK_METRICS = [
    'total_delay_minutes',
    'pct_delays_above_5min',
    'degree_centrality',
    'total_trains',
    'delay_times_1000_travelers'
]

# Incident classification and removal factors
INCIDENT_CATEGORIES = {
    'EXTERNAL': [  # 90% removal - completely beyond NMBS control
        'suspicious package', 'intrusion into tracks', 'collision with person',
        'malicious act', 'fire near tracks', 'bomb alert', 'body in tracks',
        'obstacle in/near tracks', 'exceptional weather conditions', 'strike',
        'dangerous goods near tracks', 'cable theft', 'problem with passenger',
        'measures covid 19', 'it disturbance', 'collision with animal',
        'failure to supply electricity', 'high passenger flow'
    ],
    'INFRASTRUCTURE': [  # 70% removal - partially controllable
        'damage catenary', 'disturbance with signalling', 'damage rolling stock',
        'disturbance with switch', 'infrastructure disturbance', 'derailment',
        'error during maneuver', 'crossing of red signal'
    ],
    'OPERATIONAL': [  # 30% removal - mostly NMBS responsibility
        'urgent works', 'late completion of works', 'incident at work site',
        'staffing issue'
    ]
}

# Removal factors (adjustable)
EXTERNAL_REMOVAL_FACTOR = 0.9
INFRASTRUCTURE_REMOVAL_FACTOR = 0.7
OPERATIONAL_REMOVAL_FACTOR = 0.3
MIN_INCIDENT_DELAY = 100  # Ignore incidents below this threshold



def robust_fix_travelers_data():
    """Properly fix the travelers.csv with all stations"""
    print("Fixing travelers data...")

    with open("data/travelers.csv", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Fix the malformed header
    header_line = lines[0].strip().strip(';')
    headers = [h.strip() for h in header_line.split(';') if h.strip()]

    # Create proper CSV content
    new_lines = [','.join(headers) + '\n']

    # Process all data lines
    for line in lines[1:]:
        data_line = line.strip().strip(';')
        data_values = [v.strip() for v in data_line.split(';') if v.strip()]
        if len(data_values) == len(headers):
            new_lines.append(','.join(data_values) + '\n')

    # Write the fixed file
    with open("data/travelers_fixed.csv", 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    travelers = pd.read_csv("data/travelers_fixed.csv")
    print(f"✓ Fixed travelers data: {len(travelers)} stations")
    return travelers

def create_complete_travelers_dataset(travelers, stations, trips):
    """Create a complete travelers dataset by filling missing stations"""
    print("Creating complete travelers dataset...")

    # Get all unique stations from trips data
    all_stations_from_trips = trips['Stopping place'].unique()

    # Create a mapping from station names to reasonable estimates
    station_estimates = {}

    # Map station names between datasets
    name_mapping = {
        'BRUSSEL-CENTRAAL': 'Bruxelles-Central',
        'BRUSSEL-ZUID': 'Bruxelles-Midi',
        'BRUSSEL-NOORD': 'Bruxelles-Nord',
        'BRUSSEL-CONGRES': 'Bruxelles-Congrès',
        'BRUSSEL-KAPELLEKERK': 'Bruxelles-Chapelle',
        'SCHAARBEEK': 'Schaerbeek'
    }

    # Create estimates based on station importance
    for station in all_stations_from_trips:
        # Check if station exists in travelers
        in_travelers = travelers[travelers['Station'] == station]
        if len(in_travelers) > 0:
            continue  # Already exists

        # Try to find in stations.csv to get avg_stop_times as proxy
        stations_name = name_mapping.get(station, station)
        station_data = stations[stations['name'] == stations_name]

        if len(station_data) > 0:
            avg_stop_times = station_data['avg_stop_times'].iloc[0]
            # Use avg_stop_times as rough proxy (scale appropriately)
            estimated_travelers = max(avg_stop_times * 50, 100)  # Minimum 100
        else:
            # Default estimates for known important stations
            default_estimates = {
                'BRUSSEL-CENTRAAL': 80000,
                'BRUSSEL-ZUID': 75000,
                'BRUSSEL-NOORD': 60000,
                'BRUSSEL-CONGRES': 40000,
                'BRUSSEL-KAPELLEKERK': 35000,
                'SCHAARBEEK': 25000
            }
            estimated_travelers = default_estimates.get(station, 1000)  # Default 1000

        station_estimates[station] = estimated_travelers

    # Add missing stations to travelers data
    enhanced_travelers = travelers.copy()
    for station, estimate in station_estimates.items():
        new_row = {
            'Station': station,
            'Avg number of travelers in the week': estimate,
            'Avg number of travelers on Saturday': int(estimate * 0.3),  # Rough weekend estimates
            'Avg number of travelers on Sunday': int(estimate * 0.25)
        }
        enhanced_travelers = pd.concat([enhanced_travelers, pd.DataFrame([new_row])], ignore_index=True)

    print(f"✓ Enhanced travelers data: {len(enhanced_travelers)} stations "
          f"(added {len(station_estimates)} missing stations)")

    return enhanced_travelers

def load_all_data():
    """STEP 1: Load all necessary data files"""
    print("=== STEP 1: DATA PREPARATION ===")
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
    print(f"✓ Loaded {len(trips):,} trip records from {len(trips_files)} files")

    # Load other datasets
    stations = pd.read_csv("data/stations.csv")
    travelers = robust_fix_travelers_data()  # Use the robust fixed version
    # Load incidents with proper semicolon separator and handle duplicate columns
    incidents = pd.read_csv("data/incidents.csv", sep=';')
    print(f"Original incidents columns: {list(incidents.columns)}")

    # Handle duplicate column names - rename them
    if len(incidents.columns) >= 6:  # We expect at least 6 columns
        new_columns = list(incidents.columns)
        # Rename duplicate 'Place' columns
        place_count = 0
        for i, col in enumerate(new_columns):
            if col == 'Place':
                place_count += 1
                if place_count > 1:
                    new_columns[i] = f'Place_{place_count}'

        incidents.columns = new_columns
        print(f"Renamed incidents columns: {list(incidents.columns)}")

    print(f"✓ Loaded {len(stations)} stations")
    print(f"✓ Loaded {len(travelers)} traveler records")
    print(f"✓ Loaded {len(incidents)} incidents")

    return trips, stations, travelers, incidents

def preprocess_data(trips):
    """STEP 1: Preprocess trips data with consistent units and keys"""
    print("\n=== STEP 1: DATA PREPROCESSING ===")

    # Convert timestamps to datetime objects
    trips['planned_departure_datetime'] = pd.to_datetime(
        trips['Planned departure date'] + ' ' + trips['Planned departure time'], errors='coerce'
    )

    # Extract time components
    trips['departure_hour'] = trips['planned_departure_datetime'].dt.hour
    trips['departure_dayofweek'] = trips['planned_departure_datetime'].dt.day_name()

    # Convert delays from seconds to minutes (consistent units)
    trips['departure_delay'] = trips['Delay at departure'] / 60.0

    # Define peak hours (assumed based on typical patterns)
    morning_peak = PEAK_HOURS_MORNING
    evening_peak = PEAK_HOURS_EVENING

    trips['is_peak'] = trips['departure_hour'].isin(morning_peak + evening_peak)
    trips['is_weekday'] = ~trips['planned_departure_datetime'].dt.dayofweek.isin([5, 6])

    print("✓ Converted timestamps to datetime objects")
    print("✓ Standardized delay units (seconds → minutes)")
    print("✓ Defined peak hours: 7-9 AM and 4-6 PM")
    print("✓ Added weekday/weekend classification")

    return trips, morning_peak, evening_peak

def calculate_basic_delay_metrics(trips):
    """STEP 2: Calculate basic delay metrics per station"""
    print("\n=== STEP 2: BASIC DELAY METRICS ===")

    # Filter for peak hours on weekdays (when bottlenecks matter most)
    peak_weekday_trips = trips[(trips['is_peak']) & (trips['is_weekday'])]

    print(f"Analyzing {len(peak_weekday_trips):,} peak-hour weekday trips")

    # Calculate basic metrics
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

    print(f"✓ Calculated metrics for {len(delay_metrics)} stations")
    print("✓ Metrics include: total trains, total delay, average delay, severe delays (>5min & >15min)")

    # Display top stations by total delay
    print("\nTop 10 stations by total delay (peak hours, weekdays):")
    top_delay = delay_metrics.nlargest(10, 'total_delay_minutes')[
        ['total_delay_minutes', 'avg_delay_minutes', 'total_trains', 'pct_delays_above_5min']
    ]
    print(top_delay.round(2).to_string())

    return delay_metrics

def reliable_normalize_with_travelers(delay_metrics, travelers):
    """STEP 3: Simple and reliable normalization"""
    print("\n=== STEP 3: RELIABLE TRAFFIC NORMALIZATION ===")

    # Reset index to make 'Stopping place' a column
    delay_metrics = delay_metrics.reset_index()

    # Simple merge on station name
    merged = pd.merge(delay_metrics, travelers, left_on='Stopping place', right_on='Station', how='left')

    # Handle any remaining missing values
    merged['avg_weekday_travelers'] = merged['Avg number of travelers in the week'].fillna(1000)  # Default for any remaining missing
    merged['delay_times_1000_travelers'] = (merged['total_delay_minutes'] * (merged['avg_weekday_travelers'] / 1000)).round(2)

    # Clean up
    merged = merged.replace([np.inf, -np.inf], 0)

    # Remove duplicates (keep first occurrence)
    merged = merged.drop_duplicates(subset=['Stopping place'], keep='first')

    print(f"✓ Final normalized data: {len(merged)} unique stations")

    # Verify Brussels stations
    brussels_stations = ['BRUSSEL-CENTRAAL', 'BRUSSEL-ZUID', 'BRUSSEL-NOORD',
                         'BRUSSEL-CONGRES', 'BRUSSEL-KAPELLEKERK', 'SCHAARBEEK']

    print("\nBrussels stations verification:")
    for station in brussels_stations:
        station_data = merged[merged['Stopping place'] == station]
        if len(station_data) > 0:
            travelers_val = station_data['avg_weekday_travelers'].iloc[0]
            impact_val = station_data['delay_times_1000_travelers'].iloc[0]
            print(f"  {station}: {travelers_val:,.0f} travelers, impact={impact_val:,.2f}")

    # Display top stations by delay impact
    print("\nTop 10 stations by delay impact (delay × travelers):")
    top_impact = merged[merged['avg_weekday_travelers'] > 0].nlargest(10, 'delay_times_1000_travelers')[
        ['Stopping place', 'delay_times_1000_travelers', 'avg_weekday_travelers', 'total_delay_minutes']
    ]
    print(top_impact.round(2).to_string(index=False))

    return merged

def build_route_graph(trips):
    """STEP 4: Build route graph and calculate centrality"""
    print("\n=== STEP 4: NETWORK CENTRALITY ANALYSIS ===")

    # Sample for efficiency
    if len(trips) > GRAPH_SAMPLE_SIZE:
        sample_trips = trips.sample(GRAPH_SAMPLE_SIZE, random_state=42)
        print("Sampled 50,000 trips for graph construction")
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
    edge_counts = Counter(edges)

    print(f"✓ Built directed graph with {len(edge_counts)} unique edges")

    # Calculate centrality (degree centrality)
    station_degrees = {}
    for (source, target), weight in edge_counts.items():
        station_degrees[source] = station_degrees.get(source, 0) + weight
        station_degrees[target] = station_degrees.get(target, 0) + weight

    # Normalize to 0-1 scale
    max_degree = max(station_degrees.values()) if station_degrees else 1
    centrality = {station: degree/max_degree for station, degree in station_degrees.items()}

    print(f"✓ Calculated degree centrality for {len(centrality)} stations")

    # Display top stations by centrality
    centrality_df = pd.DataFrame(list(centrality.items()), columns=['Stopping place', 'degree_centrality'])
    top_centrality = centrality_df.nlargest(10, 'degree_centrality')
    print("\nTop 10 stations by network centrality:")
    print(top_centrality.round(3).to_string(index=False))

    return centrality

def calculate_composite_bottleneck_score(final_metrics):
    """STEP 5: Z-score normalization and composite bottleneck scoring"""
    print("\n=== STEP 5: COMPOSITE BOTTLENECK SCORING ===")

    # Select metrics for the composite score
    metrics_to_include = BOTTLENECK_METRICS

    print("Selected metrics for composite score:")
    available_metrics = []
    for metric in metrics_to_include:
        if metric in final_metrics.columns:
            print(f"  ✓ {metric}")
            available_metrics.append(metric)
        else:
            print(f"  ✗ {metric} (missing)")

    if not available_metrics:
        print("❌ No metrics available for scoring!")
        return final_metrics

    # Calculate Z-scores for each metric
    zscore_data = {}
    for metric in available_metrics:
        # Handle infinite values and missing data
        clean_data = final_metrics[metric].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate Z-scores (standardize to mean=0, std=1)
        zscores = stats.zscore(clean_data, nan_policy='omit')

        # Fill any remaining NaN values with 0 (neutral score)
        zscores = np.nan_to_num(zscores, nan=0.0)

        zscore_data[f'z_{metric}'] = zscores
        print(f"✓ Calculated Z-scores for {metric}")

    # Create DataFrame of Z-scores
    zscore_df = pd.DataFrame(zscore_data)

    # Calculate composite bottleneck score (sum of Z-scores)
    final_metrics['bottleneck_score'] = zscore_df.sum(axis=1)

    # Rank stations by bottleneck score (lower rank = more critical)
    final_metrics['bottleneck_rank'] = final_metrics['bottleneck_score'].rank(ascending=False, method='min')

    print(f"✓ Calculated composite bottleneck scores for {len(final_metrics)} stations")

    return final_metrics

def display_bottleneck_ranking(final_metrics):
    """Display the final bottleneck ranking"""
    print("\n=== TOP 20 BOTTLENECK STATIONS ===")

    # Select top 20 stations by bottleneck score
    top_bottlenecks = final_metrics.nlargest(20, 'bottleneck_score')[
        ['Stopping place', 'bottleneck_score', 'bottleneck_rank',
         'total_delay_minutes', 'delay_times_1000_travelers',
         'degree_centrality', 'total_trains', 'pct_delays_above_5min']
    ].round(3)

    # Reset index for cleaner display
    top_bottlenecks = top_bottlenecks.reset_index(drop=True)

    print("Ranked by composite bottleneck score (higher = more critical):")
    print(top_bottlenecks.to_string(index=False))

    return top_bottlenecks

def analyze_bottleneck_patterns(top_bottlenecks):
    """Analyze patterns in the bottleneck ranking"""
    print("\n=== BOTTLENECK PATTERNS ANALYSIS ===")

    # Categorize stations by their primary bottleneck characteristic
    patterns = []

    for _, station in top_bottlenecks.iterrows():
        name = station['Stopping place']
        score = station['bottleneck_score']
        delay_impact = station['delay_times_1000_travelers']
        centrality = station['degree_centrality']
        total_delay = station['total_delay_minutes']
        total_trains = station['total_trains']

        # Determine primary bottleneck characteristic
        if centrality > 0.8 and total_delay > 5000:
            pattern = "MAJOR HUB: High connectivity + high delays"
        elif delay_impact > 50000:
            pattern = "HIGH IMPACT: Massive delay × passenger volume"
        elif total_delay > 7000:
            pattern = "DELAY HOTSPOT: Extreme total delays"
        elif total_trains > 1800:
            pattern = "HIGH TRAFFIC: Many trains during peak"
        elif centrality > 0.7:
            pattern = "NETWORK CRITICAL: High connectivity"
        else:
            pattern = "MIXED: Multiple contributing factors"

        patterns.append((name, pattern, score))

    print("Bottleneck categories in top 20:")
    pattern_counts = {}
    for name, pattern, score in patterns:
        print(f"  {name:25} → {pattern}")
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    print(f"\nPattern distribution:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} stations")

def proper_propagation_filter(trips, delay_metrics):
    """STEP 6: Filter out delays that are route-wide patterns, not station-specific"""
    print("=== STEP 6: PROPAGATION FILTER (Route-Wide Delay Removal) ===")

    # Calculate delay added (departure - arrival delay)
    trips['arrival_delay'] = trips['Delay at arrival'] / 60.0
    trips['delay_added'] = (trips['departure_delay'] - trips['arrival_delay']).clip(lower=0)

    # Focus on peak weekday trips for consistency
    peak_weekday_trips = trips[(trips['is_peak']) & (trips['is_weekday'])].copy()

    # Group by route to calculate route-level statistics
    route_stats = peak_weekday_trips.groupby(['Relation', 'Relation direction']).agg({
        'delay_added': ['mean', 'std', 'count']
    }).round(3)

    # Flatten column names
    route_stats.columns = ['route_avg_delay', 'route_std_delay', 'route_trip_count']
    route_stats = route_stats.reset_index()

    # Filter routes with sufficient data
    reliable_routes = route_stats[route_stats['route_trip_count'] >= MIN_ROUTE_TRIPS]
    print(f"Analyzing {len(reliable_routes)} routes with sufficient data")

    # For each station-route combination, calculate performance
    station_route_performance = []

    for _, route_data in reliable_routes.iterrows():
        route = route_data['Relation']
        direction = route_data['Relation direction']

        route_trips = peak_weekday_trips[
            (peak_weekday_trips['Relation'] == route) &
            (peak_weekday_trips['Relation direction'] == direction)
            ]

        # Get all stations on this route
        route_stations = route_trips['Stopping place'].unique()

        for station in route_stations:
            station_delays = route_trips[route_trips['Stopping place'] == station]['delay_added']
            if len(station_delays) >= MIN_STATION_OBSERVATIONS:  # Minimum observations per station
                station_avg_delay = station_delays.mean()
                route_avg = route_data['route_avg_delay']
                route_std = max(route_data['route_std_delay'], 0.1)  # Avoid division by zero

                # Calculate Z-score: how different from route baseline
                z_score = (station_avg_delay - route_avg) / route_std

                station_route_performance.append({
                    'Stopping place': station,
                    'Relation': route,
                    'Relation direction': direction,
                    'station_avg_delay': station_avg_delay,
                    'route_avg_delay': route_avg,
                    'route_std_delay': route_std,
                    'z_score': z_score,
                    'observation_count': len(station_delays)
                })

    performance_df = pd.DataFrame(station_route_performance)

    # IMPROVED SYSTEMIC ROUTE DETECTION
    route_delay_patterns = performance_df.groupby(['Relation', 'Relation direction']).agg({
        'z_score': ['mean', 'std'],
        'station_avg_delay': ['mean', 'max', 'min'],
        'observation_count': 'sum'
    }).round(3)

    # Flatten column names
    route_delay_patterns.columns = [
        'avg_z_score', 'std_z_score', 'avg_delay', 'max_delay', 'min_delay', 'total_observations'
    ]
    route_delay_patterns = route_delay_patterns.reset_index()

    # Better systemic route definition: Low variance in Z-scores + high delays
    systemic_routes = route_delay_patterns[
        (route_delay_patterns['std_z_score'] < 1.0) &      # Low variance in delays across stations
        (route_delay_patterns['avg_delay'] > 2.0) &        # Route has significant delays
        (route_delay_patterns['total_observations'] >= 20) # Enough data points
        ]

    print(f"Found {len(systemic_routes)} systemic routes (low variance + high delays)")

    # FILTER: For stations on systemic routes, only count delays where Z-score > threshold
    significant_delay_threshold = SIGNIFICANT_DELAY_Z_THRESHOLD

    # Create adjusted delay metrics
    adjusted_delay_metrics = delay_metrics.copy()

    # For each station, calculate what percentage of its routes show significant responsibility
    responsibility_factors = {}

    for station in adjusted_delay_metrics['Stopping place']:
        station_routes = performance_df[performance_df['Stopping place'] == station]

        if len(station_routes) > 0:
            # Count how many route-station combinations show significant responsibility
            significant_routes = station_routes[station_routes['z_score'] > significant_delay_threshold]
            insignificant_routes = station_routes[station_routes['z_score'] <= significant_delay_threshold]

            # Calculate responsibility factor
            if len(station_routes) > 0:
                responsibility_factor = len(significant_routes) / len(station_routes)
            else:
                responsibility_factor = 1.0  # Default: full responsibility if no route data

            responsibility_factors[station] = responsibility_factor

            # Apply the responsibility factor with hub protection
            if responsibility_factor < RESPONSIBILITY_ADJUST_THRESHOLD:
                # Get station centrality for hub protection
                station_centrality = adjusted_delay_metrics.loc[
                    adjusted_delay_metrics['Stopping place'] == station, 'degree_centrality'
                ].iloc[0]

                # Apply the responsibility factor with minimum protection
                min_responsibility = max(MIN_RESPONSIBILITY_FACTOR, station_centrality * HUB_PROTECTION_FACTOR)
                actual_factor = max(responsibility_factor, min_responsibility)

                adjusted_delay_metrics.loc[
                    adjusted_delay_metrics['Stopping place'] == station,
                    'total_delay_minutes'
                ] *= actual_factor

                # Also adjust severe delay counts proportionally
                for col in ['delays_above_5min', 'delays_above_15min']:
                    if col in adjusted_delay_metrics.columns:
                        adjusted_delay_metrics.loc[
                            adjusted_delay_metrics['Stopping place'] == station,
                            col
                        ] = (adjusted_delay_metrics.loc[
                                 adjusted_delay_metrics['Stopping place'] == station,
                                 col
                             ] * actual_factor).round()

                # Recalculate percentage delays
                if 'total_trains' in adjusted_delay_metrics.columns:
                    adjusted_delay_metrics.loc[
                        adjusted_delay_metrics['Stopping place'] == station,
                        'pct_delays_above_5min'
                    ] = (adjusted_delay_metrics.loc[
                             adjusted_delay_metrics['Stopping place'] == station,
                             'delays_above_5min'
                         ] / adjusted_delay_metrics.loc[
                             adjusted_delay_metrics['Stopping place'] == station,
                             'total_trains'
                         ] * 100).round(2)


    print(f"✓ Applied responsibility factors to {len(responsibility_factors)} stations")

    return adjusted_delay_metrics, performance_df, systemic_routes

def display_propagation_results(original_metrics, adjusted_metrics, performance_df, systemic_routes):
    """Display the results of the propagation filtering"""
    print("\n=== PROPAGATION FILTER RESULTS ===")

    # Calculate rank changes
    original_ranks = original_metrics.set_index('Stopping place')['bottleneck_rank']
    adjusted_ranks = adjusted_metrics.set_index('Stopping place')['bottleneck_rank']

    rank_comparison = pd.DataFrame({
        'original_rank': original_ranks,
        'adjusted_rank': adjusted_ranks
    }).dropna()

    rank_comparison['rank_change'] = rank_comparison['adjusted_rank'] - rank_comparison['original_rank']

    # Stations that improved (moved to less critical ranks - higher rank number)
    improved_stations = rank_comparison[rank_comparison['rank_change'] > 10].nlargest(10, 'rank_change')
    print("\nStations that improved most (less critical after propagation filter):")
    print(improved_stations.round(2).to_string())

    # Stations that worsened (moved to more critical ranks - lower rank number)
    worsened_stations = rank_comparison[rank_comparison['rank_change'] < -10].nsmallest(10, 'rank_change')
    print("\nStations that worsened most (more critical after propagation filter):")
    print(worsened_stations.round(2).to_string())

    # Show systemic routes
    print(f"\nSystemic delay routes (delays are route-wide, not station-specific):")
    for _, route in systemic_routes.iterrows():
        print(f"  {route['Relation']} ({route['Relation direction']}): "
              f"avg_delay={route['station_avg_delay']:.1f}min, avg_z={route['z_score']:.2f}")

def classify_incident(incident_description):
    """Classify incident into category based on description"""
    incident_lower = incident_description.lower()

    for category, keywords in INCIDENT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in incident_lower:
                return category
    return 'OPERATIONAL'  # Default category

def create_incident_station_mapping(incidents, stations):
    """Create mapping between incident locations and station names - FIXED VERSION"""
    print("Creating incident-station mapping...")

    # Standardize station names for matching
    station_names = set(stations['name'].str.upper().str.strip())

    # Also include the stopping place names from trips for better matching
    all_station_variations = set(station_names)

    incident_station_map = {}
    mapped_count = 0
    unmapped_count = 0

    for _, incident in incidents.iterrows():
        # Try different possible column names for location
        incident_place = None

        # Try various possible column names (including renamed duplicates)
        for col in ['Place', 'Place_1', 'Place_2', 'Place_3', 'Location']:
            if col in incidents.columns:
                place_value = incident[col]
                if pd.notna(place_value) and place_value != '-' and place_value != '':
                    incident_place = str(place_value).strip()
                    break

        if incident_place is None:
            unmapped_count += 1
            continue

        # Convert to uppercase for matching
        incident_place_upper = incident_place.upper()

        # Try exact match first
        if incident_place_upper in all_station_variations:
            incident_station_map[incident_place] = incident_place_upper
            mapped_count += 1
            continue

        # Try common variations and partial matches
        matched_station = None

        # Common name variations mapping
        name_variations = {
            'BRUXELLES': 'BRUSSEL',
            'BRUXELLES-': 'BRUSSEL-',
            'ANVERS': 'ANTWERPEN',
            'GAND': 'GENT',
            'LIEGE': 'LIÈGE',
            'LUIK': 'LIÈGE'
        }

        # Apply variations
        test_name = incident_place_upper
        for old, new in name_variations.items():
            test_name = test_name.replace(old, new)

        # Try direct match after variations
        if test_name in all_station_variations:
            matched_station = test_name
        else:
            # Try partial matching
            for station in all_station_variations:
                if (test_name in station or station in test_name or
                        test_name.replace('-', ' ') in station or
                        station.replace('-', ' ') in test_name):
                    matched_station = station
                    break

        if matched_station:
            incident_station_map[incident_place] = matched_station
            mapped_count += 1
        else:
            # If no match found, use original name but log it
            incident_station_map[incident_place] = incident_place_upper
            unmapped_count += 1

    print(f"✓ Mapped {mapped_count} incident locations to stations")
    if unmapped_count > 0:
        print(f"⚠ {unmapped_count} incident locations could not be reliably mapped")

    # Show some examples of mappings
    print("Sample incident-station mappings:")
    for i, (incident_place, station) in enumerate(list(incident_station_map.items())[:5]):
        print(f"  '{incident_place}' → '{station}'")

    return incident_station_map

def calculate_incidental_delay_impact(trips, incidents, incident_station_map, stations):
    """Calculate how much delay should be removed for each station due to incidents - FIXED VERSION"""
    print("Calculating incidental delay impact...")

    # Convert incident dates to datetime properly
    incidents['Incident date'] = pd.to_datetime(incidents['Incident date'], errors='coerce')

    # Remove incidents with invalid dates
    incidents = incidents[incidents['Incident date'].notna()]

    # Classify all incidents
    incidents['category'] = incidents['Incident description'].apply(classify_incident)

    # Group incidents by date and station to calculate daily impact
    daily_incident_impact = {}

    for _, incident in incidents.iterrows():
        # Skip incidents below threshold
        if incident['Minutes of delay'] < MIN_INCIDENT_DELAY:
            continue

        incident_date = incident['Incident date']
        if pd.isna(incident_date):
            continue

        # Find the incident location
        incident_place = None
        for col in ['Place', 'Place_1', 'Place_2', 'Place_3', 'Location']:
            if col in incidents.columns:
                place_value = incident[col]
                if pd.notna(place_value) and place_value != '-' and place_value != '':
                    incident_place = str(place_value).strip()
                    break

        if incident_place is None:
            continue

        incident_station = incident_station_map.get(incident_place)
        if incident_station is None:
            continue

        incident_delay = incident['Minutes of delay']
        category = incident['category']

        # Determine removal factor based on category
        if category == 'EXTERNAL':
            removal_factor = EXTERNAL_REMOVAL_FACTOR
        elif category == 'INFRASTRUCTURE':
            removal_factor = INFRASTRUCTURE_REMOVAL_FACTOR
        else:
            removal_factor = OPERATIONAL_REMOVAL_FACTOR

        # Key for grouping: (date, station)
        key = (incident_date.date(), incident_station)

        if key not in daily_incident_impact:
            daily_incident_impact[key] = {
                'total_incident_delay': 0,
                'weighted_removal': 0,
                'incident_count': 0
            }

        daily_incident_impact[key]['total_incident_delay'] += incident_delay
        daily_incident_impact[key]['weighted_removal'] += incident_delay * removal_factor
        daily_incident_impact[key]['incident_count'] += 1

    print(f"✓ Analyzed {len(daily_incident_impact)} station-day incident impacts")

    # Debug: Show some incident impacts
    if daily_incident_impact:
        print("Sample incident impacts:")
        for i, (key, impact) in enumerate(list(daily_incident_impact.items())[:3]):
            date, station = key
            print(f"  {date} at {station}: {impact['total_incident_delay']} min delay, {impact['incident_count']} incidents")

    return daily_incident_impact

def apply_incidental_delay_filter(trips, delay_metrics, incidents, stations):
    """STEP 7: Remove delays caused by external incidents - FIXED VERSION"""
    print("\n=== STEP 7: INCIDENTAL DELAY FILTER ===")

    # Create incident-station mapping
    incident_station_map = create_incident_station_mapping(incidents, stations)

    # Calculate incidental delay impact
    daily_incident_impact = calculate_incidental_delay_impact(trips, incidents, incident_station_map, stations)

    # Create adjusted delay metrics
    adjusted_delay_metrics = delay_metrics.copy()

    # For each station, calculate total delay to remove based on incidents
    removal_summary = {}

    for station in adjusted_delay_metrics['Stopping place']:
        # Find incidents affecting this station
        station_incidents = []

        for (incident_date, incident_station), impact in daily_incident_impact.items():
            if incident_station == station:
                station_incidents.append((incident_date, impact))

        if not station_incidents:
            continue

        # Calculate total delay to remove for this station
        total_delay_to_remove = 0

        for incident_date, impact in station_incidents:
            # Get the station's total delay on that date
            station_delays_on_date = trips[
                (trips['Stopping place'] == station) &
                (trips['planned_departure_datetime'].dt.date == incident_date)
                ]

            if len(station_delays_on_date) == 0:
                continue

            total_delay_on_date = station_delays_on_date['departure_delay'].sum()

            if total_delay_on_date > 0:
                # Calculate what portion of the delay was due to incidents
                incident_ratio = min(impact['total_incident_delay'] / total_delay_on_date, 1.0)
                delay_to_remove = total_delay_on_date * incident_ratio * (impact['weighted_removal'] / impact['total_incident_delay'])
                total_delay_to_remove += delay_to_remove

        # Apply the removal to the station's total delay (but only if significant)
        if total_delay_to_remove > 10:  # Only remove if more than 10 minutes
            original_delay = adjusted_delay_metrics.loc[
                adjusted_delay_metrics['Stopping place'] == station, 'total_delay_minutes'
            ].iloc[0]

            new_delay = max(original_delay - total_delay_to_remove, 0)
            adjusted_delay_metrics.loc[
                adjusted_delay_metrics['Stopping place'] == station, 'total_delay_minutes'
            ] = new_delay

            removal_summary[station] = {
                'original_delay': original_delay,
                'delay_removed': total_delay_to_remove,
                'new_delay': new_delay,
                'reduction_pct': (total_delay_to_remove / original_delay * 100) if original_delay > 0 else 0
            }

    # Display removal summary
    print_incident_removal_summary(removal_summary)

    return adjusted_delay_metrics, removal_summary

def print_incident_removal_summary(removal_summary):
    """Print a concise summary of incident delay removal - FIXED VERSION"""
    print(f"\n=== INCIDENT DELAY REMOVAL SUMMARY ===")

    if not removal_summary:
        print("No delays were removed by the incidental delay filter.")
        print("This could be due to:")
        print("  - Incident dates not matching trip dates")
        print("  - Station names not matching between incidents and trips")
        print("  - Incidents below the minimum delay threshold")
        return

    print(f"Applied incidental delay filter to {len(removal_summary)} stations")

    total_original_delay = sum(info['original_delay'] for info in removal_summary.values())
    total_delay_removed = sum(info['delay_removed'] for info in removal_summary.values())

    print(f"Total delay before incidents filter: {total_original_delay:,.0f} minutes")
    print(f"Total delay removed: {total_delay_removed:,.0f} minutes")
    print(f"Remaining structural delay: {total_original_delay - total_delay_removed:,.0f} minutes")

    if total_original_delay > 0:
        print(f"Overall reduction: {total_delay_removed/total_original_delay*100:.1f}%")
    else:
        print("Overall reduction: 0.0%")

    # Show top stations by delay removal
    print(f"\nTop 10 stations by incident delay removal:")
    top_removals = sorted(removal_summary.items(),
                          key=lambda x: x[1]['delay_removed'], reverse=True)[:10]

    for station, info in top_removals:
        print(f"  {station:25} | {info['delay_removed']:6.0f} min removed "
              f"({info['reduction_pct']:5.1f}%)")
def main():
    """Main analysis function with all steps including incidental delay filter"""
    print("STARTING COMPLETE BOTTLENECK ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1: Load and preprocess data
    trips, stations, travelers, incidents = load_all_data()
    trips, morning_peak, evening_peak = preprocess_data(trips)

    # Step 2: Basic delay metrics
    delay_metrics = calculate_basic_delay_metrics(trips)

    # ENHANCED: Create complete travelers dataset
    complete_travelers = create_complete_travelers_dataset(travelers, stations, trips)

    # Step 3: Normalize with complete travelers data
    normalized_metrics = reliable_normalize_with_travelers(delay_metrics, complete_travelers)

    # Step 4: Network centrality
    centrality = build_route_graph(trips)

    # Combine all metrics
    centrality_df = pd.DataFrame(list(centrality.items()), columns=['Stopping place', 'degree_centrality'])
    final_metrics = pd.merge(normalized_metrics, centrality_df, on='Stopping place', how='left').fillna(0)

    # Step 5: Composite bottleneck scoring
    final_metrics = calculate_composite_bottleneck_score(final_metrics)

    # Display initial results
    top_bottlenecks = display_bottleneck_ranking(final_metrics)
    analyze_bottleneck_patterns(top_bottlenecks)

    print("\n" + "=" * 60)
    print("STEPS 1-5 COMPLETED - PROCEEDING TO PROPAGATION FILTERING")
    print("=" * 60)

    # STEP 6: Proper Propagation Filtering
    adjusted_delay_metrics, route_performance, systemic_routes = proper_propagation_filter(trips, final_metrics)

    # Recalculate bottleneck scores with propagation filtering
    print("\nRecalculating bottleneck scores with propagation filtering...")
    propagation_adjusted_metrics = calculate_composite_bottleneck_score(adjusted_delay_metrics)

    # Display propagation results
    display_propagation_results(final_metrics, propagation_adjusted_metrics, route_performance, systemic_routes)

    print("\n" + "=" * 60)
    print("STEP 6 COMPLETED - PROCEEDING TO INCIDENTAL DELAY FILTER")
    print("=" * 60)

    # STEP 7: Incidental Delay Filter
    incident_adjusted_metrics, removal_summary = apply_incidental_delay_filter(
        trips, propagation_adjusted_metrics, incidents, stations
    )

    # Recalculate bottleneck scores with incidental delays removed
    print("\nRecalculating bottleneck scores with incidental delays removed...")
    final_bottleneck_metrics = calculate_composite_bottleneck_score(incident_adjusted_metrics)

    # Display final results
    print("\n" + "=" * 60)
    print("FINAL BOTTLENECK RANKING AFTER ALL FILTERING")
    print("=" * 60)
    final_top_bottlenecks = display_bottleneck_ranking(final_bottleneck_metrics)

    # Save final results
    final_bottleneck_metrics.to_csv('final_bottleneck_metrics.csv', index=False)
    propagation_adjusted_metrics.to_csv('propagation_adjusted_metrics.csv', index=False)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("✓ Steps 1-5: Basic bottleneck identification")
    print("✓ Step 6: Propagation filter (route-wide delays)")
    print("✓ Step 7: Incidental delay filter (external factors)")
    print(f"✓ Final results saved to 'final_bottleneck_metrics.csv'")

    return final_bottleneck_metrics

if __name__ == "__main__":
    results = main()