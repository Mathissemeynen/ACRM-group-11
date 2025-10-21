import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
from scipy import stats


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Step 2: Basic Delay Metrics
PEAK_HOURS_MORNING = [6, 7, 8, 9]      # 6-9 AM
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
    'EXTERNAL': [  # 90% removal - mostly beyond NMBS control
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
MIN_INCIDENT_DELAY = 60  # Ignore incidents below this threshold



def robust_fix_travelers_data():
    """fix the travelers.csv with all stations"""

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
    print(f" Fixed travelers data: {len(travelers)} stations")
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

    print(f" Enhanced travelers data: {len(enhanced_travelers)} stations "
          f"(added {len(station_estimates)} missing stations)")

    return enhanced_travelers

def load_all_data():
    """STEP 1: Load all necessary data files"""
    print("=== STEP 1: DATA PREPARATION ===")

    # Load trips data
    trips_folder = "data/Trips"
    trips_files = glob.glob(os.path.join(trips_folder, '*.csv'))

    all_trips = []
    for file in trips_files:
        df = pd.read_csv(file, sep=';')
        df['file_source'] = os.path.basename(file)
        all_trips.append(df)

    trips = pd.concat(all_trips, ignore_index=True)
    print(f" Loaded {len(trips):,} trip records from {len(trips_files)} files")

    # Load other datasets
    stations = pd.read_csv("data/stations.csv")
    travelers = robust_fix_travelers_data()  # Use the robust fixed version
    # Load incidents with proper semicolon separator and handle duplicate columns
    incidents = pd.read_csv("data/incidents.csv", sep=';')

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

    print(f" Loaded {len(stations)} stations")
    print(f" Loaded {len(travelers)} traveler records")
    print(f" Loaded {len(incidents)} incidents")

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

    print("- Converted timestamps to datetime objects")
    print("- Standardized delay units into minutes")
    print("- peak hours set to 6:00-9:00 and 16:00-18:00")
    print("- Added weekday/weekend classification")

    return trips, morning_peak, evening_peak

def calculate_basic_delay_metrics(trips):
    """STEP 2: Calculate basic delay metrics per station"""
    print("\n=== STEP 2: BASIC DELAY METRICS ===")

    # Filter for peak hours on weekdays (when bottlenecks matter most)
    peak_weekday_trips = trips[(trips['is_peak']) & (trips['is_weekday'])]

    print(f"Analysing {len(peak_weekday_trips):,} peak-hour weekday trips")

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

    print(f" Calculated metrics for {len(delay_metrics)} stations")
    print(" Metrics include: total trains, total delay, average delay, severe delays (>5min & >15min)")

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

    print(f" Final normalized data: {len(merged)} unique stations")

    # Verify Brussels stations
    brussels_stations = ['BRUSSEL-CENTRAAL', 'BRUSSEL-ZUID', 'BRUSSEL-NOORD',
                         'BRUSSEL-CONGRES', 'BRUSSEL-KAPELLEKERK', 'SCHAARBEEK']

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

    print(f" Built directed graph with {len(edge_counts)} unique edges")

    # Calculate centrality (degree centrality)
    station_degrees = {}
    for (source, target), weight in edge_counts.items():
        station_degrees[source] = station_degrees.get(source, 0) + weight
        station_degrees[target] = station_degrees.get(target, 0) + weight

    # Normalize to 0-1 scale
    max_degree = max(station_degrees.values()) if station_degrees else 1
    centrality = {station: degree/max_degree for station, degree in station_degrees.items()}

    print(f" Calculated degree centrality for {len(centrality)} stations")

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
            print(f"   {metric}")
            available_metrics.append(metric)
        else:
            print(f"  ✗ {metric} (missing)")

    if not available_metrics:
        print(" No metrics available for scoring!")
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
        print(f" Calculated Z-scores for {metric}")

    # Create DataFrame of Z-scores
    zscore_df = pd.DataFrame(zscore_data)

    # Calculate composite bottleneck score (sum of Z-scores)
    final_metrics['bottleneck_score'] = zscore_df.sum(axis=1)

    # Rank stations by bottleneck score (lower rank = more critical)
    final_metrics['bottleneck_rank'] = final_metrics['bottleneck_score'].rank(ascending=False, method='min')

    print(f" Calculated composite bottleneck scores for {len(final_metrics)} stations")

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
    print(f"Analyzing {len(reliable_routes)} routes")

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


    print(f" Applied responsibility factors to {len(responsibility_factors)} stations")

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

def create_incident_station_mapping(incidents, stations, trips_station_names):
    """Create mapping between incident locations and station names - FIXED VERSION"""
    print("Creating incident-station mapping...")

    # Use the actual station names from trips data, but ensure they're strings
    trips_stations = set()
    for station in trips_station_names:
        if pd.notna(station):  # Skip NaN values
            trips_stations.add(str(station).strip().upper())

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

        # FIRST: Try exact match with trips station names
        if incident_place_upper in trips_stations:
            incident_station_map[incident_place] = incident_place_upper
            mapped_count += 1
            continue

        # SECOND: Handle combined Dutch/French names in incidents data
        # If incident has combined name like "BRUSSEL-NOORD/BRUXELLES-NORD", extract Dutch part
        if '/' in incident_place_upper:
            dutch_part = incident_place_upper.split('/')[0].strip()
            if dutch_part in trips_stations:
                incident_station_map[incident_place] = dutch_part
                mapped_count += 1
                continue

        # THIRD: Try common variations
        name_variations = {
            'BRUXELLES-': 'BRUSSEL-',
            'BRUXELLES': 'BRUSSEL',
            'ANVERS': 'ANTWERPEN',
            'GAND': 'GENT',
            'LIEGE': 'LIÈGE',
            'LUIK': 'LIÈGE',
            'MONS': 'BERGEN',
            'TOURNAI': 'DOORNIK',
            'NAMUR': 'NAMEN',
            'CHARLEROI': 'CHARLEROI',
            'LOUVAIN': 'LEUVEN',
            'MALINES': 'MECHELEN',
            'HAELEN': 'HALLE',
            'TERVUEREN': 'TERVUREN'
        }

        test_name = incident_place_upper
        for old, new in name_variations.items():
            test_name = test_name.replace(old, new)

        if test_name in trips_stations:
            incident_station_map[incident_place] = test_name
            mapped_count += 1
            continue

        # FOURTH: Try fuzzy matching
        best_match = None
        best_score = 0

        for station in trips_stations:
            # Ensure station is string for string operations
            if not isinstance(station, str):
                continue

            score = 0

            # Remove common suffixes
            clean_incident = test_name.replace('STATION', '').replace('GARE', '').strip('- ').strip()
            clean_station = station.replace('STATION', '').replace('GARE', '').strip('- ').strip()

            if clean_incident == clean_station:
                score = 100
            elif clean_incident in clean_station:
                score = 80
            elif clean_station in clean_incident:
                score = 80
            elif clean_incident.replace('-', ' ') == clean_station.replace('-', ' '):
                score = 90
            elif clean_incident.replace('-', ' ') in clean_station.replace('-', ' '):
                score = 70
            elif clean_station.replace('-', ' ') in clean_incident.replace('-', ' '):
                score = 70

            if score > best_score:
                best_score = score
                best_match = station

        if best_score >= 70:
            incident_station_map[incident_place] = best_match
            mapped_count += 1
        else:
            # FIFTH: If still no match, use the incident name as-is and hope it matches
            incident_station_map[incident_place] = incident_place_upper
            unmapped_count += 1

    print(f" Mapped {mapped_count} incident locations to stations")
    if unmapped_count > 0:
        print(f" {unmapped_count} incident locations could not be reliably mapped")

    return incident_station_map


def analyze_route_incident_prone_patterns(trips, incidents, stations):
    """STEP 7: Analyze historical incident patterns with WEEKLY NORMALIZATION"""
    print("\n=== STEP 7: HISTORICAL INCIDENT PRONENESS ANALYSIS ===")
    print("Using historical incidents as training data (normalized to weekly averages)...")

    # Get all unique routes from trips
    routes = trips[['Relation', 'Relation direction']].drop_duplicates()
    print(f"Analyzing {len(routes)} unique routes from recent trips data")

    # Create incident-station mapping
    trips_station_names = trips['Stopping place'].unique()
    incident_station_map = create_incident_station_mapping(incidents, stations, trips_station_names)

    # Classify incidents and calculate removal factors
    incidents_classified = incidents.copy()
    incidents_classified['Incident date'] = pd.to_datetime(incidents_classified['Incident date'], errors='coerce')
    incidents_classified = incidents_classified[incidents_classified['Incident date'].notna()]
    incidents_classified['category'] = incidents_classified['Incident description'].apply(classify_incident)

    # Calculate removal factor for each incident
    def get_removal_factor(category):
        if category == 'EXTERNAL':
            return EXTERNAL_REMOVAL_FACTOR
        elif category == 'INFRASTRUCTURE':
            return INFRASTRUCTURE_REMOVAL_FACTOR
        else:
            return OPERATIONAL_REMOVAL_FACTOR

    incidents_classified['removal_factor'] = incidents_classified['category'].apply(get_removal_factor)
    incidents_classified['weighted_delay'] = incidents_classified['Minutes of delay'] * incidents_classified['removal_factor']

    print(f" Classified {len(incidents_classified)} historical incidents")

    # Calculate the time period for normalization
    incident_start_date = incidents_classified['Incident date'].min()
    incident_end_date = incidents_classified['Incident date'].max()
    total_incident_days = (incident_end_date - incident_start_date).days + 1
    total_incident_weeks = total_incident_days / 7.0

    print(f" Incident data covers {total_incident_days} days ({total_incident_weeks:.1f} weeks)")
    print(f" From {incident_start_date.date()} to {incident_end_date.date()}")

    # Analyze incident patterns per route with WEEKLY normalization
    route_incident_analysis = []

    for _, route in routes.iterrows():
        relation = route['Relation']
        direction = route['Relation direction']

        # Get all trips for this route in recent data (for normalization)
        route_trips_recent = trips[
            (trips['Relation'] == relation) &
            (trips['Relation direction'] == direction)
            ].copy()

        if len(route_trips_recent) == 0:
            continue

        # Get all stations on this route
        route_stations = route_trips_recent['Stopping place'].unique()

        # Count historical incidents affecting this route (regardless of date)
        route_incidents = []
        total_historical_incident_delay = 0
        total_weighted_incident_delay = 0

        for _, incident in incidents_classified.iterrows():
            if incident['Minutes of delay'] < MIN_INCIDENT_DELAY:
                continue

            # Find incident location
            incident_place = None
            for col in ['Place', 'Place_1', 'Place_2', 'Place_3', 'Location']:
                if col in incidents_classified.columns:
                    place_value = incident[col]
                    if pd.notna(place_value) and place_value != '-' and place_value != '':
                        incident_place = str(place_value).strip()
                        break

            if incident_place is None:
                continue

            incident_station = incident_station_map.get(incident_place)
            if incident_station and incident_station in route_stations:
                route_incidents.append(incident)
                total_historical_incident_delay += incident['Minutes of delay']
                total_weighted_incident_delay += incident['weighted_delay']

        # Calculate route statistics from recent trips
        total_recent_trips = len(route_trips_recent)
        total_recent_delay = route_trips_recent['departure_delay'].sum()
        avg_delay_per_trip = total_recent_delay / total_recent_trips if total_recent_trips > 0 else 0

        # Calculate WEEKLY NORMALIZED incident proneness metrics
        incident_count = len(route_incidents)

        if total_recent_trips > 0 and total_incident_weeks > 0:
            # KEY FIX: Calculate weekly average incidents instead of total incidents
            weekly_avg_incidents = incident_count / total_incident_weeks

            # Now calculate incidents per trip in a typical week
            incidents_per_trip = weekly_avg_incidents / total_recent_trips

            # Estimate incidental delay percentage based on weekly normalized data
            if incident_count > 0:
                avg_incident_delay = total_historical_incident_delay / incident_count
                avg_weighted_incident_delay = total_weighted_incident_delay / incident_count

                # Estimate incidental delay percentage more conservatively
                # Using weekly normalized incidents per trip
                base_incidental_estimate = min(incidents_per_trip * 100 * 5, 50)  # Scale factor 5, cap at 50%

                # Adjust based on incident severity
                if avg_incident_delay > 30:  # High severity incidents
                    severity_adjustment = 1.3
                elif avg_incident_delay > 15:  # Medium severity
                    severity_adjustment = 1.1
                else:  # Low severity
                    severity_adjustment = 1.0

                estimated_incidental_pct = min(base_incidental_estimate * severity_adjustment, 60)  # Max 60%
            else:
                estimated_incidental_pct = 0
                incidents_per_trip = 0
        else:
            incidents_per_trip = 0
            estimated_incidental_pct = 0
            weekly_avg_incidents = 0

        route_incident_analysis.append({
            'Relation': relation,
            'Direction': direction,
            'total_recent_trips': total_recent_trips,
            'total_recent_delay': total_recent_delay,
            'avg_delay_per_trip': avg_delay_per_trip,
            'historical_incident_count': incident_count,
            'weekly_avg_incidents': weekly_avg_incidents,
            'historical_incidents_per_trip': incidents_per_trip,
            'total_historical_incident_delay': total_historical_incident_delay,
            'total_weighted_incident_delay': total_weighted_incident_delay,
            'estimated_incidental_pct': estimated_incidental_pct,
            'route_stations_count': len(route_stations)
        })

    # Create analysis DataFrame
    analysis_df = pd.DataFrame(route_incident_analysis)

    # Filter routes with sufficient data
    reliable_routes = analysis_df[
        (analysis_df['total_recent_trips'] >= 10)  # At least 10 recent trips
    ].copy()

    print(f" Analyzed {len(reliable_routes)} routes with sufficient recent data")

    # Calculate overall statistics
    if len(reliable_routes) > 0:
        avg_incidents_per_trip = reliable_routes['historical_incidents_per_trip'].mean()
        max_incidents_per_trip = reliable_routes['historical_incidents_per_trip'].max()
        routes_with_incidents = reliable_routes[reliable_routes['historical_incident_count'] > 0]

        print(f" Routes with historical incidents: {len(routes_with_incidents)}")
        print(f" Average estimated incidental delays: {reliable_routes['estimated_incidental_pct'].mean():.1f}%")

    return reliable_routes, analysis_df

def print_incident_proneness_analysis(reliable_routes):
    """Print analysis of which routes are most incident-prone based on WEEKLY historical data"""
    print(f"\n=== HISTORICAL INCIDENT PRONENESS ANALYSIS (Weekly Normalized) ===")

    # Routes with highest incident frequency
    print(f"\n--- TOP 15 MOST INCIDENT-PRONE ROUTES (Weekly Historical Averages) ---")
    top_incident_prone = reliable_routes.nlargest(15, 'historical_incidents_per_trip')[
        ['Relation', 'Direction', 'historical_incidents_per_trip', 'weekly_avg_incidents',
         'historical_incident_count', 'total_recent_trips', 'estimated_incidental_pct', 'avg_delay_per_trip']
    ].round(6)

    for _, route in top_incident_prone.iterrows():
        if route['historical_incidents_per_trip'] > 0:
            print(f"{route['Relation']} ({route['Direction']}):")
            print(f"  {route['weekly_avg_incidents']:.2f} avg weekly incidents ({route['historical_incident_count']} total historical)")
            print(f"  {route['total_recent_trips']} recent weekly trips")
            print(f"  Estimated {route['estimated_incidental_pct']:.1f}% of delays are incidental")
            print(f"  Recent avg delay: {route['avg_delay_per_trip']:.1f} min/trip")
            print()

def apply_incident_proneness_filter(trips, delay_metrics, route_incident_prone):
    """STEP 7: Apply incidental delay filter with MORE REALISTIC thresholds"""
    print("\n=== STEP 7: APPLYING INCIDENT PRONENESS FILTER ===")

    # Create adjusted delay metrics
    adjusted_delay_metrics = delay_metrics.copy()
    removal_summary = {}

    # Create station-route mapping
    station_routes = {}
    for _, route in route_incident_prone.iterrows():
        relation = route['Relation']
        direction = route['Direction']

        # Get all stations on this route
        route_trips = trips[
            (trips['Relation'] == relation) &
            (trips['Relation direction'] == direction)
            ]

        route_stations = route_trips['Stopping place'].unique()

        for station in route_stations:
            if station not in station_routes:
                station_routes[station] = []

            station_routes[station].append({
                'relation': relation,
                'direction': direction,
                'historical_incidents_per_trip': route['historical_incidents_per_trip'],
                'estimated_incidental_pct': route['estimated_incidental_pct'],
                'route_trips': len(route_trips),
                'weekly_avg_incidents': route['weekly_avg_incidents'],
                'total_historical_incident_delay': route['total_historical_incident_delay']
            })

    print(f"Mapped {len(station_routes)} stations to routes")

    # For each station, calculate weighted average incidental percentage
    for station in adjusted_delay_metrics['Stopping place']:
        if station not in station_routes:
            continue

        routes = station_routes[station]

        # Only consider routes with meaningful incident proneness - LOWER THRESHOLD
        meaningful_routes = [r for r in routes if r['historical_incidents_per_trip'] > 0.00001]  # Much lower threshold

        if not meaningful_routes:
            continue

        # Calculate weighted average based on route usage and incident proneness
        total_weight = 0
        weighted_incidental_pct = 0

        for route_info in meaningful_routes:
            # Weight by both route usage AND incident proneness AND total historical delay
            weight = (route_info['route_trips'] *
                      route_info['historical_incidents_per_trip'] *
                      max(1, route_info['total_historical_incident_delay'] / 1000))  # Scale by historical impact

            total_weight += weight
            weighted_incidental_pct += route_info['estimated_incidental_pct'] * weight

        if total_weight > 0:
            avg_incidental_pct = weighted_incidental_pct / total_weight
        else:
            avg_incidental_pct = 0

        # ADJUSTED THRESHOLDS - BE MORE AGGRESSIVE
        # Apply removal based on historical incident proneness with REALISTIC thresholds
        if avg_incidental_pct > 0.01:
            # Be more conservative for major hubs, but still apply some removal
            station_centrality = adjusted_delay_metrics.loc[
                adjusted_delay_metrics['Stopping place'] == station, 'degree_centrality'
            ].iloc[0]

            if station_centrality > 0.7:  # Major hub
                removal_pct = min(avg_incidental_pct, 8.0) / 100.0  # Max 8% removal for hubs (was 10%)
            else:
                removal_pct = min(avg_incidental_pct, 12.0) / 100.0  # Max 12% removal for others (was 15%)

            original_delay = adjusted_delay_metrics.loc[
                adjusted_delay_metrics['Stopping place'] == station, 'total_delay_minutes'
            ].iloc[0]

            delay_to_remove = original_delay * removal_pct
            new_delay = max(original_delay - delay_to_remove, 0)

            adjusted_delay_metrics.loc[
                adjusted_delay_metrics['Stopping place'] == station, 'total_delay_minutes'
            ] = new_delay

            # Also adjust the delay impact metric
            travelers = adjusted_delay_metrics.loc[
                adjusted_delay_metrics['Stopping place'] == station, 'avg_weekday_travelers'
            ].iloc[0]

            new_impact = new_delay * (travelers / 1000)
            adjusted_delay_metrics.loc[
                adjusted_delay_metrics['Stopping place'] == station, 'delay_times_1000_travelers'
            ] = new_impact

            removal_summary[station] = {
                'original_delay': original_delay,
                'delay_removed': delay_to_remove,
                'new_delay': new_delay,
                'reduction_pct': removal_pct * 100,
                'avg_incidental_pct': avg_incidental_pct,
                'route_count': len(meaningful_routes),
                'avg_incidents_per_trip': sum(r['historical_incidents_per_trip'] for r in meaningful_routes) / len(meaningful_routes),
                'avg_weekly_incidents': sum(r['weekly_avg_incidents'] for r in meaningful_routes) / len(meaningful_routes)
            }

    # Display removal summary
    print_incident_proneness_removal_summary(removal_summary)

    return adjusted_delay_metrics, removal_summary

def print_incident_proneness_removal_summary(removal_summary):
    """Print removal summary based on WEEKLY historical incident proneness"""
    print(f"\n=== INCIDENT PRONENESS FILTER SUMMARY (Weekly Normalized) ===")

    if not removal_summary:
        print("No substantial incidental delays estimated based on weekly historical patterns.")
        print("This suggests most delays in recent data are structural and within NMBS control.")
        return

    print(f"Applied incident proneness filter to {len(removal_summary)} stations")

    total_original_delay = sum(info['original_delay'] for info in removal_summary.values())
    total_delay_removed = sum(info['delay_removed'] for info in removal_summary.values())

    print(f"Total delay before incident filter: {total_original_delay:,.0f} minutes")
    print(f"Total delay estimated as incidental: {total_delay_removed:,.0f} minutes")
    print(f"Remaining structural delay (within NMBS control): {total_original_delay - total_delay_removed:,.0f} minutes")

    if total_original_delay > 0:
        overall_reduction = (total_delay_removed / total_original_delay) * 100
        print(f"Overall reduction (estimated incidental): {overall_reduction:.1f}%")

    # Show stations with incidental delay removal
    print(f"\nStations with estimated incidental delays (based on weekly historical patterns):")
    sorted_removals = sorted(removal_summary.items(), key=lambda x: x[1]['delay_removed'], reverse=True)

    for station, info in sorted_removals[:20]:  # Show top 20 only
        print(f"  {station:25} | {info['delay_removed']:6.0f} min removed "
              f"({info['reduction_pct']:4.1f}%) | "
              f"Est. incidental: {info['avg_incidental_pct']:5.1f}% | "
              f"Routes: {info['route_count']} | "
              f"Weekly incidents/trip: {info['avg_incidents_per_trip']:.6f}")

def filter_actual_stops(trips):
    """Filter out trains that don't actually stop at stations (arrival = departure)"""
    print("\n=== FILTERING ACTUAL STOPS ===")

    # Store original count
    original_count = len(trips)

    # Create copies of the time columns as strings for comparison
    trips['actual_arrival_str'] = trips['Actual arrival time'].astype(str)
    trips['actual_departure_str'] = trips['Actual departure time'].astype(str)
    trips['planned_arrival_str'] = trips['Planned arrival time'].astype(str)
    trips['planned_departure_str'] = trips['Planned departure time'].astype(str)

    # Filter condition: Remove rows where train doesn't actually stop
    # A train is considered to NOT stop if:
    # 1. Actual arrival time = Actual departure time, AND
    # 2. Planned arrival time = Planned departure time
    stops_filter = ~(
            (trips['actual_arrival_str'] == trips['actual_departure_str']) &
            (trips['planned_arrival_str'] == trips['planned_departure_str'])
    )

    filtered_trips = trips[stops_filter].copy()

    # Remove the temporary string columns
    filtered_trips = filtered_trips.drop(['actual_arrival_str', 'actual_departure_str',
                                          'planned_arrival_str', 'planned_departure_str'], axis=1)

    # Count removed records
    removed_count = original_count - len(filtered_trips)

    # Calculate skipped stops (planned to stop but didn't)
    skipped_stops_mask = (trips['actual_arrival_str'] == trips['actual_departure_str']) & (trips['planned_arrival_str'] != trips['planned_departure_str'])
    skipped_stops_count = skipped_stops_mask.sum()

    # Calculate total planned stops
    planned_stops_mask = (trips['planned_arrival_str'] != trips['planned_departure_str'])
    total_planned_stops = planned_stops_mask.sum()

    # Calculate skipped percentage of planned stops
    skipped_pct_of_planned = (skipped_stops_count / total_planned_stops * 100) if total_planned_stops > 0 else 0

    print(f" Original trip records: {original_count:,}")
    print(f" After filtering pass-throughs: {len(filtered_trips):,}")
    print(f" Removed {removed_count:,} pass-through records ({removed_count/original_count*100:.1f}% of total)")
    print(f"\nStop Planning Analysis:")
    print(f"  - Total planned stops: {total_planned_stops:,}")
    print(f"  - Actually skipped stops: {skipped_stops_count:,} ({skipped_pct_of_planned:.1f}% of planned stops)")
    print(f"  - Pass-throughs (never planned to stop): {removed_count:,}")

    # REMOVED: Pass-through station ranking (not actionable)
    # ADDED: Critical skipped stops analysis

    if skipped_stops_count > 0:
        # Calculate skipped stops by station with percentages
        skipped_by_station = []
        for station in trips['Stopping place'].unique():
            station_mask = trips['Stopping place'] == station
            planned_at_station = planned_stops_mask[station_mask].sum()
            skipped_at_station = skipped_stops_mask[station_mask].sum()

            if planned_at_station > 0 and skipped_at_station > 0:
                skipped_pct = (skipped_at_station / planned_at_station * 100)
                skipped_by_station.append({
                    'station': station,
                    'skipped_count': skipped_at_station,
                    'planned_count': planned_at_station,
                    'skipped_pct': skipped_pct
                })

        # Sort by both count and percentage to find most critical
        skipped_by_station.sort(key=lambda x: (x['skipped_count'], x['skipped_pct']), reverse=True)

        print(f"\nStations with most skipped stops:")
        print(f"{'Station':<25} {'Skipped':<8} {'Planned':<8} {'Skip Rate':<10}")
        print("-" * 55)
        for i, station_data in enumerate(skipped_by_station[:15]):
            if station_data['skipped_pct'] > 10:  # Only show stations with >10% skip rate
                print(f"{station_data['station']:<25} {station_data['skipped_count']:<8} {station_data['planned_count']:<8} {station_data['skipped_pct']:<10.1f}%")

    return filtered_trips

def analyze_stop_patterns(trips):
    """Analyse stop patterns to understand station skipping behavior"""
    print("\n=== STOP PATTERNS ANALYSIS ===")

    # Use string comparison for stop duration analysis
    planned_non_stops = (trips['Planned arrival time'] == trips['Planned departure time'])
    actual_non_stops = (trips['Actual arrival time'] == trips['Actual departure time'])
    planned_stops = (trips['Planned arrival time'] != trips['Planned departure time'])
    actual_stops = (trips['Actual arrival time'] != trips['Actual departure time'])

    # Categorize stop patterns for filtered data
    skipped_stops_count = (planned_stops & actual_non_stops).sum()
    extra_stops_count = (planned_non_stops & actual_stops).sum()
    normal_stops_count = (planned_stops & actual_stops).sum()
    total_filtered = len(trips)

    print("Stop patterns in filtered data (actual stops only):")
    print(f" Normal stops (planned and executed): {normal_stops_count:,} ({normal_stops_count/total_filtered*100:.1f}%)")
    print(f" Extra stops (unplanned but executed): {extra_stops_count:,} ({extra_stops_count/total_filtered*100:.1f}%)")
    print(f" Skipped stops (planned but not executed): {skipped_stops_count:,} ({skipped_stops_count/total_filtered*100:.1f}%)")

    # Analyze operational impact of skipped stops
    if skipped_stops_count > 0:

        # Most problematic routes for skipping
        route_skipping = trips[planned_stops & actual_non_stops].groupby(['Relation', 'Relation direction']).size().sort_values(ascending=False).head(10)
        if len(route_skipping) > 0:
            print(f"\n   Routes with most skipped stops:")
            for (relation, direction), count in route_skipping.items():
                print(f"     {relation} ({direction}): {count} skipped stops")

    return {
        'normal_stops': normal_stops_count,
        'extra_stops': extra_stops_count,
        'skipped_stops': skipped_stops_count
    }
def create_enhanced_travelers_dataset(original_travelers, trips, tickets, subscriptions):
    """Create complete travelers dataset using predictive modeling for missing stations"""
    print("Creating complete travelers dataset")

    # Get all stations from trips data
    all_stations_from_trips = set(trips['Stopping place'].unique())

    # Get stations with known traveler data
    known_stations = set(original_travelers['Station'])
    known_travelers_dict = dict(zip(original_travelers['Station'],
                                    original_travelers['Avg number of travelers in the week']))

    # Identify missing stations
    missing_stations = all_stations_from_trips - known_stations
    print(f"Stations with known traveler data: {len(known_stations)}")
    print(f"Stations needing estimation: {len(missing_stations)}")

    if len(missing_stations) == 0:
        print(" No missing stations - using original traveler data")
        return original_travelers

    # Process ticket and subscription data
    ticket_counts = estimate_travelers_from_ticket_data_improved(tickets, all_stations_from_trips)
    subscription_counts = estimate_travelers_from_subscription_data_improved(subscriptions, all_stations_from_trips)

    # Build prediction model
    model, training_data, model_stats = build_traveler_prediction_model_no_checks(
        known_travelers_dict, ticket_counts, subscription_counts
    )

    # Create enhanced dataset starting with original data
    enhanced_travelers = original_travelers.copy()

    if model is None:
        print(" Could not build model - using fallback method")
        enhanced_travelers = apply_simple_fallback(enhanced_travelers, missing_stations, ticket_counts, subscription_counts)
    else:
        # Predict missing stations
        predictions = predict_missing_travelers_no_checks(
            model, missing_stations, ticket_counts, subscription_counts
        )

        # Add predictions for missing stations only
        for station, prediction in predictions.items():
            new_row = {
                'Station': station,
                'Avg number of travelers in the week': int(prediction['predicted_travelers']),
                'Avg number of travelers on Saturday': int(prediction['predicted_travelers'] * 0.3),
                'Avg number of travelers on Sunday': int(prediction['predicted_travelers'] * 0.25)
            }
            enhanced_travelers = pd.concat([enhanced_travelers, pd.DataFrame([new_row])], ignore_index=True)


    print(f" Enhanced travelers data: {len(enhanced_travelers)} stations "
          f"(added {len(missing_stations)} estimated stations)")

    return enhanced_travelers

def estimate_travelers_from_ticket_data_improved(tickets, trips_station_names):
    """Improved ticket data processing with better station matching"""

    def enhanced_standardize_station_name(name):
        """More robust station name standardization"""
        if pd.isna(name) or name == '':
            return None

        # Convert to uppercase and strip
        name_upper = str(name).strip().upper()

        # Handle combined Dutch/French names (take Dutch part)
        if '/' in name_upper:
            name_upper = name_upper.split('/')[0].strip()

        # Remove common descriptors
        for descriptor in ['STATION', 'GARE', ' - ', ' SNCB']:
            name_upper = name_upper.replace(descriptor, '').strip()

        # Common variations mapping (expanded)
        name_variations = {
            'BRUXELLES-': 'BRUSSEL-',
            'BRUXELLES': 'BRUSSEL',
            'ANVERS': 'ANTWERPEN',
            'GAND': 'GENT',
            'LIEGE': 'LIÈGE',
            'LUIK': 'LIÈGE',
            'MONS': 'BERGEN',
            'TOURNAI': 'DOORNIK',
            'NAMUR': 'NAMEN',
            'CHARLEROI': 'CHARLEROI',
            'LOUVAIN': 'LEUVEN',
            'MALINES': 'MECHELEN',
            'HAELEN': 'HALLE',
            'TERVUEREN': 'TERVUREN',
            'BRUSSELS AIRPORT - ZAVENTEM': 'ZAVENTEM',
            'BRUSSELS-AIRPORT-ZAVENTEM': 'ZAVENTEM',
            'BRUSSEL-LUXEMBURG': 'BRUSSEL-LUXEMBURG',
            'BRUSSEL-LUXEMBOURG': 'BRUSSEL-LUXEMBURG'
        }

        # Apply variations
        for old, new in name_variations.items():
            name_upper = name_upper.replace(old, new)

        # Final cleanup
        name_upper = name_upper.strip('- ').strip()

        return name_upper

    # Apply standardization
    tickets['start_station_std'] = tickets['start_station'].apply(enhanced_standardize_station_name)
    tickets['end_station_std'] = tickets['end_station'].apply(enhanced_standardize_station_name)

    # Count tickets per station (both as origin and destination)
    station_ticket_counts = {}

    # Count start stations
    valid_starts = tickets[tickets['start_station_std'].notna()]
    start_counts = valid_starts['start_station_std'].value_counts()

    # Count end stations
    valid_ends = tickets[tickets['end_station_std'].notna()]
    end_counts = valid_ends['end_station_std'].value_counts()

    # Combine counts, only including stations that exist in trips data
    all_stations = set(start_counts.index) | set(end_counts.index)

    for station in all_stations:
        if station in trips_station_names:
            start_count = start_counts.get(station, 0)
            end_count = end_counts.get(station, 0)
            station_ticket_counts[station] = start_count + end_count

    print(f" Processed {len(tickets)} tickets, found {len(station_ticket_counts)} matched stations")

    # Show some matching examples for verification
    sample_matches = list(station_ticket_counts.items())[:5]

    return station_ticket_counts

def estimate_travelers_from_subscription_data_improved(subscriptions, trips_station_names):
    """Improved subscription data processing"""

    def enhanced_standardize_station_name(name):
        """Same standardization as for tickets"""
        if pd.isna(name) or name == '':
            return None

        name_upper = str(name).strip().upper()

        if '/' in name_upper:
            name_upper = name_upper.split('/')[0].strip()

        for descriptor in ['STATION', 'GARE', ' - ', ' SNCB']:
            name_upper = name_upper.replace(descriptor, '').strip()

        name_variations = {
            'BRUXELLES-': 'BRUSSEL-',
            'BRUXELLES': 'BRUSSEL',
            'ANVERS': 'ANTWERPEN',
            'GAND': 'GENT',
            'LIEGE': 'LIÈGE',
            'LUIK': 'LIÈGE',
            'MONS': 'BERGEN',
            'TOURNAI': 'DOORNIK',
            'NAMUR': 'NAMEN',
            'CHARLEROI': 'CHARLEROI',
            'LOUVAIN': 'LEUVEN',
            'MALINES': 'MECHELEN',
            'HAELEN': 'HALLE',
            'TERVUEREN': 'TERVUREN',
            'BRUSSELS AIRPORT - ZAVENTEM': 'ZAVENTEM',
            'BRUSSELS-AIRPORT-ZAVENTEM': 'ZAVENTEM'
        }

        for old, new in name_variations.items():
            name_upper = name_upper.replace(old, new)

        name_upper = name_upper.strip('- ').strip()
        return name_upper

    # Apply standardization
    subscriptions['start_station_std'] = subscriptions['start_station'].apply(enhanced_standardize_station_name)
    subscriptions['end_station_std'] = subscriptions['end_station'].apply(enhanced_standardize_station_name)

    # Count subscriptions per station
    station_subscription_counts = {}

    # Count start stations
    valid_starts = subscriptions[subscriptions['start_station_std'].notna()]
    start_counts = valid_starts['start_station_std'].value_counts()

    # Count end stations
    valid_ends = subscriptions[subscriptions['end_station_std'].notna()]
    end_counts = valid_ends['end_station_std'].value_counts()

    # Combine counts
    all_stations = set(start_counts.index) | set(end_counts.index)

    for station in all_stations:
        if station in trips_station_names:
            start_count = start_counts.get(station, 0)
            end_count = end_counts.get(station, 0)
            station_subscription_counts[station] = start_count + end_count

    print(f" Processed {len(subscriptions)} subscriptions, found {len(station_subscription_counts)} matched stations")

    sample_matches = list(station_subscription_counts.items())[:5]

    return station_subscription_counts

def build_traveler_prediction_model_no_checks(known_travelers, ticket_counts, subscription_counts):
    """Build prediction model"""

    # Create training dataset
    training_data = []

    for station, actual_travelers in known_travelers.items():
        ticket_count = ticket_counts.get(station, 0)
        subscription_count = subscription_counts.get(station, 0)

        # Only include stations with some ticket/subscription data
        if ticket_count > 0 or subscription_count > 0:
            training_data.append({
                'station': station,
                'ticket_count': ticket_count,
                'subscription_count': subscription_count,
                'actual_travelers': actual_travelers
            })

    if len(training_data) < 10:
        print(" Not enough training data for model")
        return None, None, {}

    # Convert to DataFrame
    df = pd.DataFrame(training_data)

    # Prepare features and target
    X = df[['ticket_count', 'subscription_count']]
    y = df['actual_travelers']

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score

    model = LinearRegression()
    model.fit(X, y)

    # Evaluate model
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    model_stats = {
        'r2': r2,
        'mae': mae,
        'training_size': len(training_data),
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }

    print(f" Model trained (R²={r2:.3f}, MAE={mae:,.0f} travelers)")
    print(f"  Coefficients: tickets={model.coef_[0]:.6f}, subscriptions={model.coef_[1]:.6f}")

    return model, df, model_stats

def predict_missing_travelers_no_checks(model, missing_stations, ticket_counts, subscription_counts):
    """Predict without confidence checks"""
    print("Predicting traveler counts (no confidence checks)...")

    predictions = {}

    for station in missing_stations:
        ticket_count = ticket_counts.get(station, 0)
        subscription_count = subscription_counts.get(station, 0)

        # Create feature array
        X_pred = pd.DataFrame({
            'ticket_count': [ticket_count],
            'subscription_count': [subscription_count]
        })

        if ticket_count > 0 or subscription_count > 0:
            predicted_travelers = model.predict(X_pred)[0]
            # Only basic bounds
            predicted_travelers = max(10, min(predicted_travelers, 500000))
        else:
            predicted_travelers = 1000  # Basic fallback

        predictions[station] = {
            'predicted_travelers': predicted_travelers,
            'ticket_count': ticket_count,
            'subscription_count': subscription_count
        }

    print(f" Predicted travelers for {len(predictions)} missing stations")
    return predictions


def apply_simple_fallback(original_travelers, missing_stations, ticket_counts, subscription_counts):
    """Simple fallback without complex logic"""
    enhanced_travelers = original_travelers.copy()

    for station in missing_stations:
        ticket_count = ticket_counts.get(station, 0)
        subscription_count = subscription_counts.get(station, 0)

        # Very simple heuristic
        if ticket_count + subscription_count > 0:
            estimated = max(100, (ticket_count * 20 + subscription_count * 100))
        else:
            estimated = 1000

        new_row = {
            'Station': station,
            'Avg number of travelers in the week': int(estimated),
            'Avg number of travelers on Saturday': int(estimated * 0.3),
            'Avg number of travelers on Sunday': int(estimated * 0.25)
        }
        enhanced_travelers = pd.concat([enhanced_travelers, pd.DataFrame([new_row])], ignore_index=True)

    return enhanced_travelers
def filter_out_international_trains(trips):
    """Filter out ONLY truly international trains, not domestic IC trains"""

    # Define patterns for TRULY international trains (not Belgian IC trains)
    international_patterns = [
        'INT', 'TGV', 'ICE', 'EURST', 'THA', 'ICN', 'RJ', 'AVE', 'LYRIA'
    ]

    # Create a mask to identify international trains
    is_international = trips['Relation'].str.contains('|'.join(international_patterns), na=False)

    # DON'T filter by train number patterns - this was removing Belgian IC trains
    # Only use the relation patterns

    international_mask = is_international

    # Count before and after
    original_count = len(trips)
    domestic_trips = trips[~international_mask].copy()
    filtered_count = len(domestic_trips)

    print(f" Filtered out {original_count - filtered_count:,} international train records")
    print(f" Remaining trains: {filtered_count:,} ({filtered_count/original_count*100:.1f}% of original)")

    # Show what was filtered
    if international_mask.any():
        international_routes = trips[international_mask]['Relation'].value_counts().head(10)
        print(f"\nInternational routes filtered out:")
        for route, count in international_routes.items():
            print(f"  {route}: {count:,} trips")

    return domestic_trips

def calculate_composite_bottleneck_score(final_metrics):
    """STEP 5: Z-score normalization with progressive Z-score capping"""
    print("\n=== STEP 5: COMPOSITE BOTTLENECK SCORING ===")

    # PROGRESSIVE Z-SCORE CAPPING - Different caps for different metrics
    Z_SCORE_CAPS = {
        'total_delay_minutes': 7.0,      # High cap for delay (allows major hubs to stand out)
        'pct_delays_above_5min': 5.0,    # Medium cap for percentages
        'degree_centrality': 6.0,        # High cap for network importance
        'total_trains': 6.0,             # High cap for traffic volume
        'delay_times_1000_travelers': 7.0 # High cap for societal impact
    }

    print("Z-score capping configuration:")
    for metric, cap in Z_SCORE_CAPS.items():
        print(f"  {metric}: {cap} standard deviations")

    # Select metrics for the composite score
    metrics_to_include = BOTTLENECK_METRICS

    print("Selected metrics for composite score:")
    available_metrics = []
    for metric in metrics_to_include:
        if metric in final_metrics.columns:
            print(f"   {metric}")
            available_metrics.append(metric)
        else:
            print(f"  ✗ {metric} (missing)")

    if not available_metrics:
        print(" No metrics available for scoring!")
        return final_metrics

    # Calculate Z-scores for each metric WITH PROGRESSIVE CAPPING
    zscore_data = {}
    total_capped = 0

    for metric in available_metrics:
        # Handle infinite values and missing data
        clean_data = final_metrics[metric].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate Z-scores (standardize to mean=0, std=1)
        zscores = stats.zscore(clean_data, nan_policy='omit')

        # Fill any remaining NaN values with 0 (neutral score)
        zscores = np.nan_to_num(zscores, nan=0.0)

        # Get the appropriate cap for this metric
        cap = Z_SCORE_CAPS.get(metric, 5.0)  # Default to 5.0 if not specified

        # CAP THE Z-SCORES
        extreme_pos = np.sum(zscores > cap)
        extreme_neg = np.sum(zscores < -cap)

        if extreme_pos > 0 or extreme_neg > 0:
            total_capped += (extreme_pos + extreme_neg)

        zscores_capped = np.clip(zscores, -cap, cap)

        zscore_data[f'z_{metric}'] = zscores_capped

    if total_capped > 0:
        print(f" Total extreme Z-scores capped: {total_capped}")

    # Create DataFrame of Z-scores
    zscore_df = pd.DataFrame(zscore_data)

    # Calculate composite bottleneck score (sum of Z-scores)
    final_metrics['bottleneck_score'] = zscore_df.sum(axis=1)

    # Rank stations by bottleneck score (lower rank = more critical)
    final_metrics['bottleneck_rank'] = final_metrics['bottleneck_score'].rank(ascending=False, method='min')

    print(f" Calculated composite bottleneck scores for {len(final_metrics)} stations")
    print(f" Applied progressive Z-score capping to balance metric influence")

    return final_metrics


def create_final_bottleneck_ranking(final_bottleneck_metrics, propagation_adjusted_metrics, original_metrics):
    """Create comprehensive final bottleneck ranking with all metrics and rank changes"""
    print("\n=== CREATING FINAL BOTTLENECK RANKING CSV ===")

    # Load the intermediate files we saved
    try:
        step5_metrics = pd.read_csv('bottleneck_metrics_after_step5.csv')
        step6_metrics = pd.read_csv('bottleneck_metrics_after_step6.csv')
        step7_metrics = final_bottleneck_metrics.copy()  # This is after all filters

        print(" Loaded metrics from all filtering steps")
    except FileNotFoundError as e:
        print(f" Error loading intermediate files: {e}")
        return None

    # Calculate rank changes
    print("Calculating rank changes...")

    # Get ranks from each step
    step5_ranks = step5_metrics.set_index('Stopping place')['bottleneck_rank']
    step6_ranks = step6_metrics.set_index('Stopping place')['bottleneck_rank']
    step7_ranks = step7_metrics.set_index('Stopping place')['bottleneck_rank']

    # Calculate rank changes (positive = improved, negative = worsened)
    propagation_rank_change = step6_ranks - step5_ranks  # After propagation filter
    incident_rank_change = step7_ranks - step6_ranks     # After incident filter

    # Create the final comprehensive dataset
    final_ranking = step7_metrics.copy()

    # Add rank change columns
    final_ranking['propagation_rank_change'] = final_ranking['Stopping place'].map(propagation_rank_change)
    final_ranking['incident_rank_change'] = final_ranking['Stopping place'].map(incident_rank_change)

    # Fill NaN values in rank changes with 0 (no change)
    final_ranking['propagation_rank_change'] = final_ranking['propagation_rank_change'].fillna(0)
    final_ranking['incident_rank_change'] = final_ranking['incident_rank_change'].fillna(0)

    # Select and order the columns for the final output
    bottleneck_components = [
        'total_delay_minutes', 'pct_delays_above_5min', 'degree_centrality',
        'total_trains', 'delay_times_1000_travelers', 'avg_weekday_travelers'
    ]

    # Ensure all component columns exist
    available_components = [col for col in bottleneck_components if col in final_ranking.columns]

    # Define the final column order
    final_columns = [
                        'Stopping place',
                        'bottleneck_score',
                        'bottleneck_rank'
                    ] + available_components + [
                        'propagation_rank_change',
                        'incident_rank_change'
                    ]

    # Select only the columns that exist
    final_columns = [col for col in final_columns if col in final_ranking.columns]

    final_ranking = final_ranking[final_columns]

    # Sort by bottleneck rank (most critical first)
    final_ranking = final_ranking.sort_values('bottleneck_rank')

    # Save the final comprehensive ranking
    final_ranking.to_csv('final_bottleneck_ranking.csv', index=False)

    print(f" Created final_bottleneck_ranking.csv with {len(final_ranking)} stations")
    print(f" Columns included: {', '.join(final_columns)}")

    # Display summary of rank changes
    print(f"\n📊 RANK CHANGE SUMMARY:")
    print(f"  Stations that improved after propagation filter: {len(final_ranking[final_ranking['propagation_rank_change'] > 0])}")
    print(f"  Stations that worsened after propagation filter: {len(final_ranking[final_ranking['propagation_rank_change'] < 0])}")
    print(f"  Stations that improved after incident filter: {len(final_ranking[final_ranking['incident_rank_change'] > 0])}")
    print(f"  Stations that worsened after incident filter: {len(final_ranking[final_ranking['incident_rank_change'] < 0])}")

    # Show top stations with largest improvements/worsening
    if len(final_ranking) > 0:
        print(f"\n🏆 TOP 5 STATIONS WITH LARGEST IMPROVEMENTS (After Propagation Filter):")
        top_improved_propagation = final_ranking.nlargest(5, 'propagation_rank_change')[['Stopping place', 'propagation_rank_change', 'bottleneck_rank']]
        for _, station in top_improved_propagation.iterrows():
            print(f"  {station['Stopping place']}: +{station['propagation_rank_change']:.0f} rank improvement")

        print(f"\n📉 TOP 5 STATIONS WITH LARGEST WORSENING (After Propagation Filter):")
        top_worsened_propagation = final_ranking.nsmallest(5, 'propagation_rank_change')[['Stopping place', 'propagation_rank_change', 'bottleneck_rank']]
        for _, station in top_worsened_propagation.iterrows():
            print(f"  {station['Stopping place']}: {station['propagation_rank_change']:.0f} rank change")

    return final_ranking
def main():
    """Main analysis function with all steps including route-based incidental delay filter - UPDATED"""
    print("STARTING COMPLETE BOTTLENECK ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1: Load and preprocess data
    trips, stations, travelers, incidents = load_all_data()

    # NEW: Filter out international trains
    trips = filter_out_international_trains(trips)

    # Filter out trains that don't actually stop
    trips = filter_actual_stops(trips)
    # Analyze stop patterns (optional, for insights)
    stop_patterns = analyze_stop_patterns(trips)

    trips, morning_peak, evening_peak = preprocess_data(trips)

    # Step 2: Basic delay metrics
    delay_metrics = calculate_basic_delay_metrics(trips)

    # Load additional data for traveler estimation
    print("Loading ticket and subscription data")
    tickets = pd.read_csv("data/tickets.csv")
    subscriptions = pd.read_csv("data/subscriptions.csv")

    # ENHANCED: Create complete travelers dataset using predictive modeling
    complete_travelers = create_enhanced_travelers_dataset(travelers, trips, tickets, subscriptions)

    # Step 3: Normalize with complete travelers data
    normalized_metrics = reliable_normalize_with_travelers(delay_metrics, complete_travelers)

    # Step 4: Network centrality
    centrality = build_route_graph(trips)

    # Combine all metrics
    centrality_df = pd.DataFrame(list(centrality.items()), columns=['Stopping place', 'degree_centrality'])
    final_metrics = pd.merge(normalized_metrics, centrality_df, on='Stopping place', how='left').fillna(0)

    # Step 5: Composite bottleneck scoring
    final_metrics = calculate_composite_bottleneck_score(final_metrics)

    # NEW: Save bottleneck metrics after Step 5 with score and rank
    bottleneck_metrics_completed_steps = final_metrics.copy()
    bottleneck_metrics_completed_steps.to_csv('bottleneck_metrics_after_step5.csv', index=False)
    print(" Saved bottleneck metrics after Step 5 to 'bottleneck_metrics_after_step5.csv'")

    # Display initial results
    top_bottlenecks = display_bottleneck_ranking(final_metrics)

    print("\n" + "=" * 60)
    print("STEPS 1-5 COMPLETED - PROCEEDING TO PROPAGATION FILTERING")
    print("=" * 60)

    # STEP 6: Proper Propagation Filtering
    adjusted_delay_metrics, route_performance, systemic_routes = proper_propagation_filter(trips, final_metrics)

    # Recalculate bottleneck scores with propagation filtering
    print("\nRecalculating bottleneck scores with propagation filtering...")
    propagation_adjusted_metrics = calculate_composite_bottleneck_score(adjusted_delay_metrics)

    # NEW: Save bottleneck metrics after Step 6 with score and rank
    propagation_adjusted_metrics.to_csv('bottleneck_metrics_after_step6.csv', index=False)
    print(" Saved bottleneck metrics after Step 6 to 'bottleneck_metrics_after_step6.csv'")

    # Display propagation results
    display_propagation_results(final_metrics, propagation_adjusted_metrics, route_performance, systemic_routes)

    print("\n" + "=" * 60)
    print("STEP 6 COMPLETED - PROCEEDING TO INCIDENTAL DELAY FILTER")
    print("=" * 60)

    # STEP 7: Historical Incident Proneness Analysis
    print("\n=== HISTORICAL INCIDENT PATTERN ANALYSIS ===")
    print("Using historical incidents as training data to identify routes prone to external delays...")

    route_incident_prone, all_routes_analysis = analyze_route_incident_prone_patterns(trips, incidents, stations)

    # Print the incident proneness analysis
    print_incident_proneness_analysis(route_incident_prone)

    print("\n=== APPLYING INCIDENT PRONENESS FILTER ===")
    incident_adjusted_metrics, removal_summary = apply_incident_proneness_filter(
        trips, propagation_adjusted_metrics, route_incident_prone
    )

    # Recalculate bottleneck scores with incidental delays removed
    print("\nRecalculating bottleneck scores with incidental delays removed...")
    final_bottleneck_metrics = calculate_composite_bottleneck_score(incident_adjusted_metrics)

    # NEW: Create the comprehensive final ranking
    final_comprehensive_ranking = create_final_bottleneck_ranking(
        final_bottleneck_metrics,
        propagation_adjusted_metrics,
        final_metrics
    )

    # Display final results
    print("\n" + "=" * 60)
    print("FINAL BOTTLENECK RANKING AFTER ALL FILTERING")
    print("=" * 60)
    final_top_bottlenecks = display_bottleneck_ranking(final_bottleneck_metrics)

    # Save final results
    final_bottleneck_metrics.to_csv('final_bottleneck_metrics.csv', index=False)
    propagation_adjusted_metrics.to_csv('propagation_adjusted_metrics.csv', index=False)
    route_incident_prone.to_csv('route_vulnerability_analysis.csv', index=False)
    all_routes_analysis.to_csv('all_routes_analysis.csv', index=False)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(" Steps 1-5: Basic bottleneck identification")
    print(" Step 6: Propagation filter (route-wide delays)")
    print(" Step 7: Route-based incidental delay filter (external factors)")
    print(" International trains filtered out")
    print(" Route vulnerability analysis completed")
    print(f" Final comprehensive ranking saved to 'final_bottleneck_ranking.csv'")
    print(f" Step-by-step bottleneck metrics saved:")
    print(f"    - bottleneck_metrics_after_step5.csv")
    print(f"    - bottleneck_metrics_after_step6.csv")
    print(f"    - final_bottleneck_metrics.csv (after Step 7)")
    print(f" Route analysis saved to 'route_vulnerability_analysis.csv'")

    return final_bottleneck_metrics

if __name__ == "__main__":
    results = main()