# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

# import time
# import tracemalloc

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    for _, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']

    # Map route_id to a list of stops in order of their sequence
    for _, row in df_stop_times.iterrows():
        route_id = trip_to_route.get(row['trip_id'])
        if route_id:
            stop_id = row['stop_id']
            stop_sequence = row['stop_sequence']
            route_to_stops[route_id].append((stop_sequence, stop_id))

    # Ensure each route only has unique stops
    for route_id, stop_sequence_pairs in route_to_stops.items():
        sorted_stops = sorted(set(stop_sequence_pairs), key=lambda x: x[0])
        route_to_stops[route_id] = [stop_id for _, stop_id in sorted_stops]

    # Count trips per stop
    for _, row in df_stop_times.iterrows():
        route_id = trip_to_route.get(row['trip_id'])
        if route_id:
            stop_trip_count[stop_id] += 1

    # visualize_stop_route_graph_interactive(route_to_stops)
    
    # Create fare rules for routes
    for _, row in df_fare_rules.iterrows():
        fare_id = row['fare_id']
        route_id = row['route_id']
        origin_id = row['origin_id']
        destination_id = row['destination_id']
        
        if route_id not in fare_rules:
            fare_rules[route_id] = []
        fare_rules[route_id].append({
            'fare_id': fare_id,
            'origin_id': origin_id,
            'destination_id': destination_id
        })

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id')

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """

    route_trip_count = defaultdict(int)
    for route_id in trip_to_route.values():
        route_trip_count[route_id] += 1

    busiest_routes = sorted(route_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]
    return busiest_routes

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    sorted_stops = sorted(stop_trip_count, key=stop_trip_count.get, reverse=True)[:5]
    return sorted_stops

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    stop_route_counts = defaultdict(set)
    for route, stops in route_to_stops.items():
        for stop in stops:
            stop_route_counts[stop].add(route)
    return sorted([(stop, len(routes)) for stop, routes in stop_route_counts.items()], key=lambda x: x[1], reverse=True)[:5]

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    direct_route_pairs = {}

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            stop_1, stop_2 = stops[i], stops[i + 1]
            if (stop_1, stop_2) in direct_route_pairs or (stop_2, stop_1) in direct_route_pairs:
                continue
            direct_route_pairs[(stop_1, stop_2)] = route_id
    
    pair_trip_counts = []
    for (stop_1, stop_2), route_id in direct_route_pairs.items():
        combined_trip_count = stop_trip_count[stop_1] + stop_trip_count[stop_2]
        pair_trip_counts.append(((stop_1, stop_2), route_id, combined_trip_count))

    top_pairs = sorted(pair_trip_counts, key=lambda x: x[2], reverse=True)[:5]
    return [(pair, route_id) for pair, route_id, _ in top_pairs]

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    G = nx.Graph()
    for route, stops in route_to_stops.items():
        for i in range(len(stops)):
            for j in range(len(stops)):
                G.add_edge(stops[i], stops[j], route=route)

    pos = nx.spring_layout(G, dim=3, seed=42)

    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    node_x, node_y, node_z = [], [], []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                               mode='lines',
                               line=dict(color='gray', width=1),
                               hoverinfo='none'))

    fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z,
                               mode='markers+text',
                               marker=dict(size=5, color='blue'),
                               text=list(G.nodes()),
                               hoverinfo='text'))

    fig.update_layout(scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)),
        title="route_to_stops graph",
        showlegend=False)

    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    # start_time = time.time()
    # tracemalloc.start()

    # step_count = 0

    direct_routes = []

    for route_id, stops in route_to_stops.items():
        # step_count += 1
        if (start_stop in stops) and (end_stop in stops):
            direct_routes.append(route_id)

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # end_time = time.time()

    # execution_time = end_time - start_time
    # memory_usage = peak / (1024 * 1024)
    # print(f"\nInput to direct_route_brute_force: ({start_stop}, {end_stop})")
    # print(f"Execution time of direct_route_brute_force: {execution_time} s")
    # print(f"Memory usage of direct_route_brute_force: {memory_usage} MB")
    # print(f"Steps taken: {step_count}\n")

    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates
    DirectRoute(X, Y, R) <= RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y)

    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df
    route_to_stops = defaultdict(list)
    trip_to_route = {}
    stop_trip_count = defaultdict(int)
    fare_rules = {}
    merged_fare_df = None

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops)):
            +RouteHasStop(route_id, stops[i])

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """

    # start_time = time.time()
    # tracemalloc.start()

    # step_count = 0

    direct_routes_query = DirectRoute(start, end, R)
    direct_routes = set()

    for route_id in direct_routes_query:
        direct_routes.add(route_id[0])
        # step_count += 1

    result = sorted(direct_routes)

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # end_time = time.time()

    # execution_time = end_time - start_time
    # memory_usage = peak / (1024 * 1024)
    # print(f"\nInput to query_direct_routes: ({start}, {end})")
    # print(f"Execution time of query_direct_routes: {execution_time} s")
    # print(f"Memory usage of query_direct_routes: {memory_usage} MB")
    # print(f"Steps taken: {step_count}\n")

    return result


# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """

    # start_time = time.time()
    # tracemalloc.start()

    # step_count = 0

    results = set()

    for route1 in DirectRoute(start_stop_id, stop_id_to_include, R1):
        for route2 in DirectRoute(stop_id_to_include, end_stop_id, R2):
            if route1 == route2:
                results.add((route1[0], stop_id_to_include, route2[0]))
            elif max_transfers >= 1:
                results.add((route1[0], stop_id_to_include, route2[0]))
            # step_count += 1

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # end_time = time.time()

    # execution_time = end_time - start_time
    # memory_usage = peak / (1024 * 1024)
    # print(f"\nInput to forward_chaining: ({start_stop_id}, {end_stop_id}, {stop_id_to_include}, {max_transfers})")
    # print(f"Execution time of forward_chaining: {execution_time} s")
    # print(f"Memory usage of forward_chaining: {memory_usage} MB")
    # print(f"Steps taken: {step_count}\n")

    return list(results)

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """

    # start_time = time.time()
    # tracemalloc.start()

    # step_count = 0

    results = set()

    for route2 in DirectRoute(end_stop_id, stop_id_to_include, R2):
        for route1 in DirectRoute(stop_id_to_include, start_stop_id, R1):
            if route1 == route2:
                results.add((route2[0], stop_id_to_include, route1[0]))
            elif max_transfers >= 1:
                results.add((route2[0], stop_id_to_include, route1[0]))
            # step_count += 1

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # end_time = time.time()

    # execution_time = end_time - start_time
    # memory_usage = peak / (1024 * 1024)
    # print(f"\nInput to backward_chaining: ({start_stop_id}, {end_stop_id}, {stop_id_to_include}, {max_transfers})")
    # print(f"Execution time of backward_chaining: {execution_time} s")
    # print(f"Memory usage of backward_chaining: {memory_usage} MB")
    # print(f"Steps taken: {step_count}\n")

    return list(results)

pyDatalog.create_terms('Path, Transfer, Transfers')
# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """

    # start_time = time.time()
    # tracemalloc.start()

    # step_count = 0

    Transfer(X, Y, R1, R2) <= DirectRoute(X, stop_id_to_include, R1) & DirectRoute(stop_id_to_include, Y, R2)

    +Path(start_stop_id, 0)

    Path(Y, Transfers + 1) <= Path(X, Transfers) & Transfer(X, Y, R1, R2) & (Transfers + 1 <= max_transfers)
    
    result = pyDatalog.ask(f"Transfer({start_stop_id}, {end_stop_id}, R1, R2)")

    paths = []
    if result:
        for r1, r2 in result.answers:
            # step_count+=1
            paths.append((r1, stop_id_to_include, r2))
            print(f"Transfer from route {r1} to route {r2} via stop {stop_id_to_include}")

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # end_time = time.time()

    # execution_time = end_time - start_time
    # memory_usage = peak / (1024 * 1024)
    # print(f"\nInput to pddl_planning: ({start_stop_id}, {end_stop_id}, {stop_id_to_include}, {max_transfers})")
    # print(f"Execution time of pddl_planning: {execution_time} s")
    # print(f"Memory usage of pddl_planning: {memory_usage} MB")
    # print(f"Steps taken: {step_count}\n")

    return paths

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pruned_df = merged_fare_df[merged_fare_df['price'] <= initial_fare]
    return pruned_df

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    route_summary = defaultdict(lambda: {'min_price': float('inf'), 'stops': set()})

    for _, row in pruned_df.iterrows():
        route_id = row['route_id']
        stop_id = row['origin_id']
        price = row['price']

        if price < route_summary[route_id]['min_price']:
            route_summary[route_id]['min_price'] = price

        route_summary[route_id]['stops'].add(stop_id)

    return route_summary

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    queue = deque([(start_stop_id, [], 0, initial_fare)])
    visited = set()

    while queue:
        current_stop, path, transfers, fare = queue.popleft()

        if current_stop == end_stop_id:
            return path

        if transfers > max_transfers:
            continue

        for route_id, summary in route_summary.items():
            if current_stop in summary['stops']:
                route_fare = summary['min_price']

                if fare < route_fare:
                    continue

                for next_stop in summary['stops']:
                    if (next_stop, route_id) not in visited:
                        visited.add((next_stop, route_id))
                        new_path = path + [(route_id, next_stop)]

                        new_transfers = transfers + 1 if path and path[-1][0] != route_id else transfers
                        queue.append((next_stop, new_path, new_transfers, fare - route_fare))

    return []