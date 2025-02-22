import numpy as np
import pickle
import heapq
import math

import matplotlib.pyplot as plt

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_ids_path(adj_matrix, start_node, goal_node):
  def path_exists_bfs(adj_matrix, start_node, goal_node):
    n = len(adj_matrix)
    visited = [False] * n
    queue = deque([start_node])
    
    visited[start_node] = True
    
    while queue:
        node = queue.popleft()
        if node == goal_node:
            return True
        for neighbor, cost in enumerate(adj_matrix[node]):
            if cost > 0 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return False
  
  def depth_limited_search(adj_matrix, current_node, goal_node, depth, visited):
      if current_node == goal_node:
          return [current_node]
      if depth == 0:
          return None
      
      visited[current_node] = True
      for neighbor, cost in enumerate(adj_matrix[current_node]):
          if cost > 0 and not visited[neighbor]:
              path = depth_limited_search(adj_matrix, neighbor, goal_node, depth - 1, visited)
              if path:
                  return [current_node] + path
      
      visited[current_node] = False
      return None

  if not path_exists_bfs(adj_matrix, start_node, goal_node):
    return None
  
  n = len(adj_matrix)
  for depth in range(n):
      visited = [False] * n
      path = depth_limited_search(adj_matrix, start_node, goal_node, depth, visited)
      if path:
          return path
  return None
  return []


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

from collections import deque

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):

  def build_path(parent_start, parent_goal, intersect):
      path_start, path_goal = [], []
      
      node = intersect
      while node is not None:
          path_start.append(node)
          node = parent_start[node]
      
      node = intersect
      while node is not None:
          path_goal.append(node)
          node = parent_goal[node]
      
      return path_start[::-1] + path_goal[1:]

  def bfs_bidirectional(adj_matrix, start, goal):
    n = len(adj_matrix)
    
    # Initialize frontiers and visited sets for both searches
    front_start, front_goal = {start}, {goal}
    parent_start, parent_goal = {start: None}, {goal: None}
    
    while front_start and front_goal:
        # Expand from start
        new_front_start = set()
        for node in front_start:
            for neighbor, cost in enumerate(adj_matrix[node]):
                if cost > 0 and neighbor not in parent_start:
                    parent_start[neighbor] = node
                    new_front_start.add(neighbor)
                    if neighbor in parent_goal:
                        return build_path(parent_start, parent_goal, neighbor)
        front_start = new_front_start
        
        # Expand from goal
        new_front_goal = set()
        for node in front_goal:
            for neighbor, cost in enumerate(adj_matrix[node]):
                if cost > 0 and neighbor not in parent_goal:
                    parent_goal[neighbor] = node
                    new_front_goal.add(neighbor)
                    if neighbor in parent_start:
                        return build_path(parent_start, parent_goal, neighbor)
        front_goal = new_front_goal
    
    return None

  return bfs_bidirectional(adj_matrix, start_node, goal_node)

  return []


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):

  def euclidean_distance(node1, node2, node_attributes):
    x1, y1 = float(node_attributes[node1]['x']), float(node_attributes[node1]['y'])
    x2, y2 = float(node_attributes[node2]['x']), float(node_attributes[node2]['y'])
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

  def reconstruct_path(came_from, current):
      total_path = [current]
      while current in came_from:
          current = came_from[current]
          if current is None:
              break
          total_path.append(current)
      return total_path[::-1]

  n = len(adj_matrix)
  open_set = []
  heapq.heappush(open_set, (0, start_node))
  
  came_from = {start_node: None}
  g_score = {start_node: 0}
  f_score = {start_node: euclidean_distance(start_node, goal_node, node_attributes)}
  
  while open_set:
      _, current = heapq.heappop(open_set)
      
      if current == goal_node:
          return reconstruct_path(came_from, current)
      
      for neighbor, cost in enumerate(adj_matrix[current]):
          if cost > 0:
              tentative_g_score = g_score[current] + cost
              if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                  came_from[neighbor] = current
                  g_score[neighbor] = tentative_g_score
                  f_score[neighbor] = tentative_g_score + euclidean_distance(neighbor, goal_node, node_attributes)
                  heapq.heappush(open_set, (f_score[neighbor], neighbor))
  
  return None

  return []


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
  def build_path(parent_start, parent_goal, intersect):
      path_start, path_goal = [], []
      
      node = intersect
      while node is not None:
          path_start.append(node)
          node = parent_start[node]
      
      node = intersect
      while node is not None:
          path_goal.append(node)
          node = parent_goal[node]
      
      return path_start[::-1] + path_goal[1:]

  def euclidean_distance(node1, node2, node_attributes):
    x1, y1 = float(node_attributes[node1]['x']), float(node_attributes[node1]['y'])
    x2, y2 = float(node_attributes[node2]['x']), float(node_attributes[node2]['y'])
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

  def bidirectional_astar(adj_matrix, node_attributes, start, goal):
      n = len(adj_matrix)
      open_set_start, open_set_goal = [], []
      heapq.heappush(open_set_start, (0, start))
      heapq.heappush(open_set_goal, (0, goal))
      
      g_score_start = {start: 0}
      g_score_goal = {goal: 0}
      came_from_start = {start: None}
      came_from_goal = {goal: None}
      
      while open_set_start and open_set_goal:
          _, current_start = heapq.heappop(open_set_start)
          _, current_goal = heapq.heappop(open_set_goal)
          
          # Check if paths meet
          if current_start in g_score_goal:
              return build_path(came_from_start, came_from_goal, current_start)
          if current_goal in g_score_start:
              return build_path(came_from_start, came_from_goal, current_goal)
          
          # Expand from start
          for neighbor, cost in enumerate(adj_matrix[current_start]):
              if cost > 0:
                  tentative_g_score = g_score_start[current_start] + cost
                  if neighbor not in g_score_start or tentative_g_score < g_score_start[neighbor]:
                      came_from_start[neighbor] = current_start
                      g_score_start[neighbor] = tentative_g_score
                      f_score_start = tentative_g_score + euclidean_distance(neighbor, goal, node_attributes)
                      heapq.heappush(open_set_start, (f_score_start, neighbor))
          
          # Expand from goal
          for neighbor, cost in enumerate(adj_matrix[current_goal]):
              if cost > 0:
                  tentative_g_score = g_score_goal[current_goal] + cost
                  if neighbor not in g_score_goal or tentative_g_score < g_score_goal[neighbor]:
                      came_from_goal[neighbor] = current_goal
                      g_score_goal[neighbor] = tentative_g_score
                      f_score_goal = tentative_g_score + euclidean_distance(neighbor, start, node_attributes)
                      heapq.heappush(open_set_goal, (f_score_goal, neighbor))
      
      return None

  return bidirectional_astar(adj_matrix, node_attributes, start_node, goal_node)

  return []



# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def bonus_problem(adj_matrix):
  n = len(adj_matrix)
  adj_list = {i: [] for i in range(n)}
  for u in range(n):
      for v, cost in enumerate(adj_matrix[u]):
          if cost > 0:
              adj_list[u].append(v)
  
  bridges = []
  discovery_time = [-1] * n
  low = [-1] * n
  parent = [-1] * n
  time = 0
  
  def dfs(u):
      nonlocal time
      discovery_time[u] = low[u] = time
      time += 1
      for v in adj_list[u]:
          if discovery_time[v] == -1:  # v is not visited
              parent[v] = u
              dfs(v)
              low[u] = min(low[u], low[v])
              
              if low[v] > discovery_time[u]:
                  bridges.append((u, v))
          elif v != parent[u]:
              low[u] = min(low[u], discovery_time[v])
  
  for i in range(n):
      if discovery_time[i] == -1:
          dfs(i)
  
  return bridges

  return []

import time
import tracemalloc

def get_memory_time_usage(func, *args):
    tracemalloc.start()  
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, end_time - start_time, peak

def compare_uninformed_search(adj_matrix, n):
    for start_node in range(n):
        for end_node in range(n):
            if start_node != end_node:
                print(f'Comparing for Start: {start_node}, Goal: {end_node}')

                ids_result, ids_time, ids_memory = get_memory_time_usage(get_ids_path, adj_matrix, start_node, end_node)
                print(f'IDS Path: {ids_result}, Time: {ids_time:.6f}s, Memory: {ids_memory / 1024:.2f} KB')

                bfs_result, bfs_time, bfs_memory = get_memory_time_usage(get_bidirectional_search_path, adj_matrix, start_node, end_node)
                print(f'BFS Path: {bfs_result}, Time: {bfs_time:.6f}s, Memory: {bfs_memory / 1024:.2f} KB')

def compare_informed_search(adj_matrix, node_attributes, n):
    for start_node in range(n):
        for end_node in range(n):
            if start_node != end_node:
                print(f'Comparing for Start: {start_node}, Goal: {end_node}')

                astar_result, astar_time, astar_memory = get_memory_time_usage(get_astar_search_path, adj_matrix, node_attributes, start_node, end_node)
                print(f'A* Path: {astar_result}, Time: {astar_time:.6f}s, Memory: {astar_memory / 1024:.2f} KB')

                bi_astar_result, bi_astar_time, bi_astar_memory = get_memory_time_usage(get_bidirectional_heuristic_search_path, adj_matrix, node_attributes, start_node, end_node)
                print(f'Bidirectional A* Path: {bi_astar_result}, Time: {bi_astar_time:.6f}s, Memory: {bi_astar_memory / 1024:.2f} KB')

results = {
    'IDS': {'time': [], 'memory': [], 'cost': []},
    'BFS': {'time': [], 'memory': [], 'cost': []},
    'A*': {'time': [], 'memory': [], 'cost': []},
    'Bidirectional A*': {'time': [], 'memory': [], 'cost': []},
}

def calculate_path_cost(adj_matrix, path):
    if path is None:
        return float('inf')  
    cost = 0
    for i in range(len(path) - 1):
        cost += adj_matrix[path[i]][path[i + 1]]
    return cost

# Perform analysis on all pairs of nodes
def analyze_algorithms(adj_matrix, node_attributes, n):
    for start_node in range(n):
        for end_node in range(n):
            if start_node != end_node:
                print(f'Analyzing Start: {start_node}, Goal: {end_node}')

                ids_result, ids_time, ids_memory = get_memory_time_usage(get_ids_path, adj_matrix, start_node, end_node)
                ids_cost = calculate_path_cost(adj_matrix, ids_result)
                results['IDS']['time'].append(ids_time)
                results['IDS']['memory'].append(ids_memory / 1024)
                results['IDS']['cost'].append(ids_cost)

                bfs_result, bfs_time, bfs_memory = get_memory_time_usage(get_bidirectional_search_path, adj_matrix, start_node, end_node)
                bfs_cost = calculate_path_cost(adj_matrix, bfs_result)
                results['BFS']['time'].append(bfs_time)
                results['BFS']['memory'].append(bfs_memory / 1024)
                results['BFS']['cost'].append(bfs_cost)

                astar_result, astar_time, astar_memory = get_memory_time_usage(get_astar_search_path, adj_matrix, node_attributes, start_node, end_node)
                astar_cost = calculate_path_cost(adj_matrix, astar_result)
                results['A*']['time'].append(astar_time)
                results['A*']['memory'].append(astar_memory / 1024)
                results['A*']['cost'].append(astar_cost)

                bi_astar_result, bi_astar_time, bi_astar_memory = get_memory_time_usage(get_bidirectional_heuristic_search_path, adj_matrix, node_attributes, start_node, end_node)
                bi_astar_cost = calculate_path_cost(adj_matrix, bi_astar_result)
                results['Bidirectional A*']['time'].append(bi_astar_time)
                results['Bidirectional A*']['memory'].append(bi_astar_memory / 1024)
                results['Bidirectional A*']['cost'].append(bi_astar_cost)

def plot_comparison():
    # Time vs Memory Usage
    plt.figure(figsize=(10, 6))
    plt.scatter(results['IDS']['time'], results['IDS']['memory'], label='IDS', color='blue')
    plt.scatter(results['BFS']['time'], results['BFS']['memory'], label='BFS', color='green')
    plt.scatter(results['A*']['time'], results['A*']['memory'], label='A*', color='red')
    plt.scatter(results['Bidirectional A*']['time'], results['Bidirectional A*']['memory'], label='Bidirectional A*', color='purple')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Time vs Memory Usage Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Time vs Cost of Traveling
    plt.figure(figsize=(10, 6))
    plt.scatter(results['IDS']['time'], results['IDS']['cost'], label='IDS', color='blue')
    plt.scatter(results['BFS']['time'], results['BFS']['cost'], label='BFS', color='green')
    plt.scatter(results['A*']['time'], results['A*']['cost'], label='A*', color='red')
    plt.scatter(results['Bidirectional A*']['time'], results['Bidirectional A*']['cost'], label='Bidirectional A*', color='purple')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cost of Traveling')
    plt.title('Time vs Path Optimality (Cost) Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')

  n = 30

  print("=== Uninformed Search Comparison ===")
  compare_uninformed_search(adj_matrix, n)

  print("\n=== Informed Search Comparison ===")
  compare_informed_search(adj_matrix, node_attributes, n)

  analyze_algorithms(adj_matrix, node_attributes, n)
    
  plot_comparison()