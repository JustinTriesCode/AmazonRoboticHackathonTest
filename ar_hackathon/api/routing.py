"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name: hello123
Email address: [removed for other team members privacy]
*******************
"""

import heapq
from collections import deque
from typing import List, Dict, Tuple, Optional, Set, Deque
from ar_hackathon.models.graph_state import GraphState

# ==========================================
# --- TUNABLE PARAMETERS ---
# ==========================================

#TESTING - to see if score improves

# The pathfinding cost added to an edge or node if it is currently at maximum capacity.
# Higher = robots will take massive detours to avoid traffic. 
# Lower = robots are more likely to just wait their turn in traffic.
TRAFFIC_PENALTY_WEIGHT = 1000

# How many ticks a unit must be stuck on the exact same node before it wipes 
# its current route and calculates a completely new detour.
MAX_STALL_TICKS = 6

# ==========================================

# --- PERSISTENT STATE ---
UNIT_ROUTES: Dict[int, Deque[int]] = {}
UNIT_TARGET_POD: Dict[int, List[str]] = {}
ADJACENCY_LIST: Dict[int, List[Tuple[int, float]]] = {}

# Space-Time Tracking
CURRENT_TICK: int = -1
TICK_EDGE_RESERVATIONS: Dict[Tuple[int, int], int] = {}
TICK_NODE_RESERVATIONS: Dict[int, int] = {}

# Tick-Level Cache for instantaneous lookups
CACHE_NODE_OCCUPANCY: Dict[int, int] = {}
CACHE_EDGE_OCCUPANCY: Dict[Tuple[int, int], int] = {}

# Anti-Deadlock Tracking
UNIT_LAST_POS: Dict[int, int] = {}
UNIT_STALL_COUNT: Dict[int, int] = {}


def build_adjacency_list_if_needed(state: GraphState):
    global ADJACENCY_LIST
    if ADJACENCY_LIST:
        return
    for edge in state.edges:
        u = edge.from_node
        v = edge.to_node
        w = edge.weight
        if u not in ADJACENCY_LIST: ADJACENCY_LIST[u] = []
        if v not in ADJACENCY_LIST: ADJACENCY_LIST[v] = []
        ADJACENCY_LIST[u].append((v, w))
        if edge.bidirectional:
            ADJACENCY_LIST[v].append((u, w))


def dijkstra(start_node: int, target_node: int, state: GraphState) -> List[int]:
    build_adjacency_list_if_needed(state)
    pq = [(0.0, start_node)]
    distances = {start_node: 0.0}
    came_from = {}
    
    while pq:
        current_cost, current_node = heapq.heappop(pq)
        
        if current_node == target_node:
            path = []
            curr = target_node
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
            return path
            
        if current_cost > distances.get(current_node, float('inf')):
            continue
            
        for neighbor, weight in ADJACENCY_LIST.get(current_node, []):
            # ANTI-DEADLOCK: Penalize traffic jams using INSTANT CACHE
            penalty = 0
            if neighbor != target_node:
                n_obj = state.get_node(neighbor)
                if n_obj and n_obj.capacity is not None:
                    if CACHE_NODE_OCCUPANCY.get(neighbor, 0) >= n_obj.capacity:
                        penalty += TRAFFIC_PENALTY_WEIGHT
                        
                e_obj = state.get_edge(current_node, neighbor)
                if e_obj and e_obj.capacity is not None:
                    if CACHE_EDGE_OCCUPANCY.get((current_node, neighbor), 0) >= e_obj.capacity:
                        penalty += TRAFFIC_PENALTY_WEIGHT

            new_cost = current_cost + weight + penalty
            
            if new_cost < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_cost
                came_from[neighbor] = current_node
                heapq.heappush(pq, (new_cost, neighbor))
                
    return []


def get_all_claimed_pod_ids() -> Set[str]:
    claimed = set()
    for pod_list in UNIT_TARGET_POD.values():
        for p_id in pod_list:
            claimed.add(p_id)
    return claimed


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    global CURRENT_TICK, TICK_EDGE_RESERVATIONS, TICK_NODE_RESERVATIONS, UNIT_ROUTES
    global CACHE_NODE_OCCUPANCY, CACHE_EDGE_OCCUPANCY
    
    # Reset tick reservations & BUILD TICK CACHE
    if state.current_time_step != CURRENT_TICK:
        CURRENT_TICK = state.current_time_step
        TICK_EDGE_RESERVATIONS.clear()
        TICK_NODE_RESERVATIONS.clear()
        CACHE_NODE_OCCUPANCY.clear()
        CACHE_EDGE_OCCUPANCY.clear()
        
        # Build Occupancy Cache in O(U) time instead of recalculating during routing
        for u in state.drive_units:
            if u.in_transit:
                u_dest = u.transit_destination
                u_curr = u.current_node
                
                CACHE_NODE_OCCUPANCY[u_dest] = CACHE_NODE_OCCUPANCY.get(u_dest, 0) + 1
                
                e_tup = (u_curr, u_dest)
                e_tup_rev = (u_dest, u_curr)
                CACHE_EDGE_OCCUPANCY[e_tup] = CACHE_EDGE_OCCUPANCY.get(e_tup, 0) + 1
                CACHE_EDGE_OCCUPANCY[e_tup_rev] = CACHE_EDGE_OCCUPANCY.get(e_tup_rev, 0) + 1
            else:
                CACHE_NODE_OCCUPANCY[u.current_node] = CACHE_NODE_OCCUPANCY.get(u.current_node, 0) + 1

    unit = state.get_drive_unit(drive_unit_id)
    current_node = unit.current_node

    # --- STALL DETECTOR ---
    if current_node == UNIT_LAST_POS.get(drive_unit_id, -1):
        UNIT_STALL_COUNT[drive_unit_id] = UNIT_STALL_COUNT.get(drive_unit_id, 0) + 1
    else:
        UNIT_STALL_COUNT[drive_unit_id] = 0
    UNIT_LAST_POS[drive_unit_id] = current_node

    # If stuck for threshold ticks, wipe route to force a recalculation
    if UNIT_STALL_COUNT[drive_unit_id] >= MAX_STALL_TICKS:
        UNIT_ROUTES[drive_unit_id] = deque()
        UNIT_STALL_COUNT[drive_unit_id] = 0

    # Route validation - Recalculate if derailed
    if drive_unit_id in UNIT_ROUTES and UNIT_ROUTES[drive_unit_id]:
        next_planned = UNIT_ROUTES[drive_unit_id][0]
        if next_planned not in state.neighbors(current_node):
            UNIT_ROUTES[drive_unit_id] = deque()

    # Clean up target pods
    if drive_unit_id in UNIT_TARGET_POD:
        current_carrying = set(unit.carrying)
        UNIT_TARGET_POD[drive_unit_id] = [
            pid for pid in UNIT_TARGET_POD[drive_unit_id]
            if pid in current_carrying or (state.get_pod(pid) and state.get_pod(pid).carried_by is None)
        ]

    # Route destination validation
    target_station = None
    if unit.carrying:
        first_pod = state.get_pod(unit.carrying[0])
        if first_pod:
            target_station = first_pod.destination_station
            
        if UNIT_ROUTES.get(drive_unit_id):
            final_dest = UNIT_ROUTES[drive_unit_id][-1]
            valid_dest = False
            if final_dest == target_station:
                valid_dest = True
            else:
                for pid in UNIT_TARGET_POD.get(drive_unit_id, []):
                    p = state.get_pod(pid)
                    if p and p.current_node == final_dest:
                        valid_dest = True
            if not valid_dest:
                UNIT_ROUTES[drive_unit_id] = deque()

    # ROUTE PLANNING LOGIC
    if not UNIT_ROUTES.get(drive_unit_id):
        route_found = False
        
        # Find a pod
        if len(unit.carrying) < unit.capacity:
            claimed_pods = get_all_claimed_pod_ids()
            available_pods = [
                p for p in state.active_pods 
                if p.carried_by is None and p.current_node is not None
                and p.entry_time <= state.current_time_step and p.id not in claimed_pods
            ]
            if target_station is not None:
                available_pods = [p for p in available_pods if p.destination_station == target_station]
                
            if available_pods:
                best_pod, best_path, best_cost = None, [], float('inf')
                for pod in available_pods:
                    path = dijkstra(current_node, pod.current_node, state)
                    cost = len(path)
                    if cost < best_cost:
                        best_cost, best_pod, best_path = cost, pod, path
                        
                if best_pod:
                    if drive_unit_id not in UNIT_TARGET_POD:
                        UNIT_TARGET_POD[drive_unit_id] = []
                    UNIT_TARGET_POD[drive_unit_id].append(best_pod.id)
                    UNIT_ROUTES[drive_unit_id] = deque(best_path)
                    route_found = True

        # Go to station
        if not route_found and unit.carrying:
            path = dijkstra(current_node, target_station, state)
            if path:
                UNIT_ROUTES[drive_unit_id] = deque(path)
                route_found = True

        # IDLE PARKING FIX - Return to storage corners if idle
        if not route_found and not unit.carrying:
            current_node_obj = state.get_node(current_node)
            if current_node_obj and current_node_obj.node_type != "storage":
                storage_nodes = [n.id for n in state.nodes if n.node_type == "storage"]
                best_path = []
                best_cost = float('inf')
                for s_node in storage_nodes:
                    path = dijkstra(current_node, s_node, state)
                    if len(path) > 0 and len(path) < best_cost:
                        best_cost = len(path)
                        best_path = path
                if best_path:
                    UNIT_ROUTES[drive_unit_id] = deque(best_path)

    # EXECUTION & COLLISION AVOIDANCE
    if UNIT_ROUTES.get(drive_unit_id):
        intended_next_node = UNIT_ROUTES[drive_unit_id][0]
        
        # Aisle Capacity Check (USING INSTANT CACHE)
        edge = state.get_edge(current_node, intended_next_node)
        if edge is not None and edge.capacity is not None:
            current_occupancy = CACHE_EDGE_OCCUPANCY.get((current_node, intended_next_node), 0)
            new_reservations = TICK_EDGE_RESERVATIONS.get((current_node, intended_next_node), 0)
            if current_occupancy + new_reservations >= edge.capacity:
                return None
                
        # Universal Node Capacity Check (USING INSTANT CACHE)
        target_node_obj = state.get_node(intended_next_node)
        if target_node_obj and target_node_obj.capacity is not None:
            node_occupancy = CACHE_NODE_OCCUPANCY.get(intended_next_node, 0)
            new_reservations = TICK_NODE_RESERVATIONS.get(intended_next_node, 0)
            if node_occupancy + new_reservations >= target_node_obj.capacity:
                return None

        # Lock in move
        edge_tuple = (current_node, intended_next_node)
        TICK_EDGE_RESERVATIONS[edge_tuple] = TICK_EDGE_RESERVATIONS.get(edge_tuple, 0) + 1
        TICK_NODE_RESERVATIONS[intended_next_node] = TICK_NODE_RESERVATIONS.get(intended_next_node, 0) + 1
        
        # Pop with Deque
        return UNIT_ROUTES[drive_unit_id].popleft()

    return None