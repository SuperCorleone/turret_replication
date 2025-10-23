from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class Node:
    """Represents a node in the robot morphology graph"""
    node_id: str
    node_type: str  # "root", "torso", "hip", "ankle", etc.
    observation_dim: int
    action_dim: int
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    def __hash__(self):
        return hash(self.node_id)


@dataclass
class Edge:
    """Represents an edge in the robot morphology graph"""
    source: Node
    target: Node
    edge_type: str  # "physical", "symmetric", "functional"
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class MorphologyGraph:
    """Represents the graph structure of a robot morphology"""
    
    def __init__(self, robot_type: str):
        self.robot_type = robot_type
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self._node_dict: Dict[str, Node] = {}
        self._adjacency_list: Dict[Node, List[Node]] = {}
        
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        if node.node_id in self._node_dict:
            raise ValueError(f"Node {node.node_id} already exists")
        
        self.nodes.append(node)
        self._node_dict[node.node_id] = node
        self._adjacency_list[node] = []
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str = "physical") -> None:
        """Add an edge between two nodes"""
        if source_id not in self._node_dict or target_id not in self._node_dict:
            raise ValueError("Source or target node not found")
        
        source = self._node_dict[source_id]
        target = self._node_dict[target_id]
        
        edge = Edge(source=source, target=target, edge_type=edge_type)
        self.edges.append(edge)
        self._adjacency_list[source].append(target)
    
    def get_node(self, node_id: str) -> Node:
        """Get node by ID"""
        return self._node_dict.get(node_id)
    
    def get_neighbors(self, node: Node) -> List[Node]:
        """Get all neighbors of a node"""
        return self._adjacency_list.get(node, [])
    
    def get_node_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type"""
        return [node for node in self.nodes if node.node_type == node_type]
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation"""
        n_nodes = len(self.nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        for edge in self.edges:
            i = node_to_idx[edge.source]
            j = node_to_idx[edge.target]
            adj_matrix[i, j] = 1.0
        
        return adj_matrix
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)


class StandardRobotMorphology:
    """Factory for creating standard robot morphologies"""
    
    @staticmethod
    def create_half_cheetah() -> MorphologyGraph:
        """Create HalfCheetah morphology graph"""
        graph = MorphologyGraph("HalfCheetah")
        
        # Add nodes (simplified representation)
        nodes = [
            Node("torso", "torso", 8, 6),
            Node("bfoot", "foot", 6, 0),
            Node("ffoot", "foot", 6, 0),
            Node("bthigh", "thigh", 6, 1),
            Node("bshin", "shin", 6, 1),
            Node("fthigh", "thigh", 6, 1),
            Node("fshin", "shin", 6, 1),
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        # Add edges (physical connections)
        edges = [
            ("torso", "bthigh"), ("bthigh", "bshin"), ("bshin", "bfoot"),
            ("torso", "fthigh"), ("fthigh", "fshin"), ("fshin", "ffoot"),
        ]
        
        for source, target in edges:
            graph.add_edge(source, target)
        
        return graph
    
    @staticmethod
    def create_ant() -> MorphologyGraph:
        """Create Ant morphology graph"""
        graph = MorphologyGraph("Ant")
        
        # Add nodes
        nodes = [
            Node("torso", "torso", 13, 0),
            Node("front_leg_1_hip", "hip", 6, 1),
            Node("front_leg_1_ankle", "ankle", 6, 1),
            Node("front_leg_2_hip", "hip", 6, 1),
            Node("front_leg_2_ankle", "ankle", 6, 1),
            Node("back_leg_1_hip", "hip", 6, 1),
            Node("back_leg_1_ankle", "ankle", 6, 1),
            Node("back_leg_2_hip", "hip", 6, 1),
            Node("back_leg_2_ankle", "ankle", 6, 1),
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        # Add edges
        edges = [
            ("torso", "front_leg_1_hip"), ("front_leg_1_hip", "front_leg_1_ankle"),
            ("torso", "front_leg_2_hip"), ("front_leg_2_hip", "front_leg_2_ankle"),
            ("torso", "back_leg_1_hip"), ("back_leg_1_hip", "back_leg_1_ankle"),
            ("torso", "back_leg_2_hip"), ("back_leg_2_hip", "back_leg_2_ankle"),
        ]
        
        for source, target in edges:
            graph.add_edge(source, target)
        
        return graph
    
    @staticmethod
    def get_morphology(robot_type: str) -> MorphologyGraph:
        """Get morphology for standard robot type"""
        creators = {
            "HalfCheetah": StandardRobotMorphology.create_half_cheetah,
            "Ant": StandardRobotMorphology.create_ant,
        }
        
        if robot_type not in creators:
            raise ValueError(f"Unsupported robot type: {robot_type}")
        
        return creators[robot_type]()