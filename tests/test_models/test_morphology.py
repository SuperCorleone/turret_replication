import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.morphology import MorphologyGraph, Node, Edge, StandardRobotMorphology


class TestMorphology:
    """Test cases for morphology graph"""
    
    def test_node_creation(self):
        """Test node creation"""
        node = Node("test_node", "torso", 10, 4)
        assert node.node_id == "test_node"
        assert node.node_type == "torso"
        assert node.observation_dim == 10
        assert node.action_dim == 4
    
    def test_edge_creation(self):
        """Test edge creation"""
        node1 = Node("node1", "torso", 10, 4)
        node2 = Node("node2", "leg", 6, 2)
        edge = Edge(node1, node2, "physical")
        
        assert edge.source == node1
        assert edge.target == node2
        assert edge.edge_type == "physical"
    
    def test_morphology_graph(self):
        """Test morphology graph functionality"""
        graph = MorphologyGraph("test_robot")
        
        # Add nodes
        nodes = [
            Node("torso", "torso", 8, 0),
            Node("leg1", "leg", 6, 2),
            Node("leg2", "leg", 6, 2),
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        # Add edges
        graph.add_edge("torso", "leg1")
        graph.add_edge("torso", "leg2")
        
        # Test graph properties
        assert graph.num_nodes == 3
        assert graph.num_edges == 2
        assert graph.get_node("torso") == nodes[0]
        assert len(graph.get_neighbors(nodes[0])) == 2
        
        # Test adjacency matrix
        adj_matrix = graph.get_adjacency_matrix()
        assert adj_matrix.shape == (3, 3)
        assert adj_matrix.sum() == 2.0  # Two edges
    
    def test_standard_morphologies(self):
        """Test standard robot morphologies"""
        # Test HalfCheetah
        cheetah = StandardRobotMorphology.create_half_cheetah()
        assert cheetah.robot_type == "HalfCheetah"
        assert cheetah.num_nodes > 0
        assert cheetah.num_edges > 0
        
        # Test Ant
        ant = StandardRobotMorphology.create_ant()
        assert ant.robot_type == "Ant"
        assert ant.num_nodes > 0
        assert ant.num_edges > 0
        
        # Test factory method
        cheetah2 = StandardRobotMorphology.get_morphology("HalfCheetah")
        assert cheetah2.robot_type == "HalfCheetah"