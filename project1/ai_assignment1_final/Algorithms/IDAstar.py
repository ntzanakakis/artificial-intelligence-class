import copy
import time
import sys
from abc import ABC
from typing import Tuple, Union, Dict, List, Any
import math
import numpy as np

from commonroad.scenario.trajectory import State

sys.path.append('../')
from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from SMP.motion_planner.node import Node, CostNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.queue import FIFOQueue, LIFOQueue
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass
from SMP.motion_planner.utility import MotionPrimitiveStatus, initial_visualization, update_visualization

class SequentialSearch(SearchBaseClass, ABC):
    """
    Abstract class for search motion planners.
    """

    # declaration of class variables
    path_fig: Union[str, None]

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

    def initialize_search(self, time_pause, cost=True):
        """
        initializes the visualizer
        returns the initial node
        """
        self.list_status_nodes = []
        self.dict_node_status: Dict[int, Tuple] = {}
        self.time_pause = time_pause
        self.visited_nodes = []

        # first node
        if cost:
            node_initial = CostNode(list_paths=[[self.state_initial]],
                                        list_primitives=[self.motion_primitive_initial],
                                        depth_tree=0, cost=0)
        else:
            node_initial = Node(list_paths=[[self.state_initial]],
                                list_primitives=[self.motion_primitive_initial],
                                depth_tree=0)
        initial_visualization(self.scenario, self.state_initial, self.shape_ego, self.planningProblem,
                              self.config_plot, self.path_fig)
        self.dict_node_status = update_visualization(primitive=node_initial.list_paths[-1],
                                                     status=MotionPrimitiveStatus.IN_FRONTIER,
                                                     dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(self.list_status_nodes), time_pause=self.time_pause)
        self.list_status_nodes.append(copy.copy(self.dict_node_status))
        return node_initial

    def take_step(self, successor, node_current, cost=True):
        """
        Visualizes the step of a successor and checks if it collides with either an obstacle or a boundary
        cost is equal to the cost function up until this node
        Returns collision boolean and the child node if it does not collide
        """
        # translate and rotate motion primitive to current position
        list_primitives_current = copy.copy(node_current.list_primitives)
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])
        list_primitives_current.append(successor)
        self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
        if cost:
            child = CostNode(list_paths=self.path_new,
                                 list_primitives=list_primitives_current,
                                 depth_tree=node_current.depth_tree + 1,
                                 cost=self.cost_function(node_current))
        else:
            child = Node(list_paths=self.path_new, list_primitives=list_primitives_current,
                         depth_tree=node_current.depth_tree + 1)

        # check for collision, skip if is not collision-free
        if not self.is_collision_free(path_translated):

            position = self.path_new[-1][-1].position.tolist()
            self.list_status_nodes, self.dict_node_status, self.visited_nodes = self.plot_colliding_primitives(current_node=node_current,
                                                                                           path_translated=path_translated,
                                                                                           node_status=self.dict_node_status,
                                                                                           list_states_nodes=self.list_status_nodes,
                                                                                           time_pause=self.time_pause,
                                                                                           visited_nodes=self.visited_nodes)
            return True, child
        self.update_visuals()
        return False, child

    def update_visuals(self):
        """
        Visualizes a step on plot
        """
        position = self.path_new[-1][-1].position.tolist()
        if position not in self.visited_nodes:
            self.dict_node_status = update_visualization(primitive=self.path_new[-1],
                                                         status=MotionPrimitiveStatus.IN_FRONTIER,
                                                         dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                         config=self.config_plot,
                                                         count=len(self.list_status_nodes), time_pause=self.time_pause)
            self.list_status_nodes.append(copy.copy(self.dict_node_status))
        self.visited_nodes.append(position)

    def goal_reached(self, successor, node_current):
        """
        Checks if the goal is reached.
        Returns True/False if goal is reached
        """
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])
        # goal test
        if self.reached_goal(path_translated):
            # goal reached
            self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
            path_solution = self.remove_states_behind_goal(self.path_new)
            self.list_status_nodes = self.plot_solution(path_solution=path_solution, node_status=self.dict_node_status,
                                                        list_states_nodes=self.list_status_nodes, time_pause=self.time_pause)
            return True
        return False

    def get_obstacles_information(self):
        """
        Information regarding the obstacles.
        Returns a list of obstacles' information, each element
        contains information regarding an obstacle:
        [x_center_position, y_center_position, length, width]

        """
        return self.extract_collision_obstacles_information()

    def get_goal_information(self):
        """
        Information regarding the goal.
        Returns a list of the goal's information
        with the following form:
        [x_center_position, y_center_position, length, width]
        """
        return self.extract_goal_information()

    def get_node_information(self, node_current):
        """
        Information regarding the input node_current.
        Returns a list of the node's information
        with the following form:
        [x_center_position, y_center_position]
        """
        return node_current.get_position()

    def get_node_path(self, node_current):
        """
        Information regarding the input node_current.
        Returns the path starting from the initial node and ending at node_current.
        """
        return node_current.get_path()

    def cost_function(self, node_current):
        """
        Returns g(n) from initial to current node, !only works with cost nodes!
        """
        velocity = node_current.list_paths[-1][-1].velocity

        node_center = self.get_node_information(node_current)
        goal_center = self.get_goal_information()
        distance_x = goal_center[0] - node_center[0]
        distance_y = goal_center[1] - node_center[1]
        length_goal = goal_center[2]
        width_goal = goal_center[3]

        distance = 4.5
        if(abs(distance_x)<length_goal/2 and abs(distance_y)<width_goal/2):
            prev_x = node_current.list_paths[-2][-1].position[0]
            prev_y = node_current.list_paths[-2][-1].position[1]
            distance = goal_center[0] - length_goal / 2 - prev_x
        cost = node_current.cost + distance
        
        return cost

    def heuristic_function(self, node_current, alt_flag):
        """
        Enter your heuristic function h(x) calculation of distance from node_current to goal
        Returns the distance normalized to be comparable with cost function measurements
        """
        goal=self.get_goal_information()
        node=self.get_node_information(node_current)
        if alt_flag==1:
            dx = (goal[0] - node[0])
            dy = (goal[1] - node[1])
            distance = abs(dx + dy)
        else:
            dx = goal[0] - node[0]
            dy = goal[1] - node[1]
            distance = math.sqrt(dx**2+dy**2)                                        
        return distance

    def evaluation_function(self, node_current , alt_flag):
        """
        f(x) = g(x) + h(x)
        """
        g = self.cost_function(node_current)
        h = self.heuristic_function(node_current, alt_flag)
        f = g + h
        return f

    def search(self,node_current,threshold,alt_flag):
        global total_nodes #global might be the cause for counter issues, revise
        val = math.inf
        cost=self.evaluation_function(node_current,alt_flag)
        if cost>threshold:
            return cost

        
        next_set=node_current.get_successors()
        for primitive_successor in next_set:
            collision_flag, child = self.take_step(successor=primitive_successor, node_current=node_current)
            total_nodes = total_nodes + 1

            if self.goal_reached(primitive_successor,node_current):
                heuristic=self.heuristic_function(node_initial,alt_flag)
                final_path=self.get_node_path(child)
                final_cost=self.cost_function(child)
                print("Visited Nodes number:",total_nodes)
                print("Path: " , end =" ")
                for i in range(len(final_path)-1):
                    print_node=final_path[i]
                    print(str("({:.3f},{:.3f}) ->").format(print_node[0],print_node[1]), end =" ")
                print_node=final_path[len(final_path)-1]
                print(str("({:.3f},{:.3f})").format(print_node[0],print_node[1]))
                print("Heuristic Cost (initial node): {:.3f} ".format(heuristic))
                print("Estimated Cost: {:.3f} ".format(final_cost) )


                with open('output.txt', 'a') as f:
                    print("Visited Nodes number:",total_nodes, file =  f)
                    print("Path: " , end =" ", file =  f)
                    for i in range(len(final_path)-1):
                        print_node=final_path[i]
                        print(str("({:.3f},{:.3f})").format(print_node[0],print_node[1]), end ="  -> ", file =  f)
                    print_node=final_path[len(final_path)-1]
                    print(str("({:.3f},{:.3f})").format(print_node[0],print_node[1]), file =  f)
                    print("Heuristic Cost (initial node): {:.3f} ".format(heuristic), file =  f)
                    print("Estimated Cost: {:.3f} ".format(final_cost), file =  f)
                return True
            if collision_flag:
                continue
            recursion = self.search(child,threshold,alt_flag)

            if recursion==True:
                return True
            elif(recursion < val):
                val = recursion
        return val       
        
        
                        
    def execute_search(self, time_pause, alt_flag=0) -> Tuple[Union[None, List[List[State]]], Union[None, List[MotionPrimitive]], Any]:
        global node_initial
        node_initial = self.initialize_search(time_pause=time_pause)
        #print(self.get_obstacles_information())
        #print(self.get_goal_information())
        #print(self.get_node_information(node_initial))
        """Enter your code here"""
        global total_nodes
        total_nodes = 0
        threshold = self.heuristic_function(node_initial,alt_flag)
        while(True):
            ret = self.search(node_initial,threshold,alt_flag)
            if ret == True:
                return True
            if ret == math.inf:
                return False                                    
            threshold = ret                        


class IterativeDeepeningAstar(SequentialSearch):
    """
    Class for Iterative Deepening Astar Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
