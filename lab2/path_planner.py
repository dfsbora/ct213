from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # Todo: implement the Dijkstra algorithm
        start_node = Node(start_position[0], start_position[1])
        start_node.f = 0
        pq = []
        heapq.heappush(pq, (start_node.f, start_node))
        while pq:
            cost, node = heapq.heappop(pq)
            sucessors = self.node_grid.get_successors(node.i, node.j)
            for next in sucessors:   # (i,j) in (i,j)(i,j)(i,j)
                next_node = self.node_grid.get_node(next[0],next[1])
                if next_node.closed:
                    continue
                edge_cost = self.cost_map.get_edge_cost((node.i,node.j),(next_node.i,next_node.j))
                if next_node.f > node.f + edge_cost:
                    next_node.f = node.f + edge_cost
                    #print(next_node.f)
                    next_node.parent = node
                    heapq.heappush(pq, (next_node.f, next_node))
            node.closed = True

        goal_node = self.node_grid.get_node(goal_position[0], goal_position[1])
        path = self.construct_path(goal_node)
        cost = goal_node.f

        self.node_grid.reset()
        return path, cost

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        start_node = Node(start_position[0], start_position[1])
        start_node.f = start_node.distance_to(goal_position[0], goal_position[1])
        start_node.g = start_node.f
        pq = []
        heapq.heappush(pq, (start_node.f, start_node))
        while pq:
            cost, node = heapq.heappop(pq)
            if node.closed:
                continue
            node.closed = True
            sucessors = self.node_grid.get_successors(node.i, node.j)
            for sucessor in sucessors:
                next_node = self.node_grid.get_node(sucessor[0], sucessor[1])
                if next_node.closed:
                    continue
                next_node.parent = node
                next_node.f = next_node.distance_to(goal_position[0], goal_position[1])
                next_node.g = node.g + next_node.f
                if next_node.i == goal_position[0] and next_node.j == goal_position[1]:
                    goal_node = next_node
                    pq = []
                    break
                heapq.heappush(pq, (next_node.f, next_node))



        path = self.construct_path(goal_node)
        cost = goal_node.g

        self.node_grid.reset()
        return path, cost

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        start_node = Node(start_position[0], start_position[1])
        start_node.g = 0
        start_node.f = start_node.distance_to(goal_position[0], goal_position[1])
        pq = []
        heapq.heappush(pq, (start_node.f, start_node))
        while pq:
            cost, node = heapq.heappop(pq)
            sucessors = self.node_grid.get_successors(node.i, node.j)
            for sucessor in sucessors:
                next_node = self.node_grid.get_node(sucessor[0], sucessor[1])
                if next_node.closed:
                    continue
                next_node.parent = node
                heuristic = next_node.distance_to(goal_position[0], goal_position[1])
                edge_cost = self.cost_map.get_edge_cost((node.i,node.j),(next_node.i,next_node.j))
                if next_node.f > node.g + edge_cost + heuristic:
                    next_node.g = node.g + edge_cost
                    next_node.f = next_node.g + heuristic
                    #next_node.parent = node
                    heapq.heappush(pq, (next_node.f, next_node))
                if next_node.i == goal_position[0] and next_node.j == goal_position[1]:
                    goal_node = next_node
                    pq = []
                    break
                #heapq.heappush(pq, (next_node.f, next_node))
            node.closed = True

        path = self.construct_path(goal_node)
        cost = goal_node.f

        self.node_grid.reset()
        return path, cost
