import abc
from collections import deque
import pandas as pd


class Node:
    def __init__(self, *inputs):
        """

        :param inputs: list of nodes, contains inputs for current node
        """
        self.inputs = set(inputs)
        self.outputs = set([])
        for node in self.inputs:
            node.outputs.add(self)
        self.data = None

    def forward(self):
        """
        computation result of current node

        return: pd.Series indexed by integer
        """
        if self.data is not None:
            return self.data

        self.data = self._forward()
        return self.data

    @abc.abstractmethod
    def _forward(self):
        """
        computation method for each subclasses
        """
        pass

    def __str__(self):
        return self.__class__.__name__


class Input(Node):
    def __init__(self, x, graph):
        super().__init__()
        self.data = x
        graph.inputs.add(self)

    def forward(self):
        return self.data


class Add(Node):
    def __init__(self, node_left: Node, node_right: Node):
        super().__init__(node_left, node_right)
        self.node_left = node_left
        self.node_right = node_right

    def _forward(self):
        left = self.node_left.forward()
        right = self.node_right.forward()
        return left + right


class RollMean(Node):
    def __init__(self, input: Node, window: int):
        super().__init__(input)
        self.window = window
        self.input_node = input

    def _forward(self):
        x = self.input_node.forward()
        return x.rolling(window=self.window, min_periods=1).mean()


class Graph:
    """
    scheduling computational graph
    """
    def __init__(self, *inputs):
        self.inputs = set(inputs)
        self.indegrees = {} # number of inputs of each node
        self.outdegree = {} # number of outputs of each node

    def _get_degrees(self):
        color = {}
        for input in self.inputs:
            if input not in color:
                self._dfs_get_degree(input, color)

    def _dfs_get_degree(self, node, color):
        color[node] = 'g'
        self.indegrees[node] = len(node.inputs)
        for next_node in node.outputs:
            if next_node not in color:
                self._dfs_get_degree(next_node, color)
            else:
                assert color[next_node] == 'b', 'loop found in DAG'
        self.outdegree[node] = len(node.outputs)
        color[node] = 'b'

    def forward(self):
        if len(self.indegrees) == 0 and len(self.inputs) > 0:
            self._get_degrees()

        que = deque()
        visited = set([])

        # initalize by node with indegree zero
        indegrees = self.indegrees.copy()
        for node, indegree in indegrees.items():
            if indegree == 0:
                que.append(node)
                visited.add(node)

        while len(que) > 0:
            node = que.popleft()

            # all the node's inputs are ready
            for input_node in node.inputs:
                assert input_node.data is not None

            # now the node can be computed
            node.forward()

            for next_node in node.outputs:
                indegrees[next_node] -= 1
                assert indegrees[next_node] >= 0

                if indegrees[next_node] == 0:
                    assert next_node not in visited
                    visited.add(next_node)
                    que.append(next_node)




if __name__ == '__main__':
    import numpy as np

    x = pd.Series(data=np.random.normal(size=(1000,)))
    y = pd.Series(data=np.random.normal(size=(1000,)))
    z = pd.Series(data=np.random.normal(size=(1000,)))

    # get ground_truth
    print((x+y+z).rolling(window=10, min_periods=1).mean())

    graph = Graph()
    x = Input(x, graph)
    y = Input(y, graph)
    z = Input(z, graph)
    x = Add(x, Add(y, z))
    x = RollMean(x, 10)
    graph.forward()

    print(x.data)