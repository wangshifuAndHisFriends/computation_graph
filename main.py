from computation_graph.prototype import Input, Add, RollMean, Graph
import numpy as np
import pandas as pd


def main():
    x = pd.Series(data=np.random.normal(size=(1000,)))
    y = pd.Series(data=np.random.normal(size=(1000,)))
    z = pd.Series(data=np.random.normal(size=(1000,)))

    # get ground_truth
    print((x + y + z).rolling(window=10, min_periods=1).mean())

    graph = Graph()
    x = Input(x, graph)
    y = Input(y, graph)
    z = Input(z, graph)
    x = Add(x, Add(y, z))
    x = RollMean(x, 10)
    graph.forward()

    print(x.data)


if __name__ == '__main__':
    main()
