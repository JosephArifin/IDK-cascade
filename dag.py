from typing import Any


class DAG:
    def __init__(
        self, start_vertex, models, avg_times, confidence_thresholds, table
    ) -> None:
        self.start_vertex = start_vertex
        self.graph: list[Any] = [[self.start_vertex]]
        self.models = models
        self.num_models = len(models)
        self.row_indexes = {k: i for i, (k, _) in enumerate(self.models.items())}

        self.avg_times = avg_times

        # dict[str[num_layers], float[confidence]]
        self.confidence_thresholds = confidence_thresholds

        # dict[tuple[row], float[prob_a]]
        self.table = table

    def construct_graph(self):
        for curr_layer in self.graph:
            next_layer: dict[tuple[str], Vertex] = {}

            is_complete = False
            for vertex in curr_layer:
                if len(vertex.cascade) >= self.num_models:
                    is_complete = True
                    break
                self.get_vertex_children(vertex, next_layer)
            if is_complete:
                break

            self.graph.append(list(next_layer.values()))

    def get_vertex_children(self, curr_vertex, layer) -> None:
        for layers, _ in self.models.items():
            if layers not in curr_vertex.cascade:
                new_cascade = curr_vertex.cascade + [layers]
                new_cascade.sort(key=lambda e: int(e))
                if tuple(new_cascade) in layer.keys():
                    curr_vertex.edges.append(
                        Edge(
                            layer[tuple(new_cascade)],
                            self.calc_moving_cost(curr_vertex.cascade, new_cascade),
                        )
                    )
                else:
                    next_vertex = Vertex(new_cascade)
                    curr_vertex.edges.append(
                        Edge(
                            next_vertex,
                            self.calc_moving_cost(curr_vertex.cascade, new_cascade),
                        )
                    )
                    layer[tuple(new_cascade)] = next_vertex

    def calc_moving_cost(self, prev, next) -> float:
        row = [0] * self.num_models
        for layers in prev:
            row[self.row_indexes[layers]] = 1

        added_model = (set(next) - set(prev)).pop()
        cost = self.avg_times[added_model] * (1 - self.table[tuple(row)])
        return cost


class Vertex:
    def __init__(self, cascade) -> None:
        self.cascade: list[str] = cascade
        self.edges: list[Edge] = []


class Edge:
    def __init__(self, next_vertex, cost) -> None:
        self.next_vertex = next_vertex
        self.cost = cost
