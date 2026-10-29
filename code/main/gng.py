"""Canonical Growing Neural Gas with supplied-K post-grouping.

The implementation follows Fritzke (1995): competitive Hebbian edges are
aged, stale edges and isolated units are removed, units are inserted from the
largest-error region, and accumulated errors decay.  The final node vectors
are grouped into the supplied K clusters so that the comparison uses the same
target-count protocol as the other K-requiring methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class GNGResult:
    labels: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    errors: np.ndarray
    updates: int
    insertions: int
    edge_deletions: int
    node_deletions: int


class GrowingNeuralGas:
    """Growing Neural Gas followed by supplied-K grouping of final units."""

    def __init__(
        self,
        n_epochs: int = 12,
        max_nodes: int | None = None,
        insertion_interval: int = 100,
        max_age: int = 50,
        winner_learning_rate: float = 0.20,
        neighbor_learning_rate: float = 0.006,
        insertion_error_decay: float = 0.5,
        global_error_decay: float = 0.995,
        random_state: int = 7,
    ) -> None:
        self.n_epochs = int(n_epochs)
        self.max_nodes = max_nodes
        self.insertion_interval = int(insertion_interval)
        self.max_age = int(max_age)
        self.winner_learning_rate = float(winner_learning_rate)
        self.neighbor_learning_rate = float(neighbor_learning_rate)
        self.insertion_error_decay = float(insertion_error_decay)
        self.global_error_decay = float(global_error_decay)
        self.random_state = int(random_state)
        self.result_: GNGResult | None = None

    @staticmethod
    def _edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def fit(self, X: np.ndarray, K: int) -> "GrowingNeuralGas":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 3:
            raise ValueError("GNG requires a two-dimensional array with at least 3 rows")
        if K < 1:
            raise ValueError("K must be positive")
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        max_nodes = self.max_nodes or max(32, min(120, 12 * K))
        init = rng.choice(n, size=2, replace=False)
        nodes = [X[init[0]].copy(), X[init[1]].copy()]
        errors = [0.0, 0.0]
        edges: dict[tuple[int, int], int] = {self._edge_key(0, 1): 0}
        insertions = edge_deletions = node_deletions = updates = 0

        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            for xi_idx in order:
                xi = X[xi_idx]
                W = np.asarray(nodes)
                dist2 = np.einsum("ij,ij->i", W - xi, W - xi)
                if len(dist2) < 2:
                    break
                winners = np.argsort(dist2, kind="mergesort")[:2]
                s1, s2 = int(winners[0]), int(winners[1])
                errors[s1] += float(dist2[s1])
                # Age every edge incident to the winner before refreshing s1-s2.
                for edge in list(edges):
                    if s1 in edge:
                        edges[edge] += 1
                for j in range(len(nodes)):
                    if self._edge_key(s1, j) in edges:
                        nodes[j] += self.neighbor_learning_rate * (xi - nodes[j])
                nodes[s1] += self.winner_learning_rate * (xi - nodes[s1])
                edge = self._edge_key(s1, s2)
                edges[edge] = 0
                # Remove old edges; then remove units without any incident edge.
                old_edges = [e for e, age in edges.items() if age > self.max_age]
                for e in old_edges:
                    del edges[e]
                edge_deletions += len(old_edges)
                incident = {i for e in edges for i in e}
                dead = [i for i in range(len(nodes)) if i not in incident]
                if len(nodes) - len(dead) >= 2 and dead:
                    keep = [i for i in range(len(nodes)) if i not in dead]
                    remap = {old: new for new, old in enumerate(keep)}
                    nodes = [nodes[i] for i in keep]
                    errors = [errors[i] for i in keep]
                    edges = {self._edge_key(remap[a], remap[b]): age for (a, b), age in edges.items() if a in remap and b in remap}
                    node_deletions += len(dead)
                updates += 1
                if (
                    updates % self.insertion_interval == 0
                    and len(nodes) < max_nodes
                    and len(nodes) >= 2
                ):
                    q = int(np.argmax(errors))
                    q_neighbors = [b if a == q else a for a, b in edges if q in (a, b)]
                    if q_neighbors:
                        f = max(q_neighbors, key=lambda j: errors[j])
                        r = len(nodes)
                        nodes.append(0.5 * (nodes[q] + nodes[f]))
                        qf = self._edge_key(q, f)
                        if qf in edges:
                            del edges[qf]
                            edge_deletions += 1
                        edges[self._edge_key(q, r)] = 0
                        edges[self._edge_key(r, f)] = 0
                        errors[q] *= self.insertion_error_decay
                        errors[f] *= self.insertion_error_decay
                        errors.append(errors[q])
                        insertions += 1
                # Fritzke order: reduce q/f, initialize the new error from the
                # reduced q error, then decay every node's accumulated error.
                errors = [e * self.global_error_decay for e in errors]

        nodes_arr = np.asarray(nodes, dtype=float)
        errors_arr = np.asarray(errors, dtype=float)
        edge_arr = np.asarray(sorted(edges), dtype=int) if edges else np.empty((0, 2), dtype=int)
        self.result_ = GNGResult(
            labels=np.empty(n, dtype=int), nodes=nodes_arr, edges=edge_arr,
            errors=errors_arr, updates=updates, insertions=insertions,
            edge_deletions=edge_deletions, node_deletions=node_deletions,
        )
        # Supplied-K grouping is deliberately explicit and reproducible.
        n_groups = min(int(K), len(nodes_arr))
        grouping = KMeans(n_clusters=n_groups, n_init=20, random_state=self.random_state).fit(nodes_arr)
        nearest = np.argmin(((X[:, None, :] - nodes_arr[None, :, :]) ** 2).sum(axis=2), axis=1)
        self.result_.labels = grouping.labels_[nearest]
        self.nodes_ = nodes_arr
        self.edges_ = edge_arr
        self.n_nodes_ = len(nodes_arr)
        return self

    def fit_predict(self, X: np.ndarray, K: int) -> np.ndarray:
        return self.fit(X, K).result_.labels.copy()

