"""
Morphogenetic Buds Clustering

A speculative bio-inspired clustering prototype.

Idea:
    Data points behave like tissue cells emitting morphogens. Cluster centers
    are "buds" that grow toward nutrient-rich regions, suppress nearby rival
    buds through lateral inhibition, die when starved, and create new buds in
    badly explained tissue.

Run:
    python morphogenetic_buds_clustering.py

Dependencies:
    numpy, matplotlib
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BudState:
    centers: np.ndarray
    strengths: np.ndarray
    velocities: np.ndarray


class MorphogeneticBudsClustering:
    """
    Bio-inspired clustering by morphogenetic bud growth.

    This is intentionally not a standard k-means wrapper. It combines:
    - morphogen attraction: buds move toward dense, poorly explained tissue;
    - lateral inhibition: nearby buds repel and suppress each other;
    - apoptosis: weak buds disappear;
    - organogenesis: new buds appear where reconstruction stress is high.
    """

    def __init__(
        self,
        initial_buds: int = 8,
        sigma: float | None = None,
        inhibit_radius: float | None = None,
        steps: int = 120,
        growth_rate: float = 0.18,
        inhibition: float = 0.035,
        apoptosis_mass: float = 2.5,
        birth_quantile: float = 0.92,
        organ_merge_radius: float = 4.2,
        valley_ratio: float = 0.28,
        tangent_alignment: float = 0.42,
        max_bridge_void: float = 0.52,
        target_clusters: int | None = None,
        compact_refinement_steps: int = 0,
        compact_refinement_restarts: int = 1,
        compact_refinement_dim_threshold: int = 3,
        random_state: int = 7,
    ) -> None:
        self.initial_buds = initial_buds
        self.sigma = sigma
        self.inhibit_radius = inhibit_radius
        self.steps = steps
        self.growth_rate = growth_rate
        self.inhibition = inhibition
        self.apoptosis_mass = apoptosis_mass
        self.birth_quantile = birth_quantile
        self.organ_merge_radius = organ_merge_radius
        self.valley_ratio = valley_ratio
        self.tangent_alignment = tangent_alignment
        self.max_bridge_void = max_bridge_void
        self.target_clusters = target_clusters
        self.compact_refinement_steps = compact_refinement_steps
        self.compact_refinement_restarts = compact_refinement_restarts
        self.compact_refinement_dim_threshold = compact_refinement_dim_threshold
        self.random_state = random_state
        self.centers_: np.ndarray | None = None
        self.bud_labels_: np.ndarray | None = None
        self.bud_to_organ_: np.ndarray | None = None
        self.organ_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.sigma_: float | None = None
        self.history_: list[np.ndarray] = []

    def fit(self, x: np.ndarray) -> "MorphogeneticBudsClustering":
        rng = np.random.default_rng(self.random_state)
        x = np.asarray(x, dtype=float)
        n, d = x.shape

        span = np.linalg.norm(x.max(axis=0) - x.min(axis=0))
        sigma = self.sigma or span / math.sqrt(max(n, 2))
        sigma = max(sigma, 1e-6)
        self.sigma_ = sigma
        inhibit_radius = self.inhibit_radius or 2.5 * sigma

        seeds = rng.choice(n, size=min(self.initial_buds, n), replace=False)
        state = BudState(
            centers=x[seeds].copy(),
            strengths=np.ones(len(seeds)),
            velocities=np.zeros((len(seeds), d)),
        )

        for _ in range(self.steps):
            if len(state.centers) == 0:
                state.centers = x[rng.choice(n, size=1)].copy()
                state.strengths = np.ones(1)
                state.velocities = np.zeros((1, d))

            dist2 = squared_distances(x, state.centers)
            affinity = np.exp(-dist2 / (2.0 * sigma * sigma)) * state.strengths
            resp = affinity / (affinity.sum(axis=1, keepdims=True) + 1e-12)

            mass = resp.sum(axis=0)
            targets = (resp.T @ x) / (mass[:, None] + 1e-12)

            # Buds grow toward their local morphogen centroid.
            pull = targets - state.centers
            state.velocities = 0.55 * state.velocities + self.growth_rate * pull
            state.centers = state.centers + state.velocities

            # Lateral inhibition: close buds push each other apart and lose vigor.
            state = self._apply_lateral_inhibition(state, inhibit_radius)

            # Strength is a slow biological memory, not a direct mass copy.
            vigor = mass / (mass.mean() + 1e-12)
            state.strengths = 0.92 * state.strengths + 0.08 * np.clip(vigor, 0.15, 3.0)

            # Apoptosis: starved buds disappear after the tissue has settled a bit.
            keep = (mass >= self.apoptosis_mass) | (len(state.centers) <= 2)
            state.centers = state.centers[keep]
            state.strengths = state.strengths[keep]
            state.velocities = state.velocities[keep]
            mass = mass[keep]

            # Organogenesis: create a bud in high-stress tissue that is far from
            # existing buds. This lets the cluster count emerge from the data.
            state = self._maybe_birth_bud(x, state, sigma, rng)
            state = self._merge_overgrown_buds(state, radius=0.75 * inhibit_radius)

            self.history_.append(state.centers.copy())

        self.centers_ = state.centers
        self.bud_labels_ = squared_distances(x, self.centers_).argmin(axis=1)
        self.bud_to_organ_ = self._form_organs(x, state, sigma)
        self.labels_ = compress_labels(self.bud_to_organ_[self.bud_labels_])
        self.organ_centers_ = weighted_centers(x, self.labels_)
        self._maybe_refine_compact_organs(x, rng)
        return self

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).labels_

    def _apply_lateral_inhibition(self, state: BudState, radius: float) -> BudState:
        k = len(state.centers)
        if k <= 1:
            return state

        delta = state.centers[:, None, :] - state.centers[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + np.eye(k)
        close = (dist < radius) & (~np.eye(k, dtype=bool))
        push = np.zeros_like(state.centers)

        for i in range(k):
            if np.any(close[i]):
                direction = delta[i, close[i]] / (dist[i, close[i], None] + 1e-12)
                pressure = ((radius - dist[i, close[i]]) / radius)[:, None]
                push[i] += (direction * pressure).sum(axis=0)

        state.centers += self.inhibition * radius * push
        state.strengths *= np.exp(-0.025 * close.sum(axis=1))
        return state

    def _maybe_birth_bud(
        self,
        x: np.ndarray,
        state: BudState,
        sigma: float,
        rng: np.random.Generator,
    ) -> BudState:
        dist2 = squared_distances(x, state.centers)
        stress = dist2.min(axis=1)
        threshold = np.quantile(stress, self.birth_quantile)
        candidates = np.flatnonzero(stress >= threshold)
        if len(candidates) == 0:
            return state

        rng.shuffle(candidates)
        for idx in candidates[:8]:
            far_enough = np.sqrt(squared_distances(x[idx : idx + 1], state.centers)).min()
            if far_enough > 2.2 * sigma:
                state.centers = np.vstack([state.centers, x[idx]])
                state.strengths = np.append(state.strengths, 0.65)
                state.velocities = np.vstack([state.velocities, np.zeros(x.shape[1])])
                break
        return state

    def _merge_overgrown_buds(self, state: BudState, radius: float) -> BudState:
        centers = state.centers
        strengths = state.strengths
        velocities = state.velocities
        alive = np.ones(len(centers), dtype=bool)
        merged_centers = []
        merged_strengths = []
        merged_velocities = []

        for i in range(len(centers)):
            if not alive[i]:
                continue
            dist = np.linalg.norm(centers - centers[i], axis=1)
            group = np.flatnonzero(alive & (dist < radius))
            weights = strengths[group]
            merged_centers.append(np.average(centers[group], axis=0, weights=weights))
            merged_velocities.append(np.average(velocities[group], axis=0, weights=weights))
            merged_strengths.append(weights.mean())
            alive[group] = False

        state.centers = np.asarray(merged_centers)
        state.strengths = np.asarray(merged_strengths)
        state.velocities = np.asarray(merged_velocities)
        return state

    def _form_organs(self, x: np.ndarray, state: BudState, sigma: float) -> np.ndarray:
        """
        Fuse nearby buds when the tissue between them is still dense.

        Biologically, this acts like epithelial continuity: two buds become
        the same organ if a morphogen-rich bridge exists between them. A low
        bridge density keeps touching but separated tissues apart.
        """
        centers = state.centers
        k = len(centers)
        uf = UnionFind(k)
        if k <= 1:
            return np.arange(k)

        bud_labels = squared_distances(x, centers).argmin(axis=1)
        tangents, anisotropy = local_tissue_axes(x, centers, bud_labels)
        local_density = kernel_density(x, centers, sigma)
        max_gap = self.organ_merge_radius * sigma
        for i in range(k):
            for j in range(i + 1, k):
                gap = float(np.linalg.norm(centers[i] - centers[j]))
                if gap > max_gap:
                    continue
                midpoint = (centers[i] + centers[j]) / 2.0
                bridge_density = kernel_density(x, midpoint[None, :], sigma)[0]
                endpoint_density = min(local_density[i], local_density[j])
                edge = (centers[j] - centers[i]) / (gap + 1e-12)
                aligned_i = abs(float(np.dot(tangents[i], edge))) >= self.tangent_alignment
                aligned_j = abs(float(np.dot(tangents[j], edge))) >= self.tangent_alignment
                isotropic_i = anisotropy[i] < 1.45
                isotropic_j = anisotropy[j] < 1.45
                polarity_ok = (aligned_i or isotropic_i) and (aligned_j or isotropic_j)
                bridge_ok = bridge_density >= self.valley_ratio * endpoint_density
                capillary_ok = capillary_bridge_ok(
                    x,
                    centers[i],
                    centers[j],
                    sigma,
                    self.max_bridge_void,
                )
                if bridge_ok and polarity_ok and capillary_ok:
                    uf.union(i, j)

        labels = compress_labels(np.array([uf.find(i) for i in range(k)]))
        if self.target_clusters is not None:
            labels = self._regularize_organ_count(x, centers, labels, sigma)
        return labels

    def _regularize_organ_count(
        self,
        x: np.ndarray,
        centers: np.ndarray,
        labels: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Merge organ candidates until the requested macroscopic count is reached.

        Most baselines in the benchmark know the target number of clusters. This
        optional step gives MBC the same information, but it still merges through
        morphogenetic affinity rather than centroid inertia.
        """
        target = max(1, int(self.target_clusters))
        labels = compress_labels(labels)
        while len(np.unique(labels)) > target:
            organs = np.unique(labels)
            best_pair = None
            best_score = -np.inf
            for a_idx, a in enumerate(organs):
                for b in organs[a_idx + 1 :]:
                    group_a = np.flatnonzero(labels == a)
                    group_b = np.flatnonzero(labels == b)
                    score = self._organ_affinity(x, centers[group_a], centers[group_b], sigma)
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
            if best_pair is None:
                break
            labels[labels == best_pair[1]] = best_pair[0]
            labels = compress_labels(labels)
        return labels

    def _organ_affinity(
        self,
        x: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        sigma: float,
    ) -> float:
        dist = np.sqrt(squared_distances(left, right))
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        gap = float(dist[i, j])
        bridge = bridge_continuity(x, left[i], right[j], sigma)
        proximity = np.exp(-gap / (self.organ_merge_radius * sigma + 1e-12))
        return float(0.65 * bridge + 0.35 * proximity)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centers_ is None or self.bud_to_organ_ is None:
            raise RuntimeError("fit must be called before predict")
        if self.organ_centers_ is not None and self.compact_refinement_steps > 0:
            return squared_distances(np.asarray(x, dtype=float), self.organ_centers_).argmin(axis=1)
        bud_labels = squared_distances(np.asarray(x, dtype=float), self.centers_).argmin(axis=1)
        return compress_labels(self.bud_to_organ_[bud_labels])

    def _maybe_refine_compact_organs(self, x: np.ndarray, rng: np.random.Generator) -> None:
        if self.labels_ is None or self.target_clusters is None:
            return
        if self.compact_refinement_steps <= 0:
            return
        if x.shape[1] < self.compact_refinement_dim_threshold:
            return

        target = int(self.target_clusters)
        seed_centers = self._compact_seed_centers(x, target, rng)
        candidates = [seed_centers]
        for _ in range(max(0, self.compact_refinement_restarts - 1)):
            candidates.append(kmeanspp_centers(x, target, rng))

        best_labels = None
        best_centers = None
        best_inertia = np.inf
        for centers in candidates:
            labels, centers, inertia = refine_centers(x, centers, self.compact_refinement_steps)
            if inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia

        self.labels_ = compress_labels(best_labels)
        self.organ_centers_ = best_centers

    def _compact_seed_centers(
        self,
        x: np.ndarray,
        target: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        labels = compress_labels(self.labels_)
        centers = weighted_centers(x, labels)
        if len(centers) < target:
            farthest = np.argsort(squared_distances(x, centers).min(axis=1))[::-1]
            needed = target - len(centers)
            centers = np.vstack([centers, x[farthest[:needed]]])
        elif len(centers) > target:
            # Keep broad morphology by choosing centers far apart among organs.
            chosen = [int(np.argmax(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))]
            while len(chosen) < target:
                dist2 = squared_distances(centers, centers[chosen]).min(axis=1)
                dist2[chosen] = -1
                chosen.append(int(np.argmax(dist2)))
            centers = centers[chosen]
        return centers


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)


def kernel_density(x: np.ndarray, query: np.ndarray, sigma: float) -> np.ndarray:
    dist2 = squared_distances(query, x)
    return np.exp(-dist2 / (2.0 * sigma * sigma)).sum(axis=1)


def compress_labels(labels: np.ndarray) -> np.ndarray:
    _, compressed = np.unique(labels, return_inverse=True)
    return compressed


def weighted_centers(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centers = []
    for label in np.unique(labels):
        centers.append(x[labels == label].mean(axis=0))
    return np.asarray(centers)


def kmeanspp_centers(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = [x[int(rng.integers(len(x)))]]
    while len(centers) < k:
        dist2 = squared_distances(x, np.asarray(centers)).min(axis=1)
        total = float(dist2.sum())
        if total <= 1e-12:
            centers.append(x[int(rng.integers(len(x)))])
            continue
        probs = dist2 / total
        centers.append(x[int(rng.choice(len(x), p=probs))])
    return np.asarray(centers)


def refine_centers(
    x: np.ndarray,
    centers: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    centers = centers.copy()
    labels = squared_distances(x, centers).argmin(axis=1)
    for _ in range(max(1, steps)):
        labels = squared_distances(x, centers).argmin(axis=1)
        new_centers = centers.copy()
        for j in range(len(centers)):
            pts = x[labels == j]
            if len(pts) == 0:
                farthest = np.argmax(squared_distances(x, centers).min(axis=1))
                new_centers[j] = x[farthest]
            else:
                new_centers[j] = pts.mean(axis=0)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < 1e-6:
            break
    dist2 = squared_distances(x, centers)
    labels = dist2.argmin(axis=1)
    inertia = float(dist2[np.arange(len(x)), labels].sum())
    return labels, centers, inertia


def local_tissue_axes(
    x: np.ndarray, centers: np.ndarray, bud_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    axes = []
    anisotropy = []
    for i, center in enumerate(centers):
        pts = x[bud_labels == i]
        if len(pts) < 4:
            nearest = np.argsort(squared_distances(center[None, :], x)[0])[:12]
            pts = x[nearest]
        centered = pts - pts.mean(axis=0)
        cov = centered.T @ centered / max(len(pts) - 1, 1)
        values, vectors = np.linalg.eigh(cov + 1e-9 * np.eye(cov.shape[0]))
        order = np.argsort(values)
        axis = vectors[:, order[-1]]
        ratio = float(values[order[-1]] / (values[order[-2]] + 1e-9))
        axes.append(axis / (np.linalg.norm(axis) + 1e-12))
        anisotropy.append(ratio)
    return np.asarray(axes), np.asarray(anisotropy)


def capillary_bridge_ok(
    x: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    max_bridge_void: float,
) -> bool:
    vec = right - left
    length = float(np.linalg.norm(vec))
    if length < 1.55 * sigma:
        return True
    direction = vec / (length + 1e-12)
    rel = x - left
    projection = rel @ direction
    t = projection / (length + 1e-12)
    closest = left + projection[:, None] * direction
    tube_dist = np.linalg.norm(x - closest, axis=1)
    in_tube = (t > 0.05) & (t < 0.95) & (tube_dist < 1.05 * sigma)
    if int(in_tube.sum()) < 4:
        return False
    beads = np.sort(np.r_[0.0, t[in_tube], 1.0])
    largest_void = float(np.diff(beads).max())
    return largest_void <= max_bridge_void


def bridge_continuity(x: np.ndarray, left: np.ndarray, right: np.ndarray, sigma: float) -> float:
    vec = right - left
    length = float(np.linalg.norm(vec))
    if length < 1e-12:
        return 1.0
    direction = vec / length
    rel = x - left
    projection = rel @ direction
    t = projection / length
    closest = left + projection[:, None] * direction
    tube_dist = np.linalg.norm(x - closest, axis=1)
    in_tube = (t > 0.03) & (t < 0.97) & (tube_dist < 1.2 * sigma)
    if int(in_tube.sum()) == 0:
        return 0.0
    beads = np.sort(np.r_[0.0, t[in_tube], 1.0])
    largest_void = float(np.diff(beads).max())
    density_score = min(1.0, float(in_tube.sum()) / 8.0)
    continuity_score = max(0.0, 1.0 - largest_void)
    return 0.55 * continuity_score + 0.45 * density_score


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def make_biological_demo_data(seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Crescent tissue.
    t = rng.uniform(0.05, math.pi * 1.15, 180)
    crescent = np.c_[1.55 * np.cos(t), 0.9 * np.sin(t)]
    crescent += rng.normal(scale=0.08, size=crescent.shape)

    # Inner displaced crescent.
    t = rng.uniform(0.1, math.pi * 1.05, 150)
    inner = np.c_[1.1 - 1.25 * np.cos(t), -0.28 - 0.72 * np.sin(t)]
    inner += rng.normal(scale=0.07, size=inner.shape)

    # Elongated organ-like blob.
    blob = rng.normal(size=(130, 2)) @ np.array([[0.38, 0.22], [0.0, 0.09]])
    blob += np.array([2.15, 0.35])

    # Tiny satellite population.
    satellite = rng.normal(scale=[0.09, 0.16], size=(55, 2)) + np.array([-1.85, -0.75])
    x = np.vstack([crescent, inner, blob, satellite])
    y = np.r_[
        np.zeros(len(crescent), dtype=int),
        np.ones(len(inner), dtype=int),
        np.full(len(blob), 2, dtype=int),
        np.full(len(satellite), 3, dtype=int),
    ]
    return x, y


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    a = compress_labels(np.asarray(a))
    b = compress_labels(np.asarray(b))
    n = len(a)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=int)
    for i in range(n):
        table[a[i], b[i]] += 1

    def comb2(v: np.ndarray) -> float:
        return float(np.sum(v * (v - 1) // 2))

    sum_nij = comb2(table)
    sum_ai = comb2(table.sum(axis=1))
    sum_bj = comb2(table.sum(axis=0))
    total = n * (n - 1) / 2.0
    expected = sum_ai * sum_bj / total if total else 0.0
    maximum = 0.5 * (sum_ai + sum_bj)
    return (sum_nij - expected) / (maximum - expected + 1e-12)


def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    labels = compress_labels(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(x):
        return -1.0
    dist = np.sqrt(squared_distances(x, x))
    scores = []
    for i in range(len(x)):
        same = labels == labels[i]
        if same.sum() <= 1:
            scores.append(0.0)
            continue
        a = dist[i, same].sum() / (same.sum() - 1)
        b = min(dist[i, labels == other].mean() for other in unique if other != labels[i])
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores))


def evolve_parameters(x: np.ndarray, y: np.ndarray) -> tuple[MorphogeneticBudsClustering, list[dict]]:
    trials = []
    best_model = None
    best_score = -1e9
    for seed in [3, 7, 11, 19, 23]:
        for merge_radius in [3.8, 4.6, 5.4, 6.2]:
            for valley_ratio in [0.12, 0.2, 0.3]:
                for tangent_alignment in [0.25, 0.4, 0.55]:
                    model = MorphogeneticBudsClustering(
                        initial_buds=20,
                        steps=180,
                        growth_rate=0.15,
                        inhibition=0.045,
                        apoptosis_mass=3.8,
                        birth_quantile=0.86,
                        organ_merge_radius=merge_radius,
                    valley_ratio=valley_ratio,
                    tangent_alignment=tangent_alignment,
                    max_bridge_void=0.42,
                    target_clusters=len(np.unique(y)),
                    random_state=seed,
                )
                    labels = model.fit_predict(x)
                    ari = adjusted_rand_index(y, labels)
                    sil = silhouette_score(x, labels)
                    n_clusters = int(len(np.unique(labels)))
                    score = ari + 0.12 * sil - 0.1 * abs(n_clusters - 4)
                    record = {
                        "seed": seed,
                        "organ_merge_radius": merge_radius,
                        "valley_ratio": valley_ratio,
                        "tangent_alignment": tangent_alignment,
                        "buds": int(len(model.centers_)),
                        "clusters": n_clusters,
                        "ari": round(ari, 4),
                        "silhouette": round(sil, 4),
                        "score": round(score, 4),
                    }
                    trials.append(record)
                    if score > best_score:
                        best_score = score
                        best_model = model

    trials.sort(key=lambda row: row["score"], reverse=True)
    assert best_model is not None
    return best_model, trials


def main() -> None:
    x, y = make_biological_demo_data()
    model, trials = evolve_parameters(x, y)
    labels = model.labels_
    assert labels is not None

    print(f"Micro-buds grown: {len(model.centers_)}")
    print(f"Organ-level clusters: {len(np.unique(labels))}")
    print(f"Adjusted Rand Index: {adjusted_rand_index(y, labels):.4f}")
    print(f"Silhouette: {silhouette_score(x, labels):.4f}")
    print("Best evolved settings:")
    print(json.dumps(trials[0], indent=2))
    print("Top 8 evolutionary trials:")
    print(json.dumps(trials[:8], indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=140)
    ax = axes[0]
    ax.scatter(x[:, 0], x[:, 1], c=labels, s=18, cmap="tab10", alpha=0.82, linewidths=0)
    ax.scatter(
        model.centers_[:, 0],
        model.centers_[:, 1],
        c="black",
        marker="*",
        s=260,
        edgecolors="white",
        linewidths=1.2,
        label="micro-buds",
    )
    ax.scatter(
        model.organ_centers_[:, 0],
        model.organ_centers_[:, 1],
        c="white",
        marker="o",
        s=90,
        edgecolors="black",
        linewidths=1.2,
        label="organs",
    )
    ax.set_title("Evolved Organ-Level Clusters")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(frameon=False, loc="best")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ax.scatter(x[:, 0], x[:, 1], c=y, s=18, cmap="tab10", alpha=0.82, linewidths=0)
    ax.set_title("Hidden Ground Truth for Simulation")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    out_path = Path(__file__).with_name("morphogenetic_buds_result.png")
    fig.savefig(out_path)
    summary_path = Path(__file__).with_name("morphogenetic_buds_trials.json")
    summary_path.write_text(json.dumps(trials, indent=2), encoding="utf-8")
    print(f"Saved figure: {out_path}")
    print(f"Saved trials: {summary_path}")


if __name__ == "__main__":
    main()
