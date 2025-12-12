"""Centroid-based object tracker to keep persistent IDs across frames."""
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np


class CentroidTracker:
    """Simple centroid tracker adapted from PyImageSearch examples."""

    def __init__(self, max_disappeared: int = 15, max_distance: float = 80.0):
        self.next_object_id = 0
        self.objects: Dict[int, Tuple[int, int, int, int]] = OrderedDict()
        self.disappeared: Dict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, rect: Tuple[int, int, int, int]):
        self.objects[self.next_object_id] = rect
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, rects: List[Tuple[int, int, int, int]]):
        """Update tracked objects using new bounding boxes."""
        if len(rects) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for rect in rects:
                self.register(rect)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = self._compute_centroids(list(self.objects.values()))
        input_centroids = self._compute_centroids(rects)

        distances = self._pairwise_distances(object_centroids, input_centroids)
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for row, col in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = rects[col]
            self.disappeared[object_id] = 0

            assigned_rows.add(row)
            assigned_cols.add(col)

        unassigned_rows = set(range(distances.shape[0])).difference(assigned_rows)
        for row in unassigned_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unassigned_cols = set(range(len(rects))).difference(assigned_cols)
        for col in unassigned_cols:
            self.register(rects[col])

        return self.objects

    @staticmethod
    def _compute_centroids(rects: List[Tuple[int, int, int, int]]):
        centroids = []
        for (x, y, w, h) in rects:
            c_x = int(x + w / 2)
            c_y = int(y + h / 2)
            centroids.append((c_x, c_y))
        return np.array(centroids)

    @staticmethod
    def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        return np.linalg.norm(diff, axis=2)
