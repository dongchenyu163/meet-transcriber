"""Whole-meeting clustering with a known number of speakers and revisable turns."""
import numpy as np
from scipy.optimize import linear_sum_assignment


class MeetingClustering:
    """Keep compact voice features, not audio; re-evaluate earlier assignments.

    The known count is passed to sherpa's actual clustering algorithm. Live
    results may use fewer groups until more voices arrive; the final pass uses
    the requested count, limited only by the number of usable observations.
    """
    def __init__(self, num_speakers, profiles=None, fit_budget=4096):
        if not 1 <= num_speakers <= 100:
            raise ValueError('已知发言人数必须在 1～100 之间。')
        if fit_budget < num_speakers:
            raise ValueError('声纹聚类预算不能小于人数。')
        self.num_speakers = num_speakers
        self.profiles = profiles or {}
        self.fit_budget = fit_budget
        self.observations = []
        self.previous = {}
        self.centers = {}

    @staticmethod
    def _cluster(features, count):
        import sherpa_onnx as so
        model = so.FastClustering(so.FastClusteringConfig(num_clusters=count, threshold=.5))
        # Native clustering normalizes its input in place.
        return np.asarray(model(np.array(features, dtype=np.float32, order='C', copy=True)))

    def update(self, observations=(), final=False):
        self.observations.extend(observations)
        usable = [i for i, item in enumerate(self.observations)
                  if item.get('vector') is not None and not item.get('overlap')]
        speakers, names = {}, {}
        if usable:
            features = np.asarray([self.observations[i]['vector'] for i in usable], dtype=np.float32)
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            if not np.isfinite(features).all() or np.any(norms < 1e-6):
                raise ValueError('无效的声纹。')
            features = features / norms
            # Bound the quadratic native distance matrix for long meetings.
            # Every observation still receives an assignment and can be revised.
            selected = np.unique(np.linspace(0, len(features)-1,
                min(len(features), self.fit_budget), dtype=int))
            fit = features[selected]
            count = min(self.num_speakers, len(fit))
            if not final and count > 1:
                count = min(count, len(np.unique(self._cluster(fit, -1))))
            labels = self._cluster(fit, count)
            clusters = sorted(set(labels), key=lambda label: int(np.flatnonzero(labels == label)[0]))
            centers = np.asarray([np.mean(fit[labels == label], axis=0) for label in clusters])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
            assigned = np.argmax(features @ centers.T, axis=1)
            # Preserve the hierarchical partition for all fitted observations.
            remap = {label: i for i, label in enumerate(clusters)}
            assigned[selected] = [remap[label] for label in labels]

            # Match the new partition to prior IDs by retained speaking time.
            # IDs stay useful across windows while membership remains revisable.
            identities = [f'SPEAKER_{i+1:02d}' for i in range(self.num_speakers)]
            scores = np.zeros((len(centers), len(identities)))
            for pos, observation_id in enumerate(usable):
                old = self.previous.get(observation_id, {}).get('speaker')
                if old in identities:
                    item = self.observations[observation_id]
                    scores[assigned[pos], identities.index(old)] += item['end'] - item['start']
            for j, identity in enumerate(identities):
                if identity in self.centers:
                    scores[:, j] += .001 * (centers @ self.centers[identity])
            rows, cols = linear_sum_assignment(-scores)
            mapping = {row: identities[col] for row, col in zip(rows, cols)}
            self.centers = {mapping[i]: center for i, center in enumerate(centers)}
            speakers = {observation_id: mapping[assigned[pos]] for pos, observation_id in enumerate(usable)}
            names = self._match_names()

        changed = []
        for observation_id, item in enumerate(self.observations):
            speaker = 'OVERLAP' if item.get('overlap') else speakers.get(observation_id, 'UNKNOWN')
            turn = {'id': f'voice-{observation_id}', 'start': item['start'], 'end': item['end'],
                    'speaker': speaker, 'speaker_name': names.get(speaker)}
            if self.previous.get(observation_id) != turn:
                changed.append(turn)
                self.previous[observation_id] = turn
        return changed

    def _match_names(self):
        """Names need enrolled samples; use one consistent name per global group."""
        candidates = []
        for speaker, center in self.centers.items():
            scores = sorted([(float(center @ (np.asarray(v) / np.linalg.norm(v))), name)
                             for name, v in self.profiles.items()], reverse=True)
            if scores and scores[0][0] >= .6 and (len(scores) == 1 or scores[0][0]-scores[1][0] >= .08):
                candidates.append((scores[0][0], speaker, scores[0][1]))
        result, used = {}, set()
        for _, speaker, name in sorted(candidates, reverse=True):
            if name not in used:
                result[speaker] = name
                used.add(name)
        return result
