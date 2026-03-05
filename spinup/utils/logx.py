import json
import os

import numpy as np


class EpochLogger:
    def __init__(self, output_dir=None, exp_name=None, **kwargs):
        self.output_dir = output_dir or os.path.join(os.getcwd(), "data", exp_name or "experiment")
        os.makedirs(self.output_dir, exist_ok=True)
        self.exp_name = exp_name
        self._store = {}

    def save_config(self, config):
        path = os.path.join(self.output_dir, "config.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, default=str)
        except Exception:
            pass

    def store(self, **kwargs):
        for k, v in kwargs.items():
            self._store.setdefault(k, []).append(v)

    def setup_tf_saver(self, sess, inputs=None, outputs=None):
        self._sess = sess
        self._inputs = inputs or {}
        self._outputs = outputs or {}

    def save_state(self, state_dict=None, itr=None):
        path = os.path.join(self.output_dir, "state_meta.json")
        payload = {"itr": itr, "has_state": state_dict is not None}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        if val is None:
            vals = self._store.get(key, [])
            if len(vals) == 0:
                val = 0.0
            else:
                arr = np.asarray(vals, dtype=np.float64)
                val = float(np.mean(arr))
        print(f"{key}: {val}")

    def dump_tabular(self):
        print("----")
        self._store = {}
