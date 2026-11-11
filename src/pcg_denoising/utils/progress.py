import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, is_dataclass

MODES = ("resume", "restart")


class ProgressError(RuntimeError):
    pass


def fingerprint(*objects):
    """Short hash of dataclasses / dicts / plain values (Path and other types go through str).
    Same content -> same hash, any change -> a different one."""
    data = [asdict(o) if is_dataclass(o) else o for o in objects]
    text = json.dumps(data, default=str, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


class Progress:
    """Remembers which stages of a run are finished, in <run_dir>/progress.json.

    mode="resume":  keep what is recorded; finished stages are skipped.
    mode="restart": forget everything and run every stage again (outputs are overwritten).

    fingerprint: hash of the settings the whole run depends on, e.g. fingerprint(cfg).
    Resuming with a different one raises ProgressError, because the finished stages would belong
    to other settings. A stage can also carry its own `deps` (the settings only it depends on).
    """

    def __init__(self, run_dir, fingerprint=None, mode="resume"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        self.path = Path(run_dir) / "progress.json"
        self.restarted = False
        self._stages = {}
        self._fingerprint = fingerprint

        if mode == "restart":
            self.restarted = True
            self._save()
            return

        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("fingerprint") != fingerprint:
                raise ProgressError(
                    f"The settings changed since this run started ({data.get('fingerprint')} -> {fingerprint}). "
                    f"Use mode='restart' to run it again from the beginning, or a new experiment name.")
            self._stages = data["stages"]
        else:
            self._save()

    # ---- queries ---------------------------------------------------------------------------------
    def done(self, key, deps=None, requires=()):
        """True if `key` was marked, with the same deps, and every path in `requires` still exists."""
        entry = self._stages.get(key)
        if entry is None or entry["deps"] != self._hash(deps):
            return False
        return all(Path(p).exists() for p in requires)

    @property
    def stages(self):
        return list(self._stages)

    # ---- updates ---------------------------------------------------------------------------------
    def mark(self, key, deps=None):
        """Record that `key` finished. Call it after its output is complete."""
        self._stages[key] = {"deps": self._hash(deps), "time": datetime.now().isoformat(timespec="seconds")}
        self._save()

    def run(self, key, fn, *args, deps=None, requires=(), **kwargs):
        """Call fn(*args, **kwargs) unless `key` is already done, then mark it.
        Returns True if fn ran and False if it was skipped. If fn raises, nothing is marked."""
        if self.done(key, deps, requires):
            print(f"Skipping {key} (already done)")
            return False
        fn(*args, **kwargs)
        self.mark(key, deps)
        return True

    def release(self, key, paths):
        """Delete intermediate files (files or whole folders) that later stages no longer need,
        and record on `key` that they were removed on purpose. Safe to call again: missing paths
        are ignored, so it also finishes a release that was interrupted half-way.
        Only paths inside the run folder are accepted. Returns the paths that were deleted.

        The stage stays marked as done, but done(key, requires=paths) is False from now on, so a
        stage that needs those files again is rebuilt; mark() on a rebuild clears the record."""
        root = self.path.parent.resolve()
        removed = []
        for p in map(Path, paths):
            if root not in p.resolve().parents:
                raise ProgressError(f"Refusing to delete {p}: it is not inside {root}")
            if p.is_dir():
                shutil.rmtree(p)
                removed.append(p)
            elif p.exists():
                p.unlink()
                removed.append(p)

        if key in self._stages:
            self._stages[key]["released"] = datetime.now().isoformat(timespec="seconds")
            self._save()
        return removed

    def released(self, key):
        """True if the files of `key` were deleted on purpose with release()."""
        return "released" in self._stages.get(key, {})

    def forget(self, prefix):
        """Forget every stage whose key starts with prefix, e.g. 'fold_3/' to redo one fold."""
        self._stages = {k: v for k, v in self._stages.items() if not k.startswith(prefix)}
        self._save()

    # ---- internals ---------------------------------------------------------------------------------
    @staticmethod
    def _hash(deps):
        return None if deps is None else fingerprint(deps)

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.json")
        tmp.write_text(json.dumps({"fingerprint": self._fingerprint, "stages": self._stages}, indent=2))
        tmp.replace(self.path)