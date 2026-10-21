"""video-level LabGym transition inputs and identity contract."""

import ast
from pathlib import Path
import numpy as np
import pandas as pd

def _event(value):
    if isinstance(value, (list, tuple)):
        item = value
    elif pd.isna(value):
        return None
    else:
        try:
            item = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            raise ValueError(f"Invalid event cell: {value!r}") from None
    if not isinstance(item, (list, tuple)) or not item:
        raise ValueError(f"Invalid event: {value!r}")
    state = item[0]
    if state is None or str(state).strip().upper() in ("NA", "NAN", "NONE", ""):
        return None
    return str(state)

def load_recording(folder, group, behaviors, excluded=(), collapse_repeats=True,
                   include_self=False):
    """Read all supported probability workbooks from one analyzed-video folder.

    Background-subtraction: all_event_probability.xlsx.
    Detector: {species}_all_event_probability.xlsx (one input per species).
    Unrelated workbooks, including batch all_events.xlsx, are ignored.
    """
    folder = Path(folder)
    background = folder / "all_event_probability.xlsx"
    suffix = "_all_event_probability.xlsx"
    detector_paths = sorted(
        p for p in folder.glob("*" + suffix)
        if p.name != background.name and p.name[:-len(suffix)]
    )
    paths = ([(background, None)] if background.is_file() else []) + [
        (p, p.name[:-len(suffix)]) for p in detector_paths
    ]
    if not paths:
        raise FileNotFoundError(
            f"No per-video event probability workbook found in {folder}"
        )
    frames = []
    for path, species in paths:
        df = pd.read_excel(path)
        frames.append(_load_workbook(df, path, folder, group, species, behaviors,
                                     excluded, collapse_repeats, include_self))
    return pd.concat(frames, ignore_index=True)


def _load_workbook(df, path, folder, group, species, behaviors, excluded,
                   collapse_repeats, include_self):
    if "time/ID" not in df.columns:
        raise ValueError(f"Missing time/ID in {path}")
    if not df["time/ID"].is_unique:
        raise ValueError(f"Duplicate timestamps in {path}")
    vocab = list(behaviors)
    if not vocab or len(vocab) != len(set(vocab)):
        raise ValueError("Behavior vocabulary must be nonempty and unique")
    excluded = set(excluded)
    if not excluded.issubset(vocab):
        raise ValueError(f"Unknown excluded behaviors: {excluded - set(vocab)}")
    selected = [b for b in vocab if b not in excluded]
    if not selected:
        raise ValueError("No behaviors remain after exclusions")
    rows = []
    for col in df.columns:
        if col == "time/ID":
            continue
        animal_id = str(col)
        sequence = []
        bouts = []
        def flush():
            if sequence:
                bouts.append(sequence.copy())
                sequence.clear()
        for value in df[col]:
            state = _event(value)
            if state is None or state in excluded:
                flush()
                continue
            if state not in vocab:
                raise ValueError(f"Unknown behavior {state!r} in {path}, animal {animal_id}")
            if not collapse_repeats or not sequence or sequence[-1] != state:
                sequence.append(state)
        flush()
        counts = {(a,b): 0 for a in selected for b in selected
                  if include_self or a != b}
        for bout in bouts:
            for a,b in zip(bout, bout[1:]):
                if (a,b) in counts:
                    counts[a,b] += 1
        for a in selected:
            denominator = sum(n for (src,_),n in counts.items() if src == a)
            for b in selected:
                if not include_self and a == b:
                    continue
                count = counts[a,b]
                rows.append(dict(group=str(group), source_recording=str(folder.resolve()),
                                 species=species, animal_id=animal_id, source_behavior=a,
                                 target_behavior=b, transition_count=count,
                                 outgoing_count=denominator,
                                 transition_probability=(count/denominator
                                                         if denominator else np.nan),
                                 probability_defined=bool(denominator)))
    return pd.DataFrame(rows)

def build_transition_dataset(recordings, output_csv=None, collapse_repeats=True,
                             include_self=False):
    """recordings: iterable of dicts with folder, group, behaviors, excluded(optional).

    Requires the exact same full categorizer vocabulary across recordings.
    Group membership is supplied explicitly; never inferred from filenames.
    """
    recordings = list(recordings)
    if not recordings:
        raise ValueError("No recordings supplied")
    groups = {str(r["group"]) for r in recordings}
    if len(groups) != 2:
        raise ValueError("First version requires exactly two groups")
    expected = set(recordings[0]["behaviors"])
    excluded = set(recordings[0].get("excluded", ()))
    frames = []
    identities = set()
    for r in recordings:
        if set(r["behaviors"]) != expected:
            raise ValueError(f"Incompatible behavior sets: {r['folder']}")
        if set(r.get("excluded", ())) != excluded:
            raise ValueError("All recordings must use the same exclusions")
        folder = Path(r["folder"]).resolve()
        identity = (str(r["group"]), str(folder))
        if identity in identities:
            raise ValueError(f"Duplicate recording: {folder}")
        identities.add(identity)
        frame = load_recording(folder, r["group"], r["behaviors"], excluded,
                               collapse_repeats, include_self)
        # Use the resolved recording path to avoid collisions between equal basenames.
        frame["source_recording"] = str(folder)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if output_csv is not None:
        dest = Path(output_csv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(dest, index=False)
    return result
