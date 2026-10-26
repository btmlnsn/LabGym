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



"""Checkpoint 2: traceable per-animal transition and bout measurements.

No group aggregation or inferential statistics. C1 counting is reused unchanged.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

IDENTITY = ['group', 'source_recording', 'species', 'animal_id']


def _workbooks(folder):
    suffix = '_all_event_probability.xlsx'
    paths = list(Path(folder).glob('*' + suffix))
    background = Path(folder) / 'all_event_probability.xlsx'
    if background.is_file():
        paths.append(background)
    return [(p, None if p.name == background.name else p.name[:-len(suffix)])
            for p in sorted(set(paths)) if p.name == background.name or p.name[:-len(suffix)]]


def _fps(folder, time_index, explicit=None):
    """Resolve FPS: explicit value, analysis log, or rounded time/ID seconds.

    Timestamp inference requires at least three timestamps and a unique,
    plausible integer FPS matching centisecond-rounded frame times.
    """
    if explicit is not None:
        fps, source = float(explicit), 'explicit'
    else:
        fps, source = None, None
        log = Path(folder) / 'Analysis log.txt'
        if log.is_file():
            match = re.search(
                r'Video\s*fps\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
                log.read_text(errors='replace'), re.I
            )
            if match:
                fps, source = float(match.group(1)), 'Analysis log.txt'

        if fps is None:
            times = pd.to_numeric(
                pd.Series(time_index), errors='coerce'
            ).to_numpy(dtype=float)
            if len(times) < 3 or not np.isfinite(times).all():
                raise ValueError(
                    f'Cannot infer fps from time/ID for {folder}; supply fps explicitly'
                )
            diffs = np.diff(times)
            if not (diffs > 0).all():
                raise ValueError(f'Invalid time/ID for {folder}: non-increasing timestamps')
            if np.median(diffs) >= 1:
                raise ValueError(
                    f'Ambiguous time/ID units for {folder}; supply fps explicitly'
                )

            # LabGym rounds frame timestamps to hundredths of a second.
            # Thus 30 fps produces alternating 0.03 and 0.04 second steps.
            # Fit against all frames, not the individual differences.
            frame = np.arange(len(times), dtype=float)
            slope = np.polyfit(frame, times, 1)[0]
            if not np.isfinite(slope) or slope <= 0:
                raise ValueError(f'Invalid time/ID for {folder}')
            candidate = int(round(1.0 / slope))
            if candidate < 2 or candidate > 100:
                raise ValueError(
                    f'Ambiguous time/ID fps for {folder}; supply fps explicitly'
                )

            # Allow a constant start-time offset and centisecond rounding.
            # The total drift must remain within one centisecond.
            offsets = times - frame / candidate
            if (np.ptp(offsets) > 0.0100001
                    or abs(slope - 1.0 / candidate) * (len(times) - 1) > 0.0100001):
                raise ValueError(
                    f'Ambiguous time/ID fps for {folder}; supply fps explicitly'
                )
            fps, source = float(candidate), 'time/ID (centisecond-rounded)'

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    return fps, source


def _bouts(cells, selected):
    """Consecutive equal labels form one bout; NA/excluded labels terminate bouts."""
    bouts = []
    current, length = None, 0
    valid_frames = 0
    for cell in cells:
        state = _event(cell)
        if state in selected:
            valid_frames += 1
            if state == current:
                length += 1
            else:
                if current is not None:
                    bouts.append((current, length))
                current, length = state, 1
        else:
            if current is not None:
                bouts.append((current, length))
            current, length = None, 0
    if current is not None:
        bouts.append((current, length))
    return bouts, valid_frames


def build_per_animal_measurements(recordings, output_folder=None,
                                  collapse_repeats=True, include_self=False,
                                  fps_tolerance=0.01):
    """Return (transitions, behaviors, bouts), all tidy per-animal DataFrames.

    Recordings follow C1's {folder, group, behaviors, excluded?} contract;
    optional fps overrides missing/ambiguous metadata. Mixed FPS is allowed.
    fps_tolerance is retained for backwards compatibility but not enforced.
    """
    recordings = list(recordings)
    if not recordings or len({str(r['group']) for r in recordings}) != 2:
        raise ValueError('Exactly two groups are required')
    expected = set(recordings[0]['behaviors'])
    excluded = set(recordings[0].get('excluded', ()))
    if fps_tolerance < 0:
        raise ValueError('fps_tolerance must be nonnegative')
    transitions, behaviors, bout_rows = [], [], []
    seen = set()
    for rec in recordings:
        folder = Path(rec['folder']).resolve()
        if (str(rec['group']), str(folder)) in seen:
            raise ValueError(f'Duplicate recording: {folder}')
        seen.add((str(rec['group']), str(folder)))
        if set(rec['behaviors']) != expected:
            raise ValueError('Incompatible behavior sets')
        if set(rec.get('excluded', ())) != excluded:
            raise ValueError('Incompatible exclusions')
        selected = [b for b in rec['behaviors'] if b not in excluded]
        if not selected:
            raise ValueError('No included behaviors')
        # Keep C1 transition semantics, counts and denominators unchanged.
        c1 = load_recording(folder, rec['group'], rec['behaviors'], excluded,
                            collapse_repeats, include_self)
        workbook_paths = _workbooks(folder)
        for path, species in workbook_paths:
            df = pd.read_excel(path)
            fps, fps_source = _fps(folder, df['time/ID'], rec.get('fps'))
            for col in df.columns:
                if col == 'time/ID':
                    continue
                animal_id = str(col)
                identity = dict(group=str(rec['group']), source_recording=str(folder),
                                species=species, animal_id=animal_id)
                bouts, valid_frames = _bouts(df[col], set(selected))
                for number, (state, length) in enumerate(bouts, start=1):
                    bout_rows.append(dict(**identity, behavior=state, bout_number=number,
                                          bout_length_frames=length, bout_duration_seconds=length/fps,
                                          fps=fps, fps_source=fps_source))
                subset = c1[(c1['animal_id'] == animal_id) &
                            (c1['species'].isna() if species is None else c1['species'].eq(species))].copy()
                total_transitions = int(subset['transition_count'].sum())
                rate = 1000 * total_transitions / valid_frames if valid_frames else np.nan
                subset['valid_included_frames'] = valid_frames
                subset['animal_transition_count'] = total_transitions
                subset['switching_rate_per_1000_frames'] = rate
                subset['fps'] = fps
                subset['fps_source'] = fps_source
                transitions.append(subset)
                for behavior in selected:
                    lengths = [length for state, length in bouts if state == behavior]
                    durations = np.asarray(lengths, dtype=float) / fps
                    behaviors.append(dict(**identity, behavior=behavior,
                                          valid_included_frames=valid_frames,
                                          animal_transition_count=total_transitions,
                                          switching_rate_per_1000_frames=rate,
                                          bout_count=len(lengths),
                                          total_bout_frames=sum(lengths),
                                          mean_bout_frames=float(np.mean(lengths)) if lengths else np.nan,
                                          median_bout_frames=float(np.median(lengths)) if lengths else np.nan,
                                          mean_bout_duration_seconds=float(np.mean(durations)) if lengths else np.nan,
                                          median_bout_duration_seconds=float(np.median(durations)) if lengths else np.nan,
                                          fps=fps, fps_source=fps_source))
    transition_df = pd.concat(transitions, ignore_index=True)
    behavior_df = pd.DataFrame(behaviors)
    bout_df = pd.DataFrame(bout_rows, columns=IDENTITY + ['behavior', 'bout_number',
        'bout_length_frames', 'bout_duration_seconds', 'fps', 'fps_source'])
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        transition_df.to_csv(output_folder / 'per_animal_transitions.csv', index=False)
        behavior_df.to_csv(output_folder / 'per_animal_behaviors.csv', index=False)
        bout_df.to_csv(output_folder / 'per_animal_bouts.csv', index=False)
    return transition_df, behavior_df, bout_df
