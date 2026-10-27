"""Regression tests for comparative transition checkpoints 1 and 2."""

from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from LabGym.comparative_transition_analysis import (
	build_transition_dataset,  # noqa: F401  -- used by the remaining C1 cases
	load_recording,
    build_per_animal_measurements,
)


def write_result_workbook(parent, folder_name, filename, animals):
	"""Write one per-video result folder and an event-probability workbook.

	``animals`` maps each local animal-ID column to exported cell strings,
	for example ``"['Walk', 0.9]"`` and ``"['NA', -1]"``.
	"""
	folder = Path(parent) / folder_name
	folder.mkdir(parents=True, exist_ok=True)
	sequences = list(animals.values())
	nrow = len(sequences[0])
	if any(len(cells) != nrow for cells in sequences):
		raise ValueError("Animal sequences in one workbook must have equal length")
	frame = {"time/ID": list(range(nrow))}
	for animal_id, cells in animals.items():
		frame[str(animal_id)] = list(cells)
	pd.DataFrame(frame).to_excel(folder / filename, index=False)
	return folder


def test_na_cell_does_not_link_neighbors(tmp_path):
	folder = write_result_workbook(
		tmp_path,
		"recording",
		"all_event_probability.xlsx",
		{"0": ["['A', 0.9]", "['NA', -1]", "['B', 0.8]"]},
	)

	frame = load_recording(folder, "control", ["A", "B"])
	pair = frame[
		(frame["animal_id"] == "0")
		& (frame["source_behavior"] == "A")
		& (frame["target_behavior"] == "B")
	]

	assert len(pair) == 1
	assert pair.iloc[0]["transition_count"] == 0
	assert (frame["transition_count"] == 0).all()
	row = pair.iloc[0]
	assert row["group"] == "control"
	assert row["source_recording"] == str(folder.resolve())
	assert row["animal_id"] == "0"
	assert row["species"] is None or pd.isna(row["species"])


# TODO (Myra): complete the remaining C1 coverage
#
# - excluded behavior breaks a bout:
#   A, excluded, B must not create an A → B transition.
# - absent behavior remains represented with zero count and undefined probability:
#   a listed behavior the animal never performs still has outgoing rows with
#   transition_count 0, probability_defined False, and undefined probability.
# - a single-bout animal remains represented without invented transitions:
#   one bout (for example only A, or repeated A collapsed) stays in the result
#   with no positive transition counts.
# - recordings with different behavior labels do not merge animals or invent transitions:
#   mismatched vocabularies are rejected, and animals are not combined across recordings.
# - all_event_probability.xlsx is discovered for background subtraction:
#   that filename is read and species is missing.
# - multiple {species}_all_event_probability.xlsx files create separate species inputs:
#   each species file is its own input, identified by that species, not one merged animal.
# - batch all_events.xlsx is not selected:
#   all_events.xlsx alone is not a probability workbook, and it is ignored when a
#   real per-video workbook is also present.
# - matching local animal IDs in two recording folders remain distinct by source recording:
#   the same animal_id in two folders remains two identities, separated by source_recording.

def test_excluded_behavior_breaks_sequence(tmp_path):
    folder = write_result_workbook(
        tmp_path, "excluded", "all_event_probability.xlsx",
        {"0": ["['A', 0.9]", "['C', 0.9]", "['B', 0.9]"]},
    )
    result = load_recording(
        folder, "control", ["A", "B", "C"], excluded=["C"]
    )
    assert result["transition_count"].sum() == 0
    assert set(result["source_behavior"]) == {"A", "B"}
    assert set(result["target_behavior"]) == {"A", "B"}


def test_absent_behavior_has_undefined_probability(tmp_path):
    folder = write_result_workbook(
        tmp_path, "absent", "all_event_probability.xlsx",
        {"0": ["['A', 0.9]", "['B', 0.9]"]},
    )
    result = load_recording(folder, "control", ["A", "B", "C"])

    absent = result[result["source_behavior"] == "C"]
    assert len(absent) == 2
    assert (absent["transition_count"] == 0).all()
    assert (absent["outgoing_count"] == 0).all()
    assert (~absent["probability_defined"].astype(bool)).all()
    assert absent["transition_probability"].isna().all()

    # Observed zero differs from undefined probability.
    ac = result[
        (result["source_behavior"] == "A")
        & (result["target_behavior"] == "C")
    ].iloc[0]
    assert ac["transition_probability"] == 0.0
    assert bool(ac["probability_defined"])


def test_single_bout_animal(tmp_path):
    folder = write_result_workbook(
        tmp_path, "single", "all_event_probability.xlsx",
        {"0": ["['A', 0.9]"] * 4},
    )
    result = load_recording(folder, "control", ["A", "B", "C"])

    assert not result.empty
    assert set(result["animal_id"]) == {"0"}
    assert result["transition_count"].sum() == 0
    assert result["transition_probability"].isna().all()


def test_incompatible_behavior_sets(tmp_path):
    first = write_result_workbook(
        tmp_path, "first", "all_event_probability.xlsx",
        {"0": ["['A', 0.9]", "['B', 0.9]"]},
    )
    second = write_result_workbook(
        tmp_path, "second", "all_event_probability.xlsx",
        {"0": ["['B', 0.9]", "['D', 0.9]"]},
    )

    with pytest.raises(ValueError, match="Incompatible behavior sets"):
        build_transition_dataset([
            {
                "folder": first,
                "group": "control",
                "behaviors": ["A", "B", "C"],
            },
            {
                "folder": second,
                "group": "treated",
                "behaviors": ["A", "B", "D"],
            },
        ])


def test_background_workbook_discovery(tmp_path):
    folder = write_result_workbook(
        tmp_path, "background", "all_event_probability.xlsx",
        {"0": ["['A', 0.9]", "['B', 0.8]"]},
    )
    result = load_recording(folder, "control", ["A", "B"])

    assert result["transition_count"].sum() == 1
    assert result["species"].isna().all()
    assert set(result["source_recording"]) == {str(folder.resolve())}


def test_detector_species_workbooks(tmp_path):
    folder = write_result_workbook(
        tmp_path, "detector", "mouse_all_event_probability.xlsx",
        {"0": ["['A', 0.9]", "['B', 0.9]"]},
    )
    write_result_workbook(
        tmp_path, "detector", "fly_all_event_probability.xlsx",
        {"0": ["['B', 0.9]", "['A', 0.9]"]},
    )

    result = load_recording(folder, "control", ["A", "B"])

    assert set(result["species"]) == {"mouse", "fly"}
    assert set(result["animal_id"]) == {"0"}
    assert result["transition_count"].sum() == 2

    mouse = result[result["species"] == "mouse"]
    fly = result[result["species"] == "fly"]

    assert mouse.loc[
        (mouse["source_behavior"] == "A")
        & (mouse["target_behavior"] == "B"),
        "transition_count",
    ].iloc[0] == 1

    assert fly.loc[
        (fly["source_behavior"] == "B")
        & (fly["target_behavior"] == "A"),
        "transition_count",
    ].iloc[0] == 1


def test_batch_all_events_is_ignored(tmp_path):
    folder = write_result_workbook(
        tmp_path, "valid", "all_event_probability.xlsx",
        {"0": ["['A', 0.9]", "['B', 0.9]"]},
    )

    # Deliberately invalid: loader must never read this file.
    (folder / "all_events.xlsx").write_bytes(b"not an Excel file")

    result = load_recording(folder, "control", ["A", "B"])
    assert result["transition_count"].sum() == 1

    batch_only = tmp_path / "batch_only"
    batch_only.mkdir()
    (batch_only / "all_events.xlsx").write_bytes(b"not an Excel file")

    with pytest.raises(FileNotFoundError):
        load_recording(batch_only, "control", ["A", "B"])


def test_same_local_id_across_recordings(tmp_path):
    first = write_result_workbook(
        tmp_path, "video_01", "all_event_probability.xlsx",
        {"0": ["['A', 0.9]", "['B', 0.9]"]},
    )
    second = write_result_workbook(
        tmp_path, "video_02", "all_event_probability.xlsx",
        {"0": ["['B', 0.9]", "['A', 0.9]"]},
    )

    result = build_transition_dataset([
        {
            "folder": first,
            "group": "control",
            "behaviors": ["A", "B"],
        },
        {
            "folder": second,
            "group": "treated",
            "behaviors": ["A", "B"],
        },
    ])

    assert set(result["animal_id"]) == {"0"}
    assert set(result["source_recording"]) == {
        str(first.resolve()), str(second.resolve())
    }
    assert result["transition_count"].sum() == 2

    identities = result[
        ["group", "source_recording", "species", "animal_id"]
    ].drop_duplicates()
    assert len(identities) == 2

    first_ab = result[
        (result["source_recording"] == str(first.resolve()))
        & (result["source_behavior"] == "A")
        & (result["target_behavior"] == "B")
    ]
    second_ba = result[
        (result["source_recording"] == str(second.resolve()))
        & (result["source_behavior"] == "B")
        & (result["target_behavior"] == "A")
    ]

    assert first_ab.iloc[0]["transition_count"] == 1
    assert second_ba.iloc[0]["transition_count"] == 1



def recording(tmp_path, name, animals, fps=10, species=None):
    folder = tmp_path / name
    folder.mkdir()
    file = f'{species}_all_event_probability.xlsx' if species else 'all_event_probability.xlsx'
    pd.DataFrame({'time/ID': np.arange(len(next(iter(animals.values())))) / fps,
                  **{k: [str([v, 0.9]) for v in values] for k, values in animals.items()}}
                 ).to_excel(folder / file, index=False)
    return folder


def configs(a, b, **kwargs):
    return [dict(folder=a, group='control', behaviors=['A','B','C'], **kwargs),
            dict(folder=b, group='treated', behaviors=['A','B','C'], **kwargs)]


def test_bouts_and_switching_rate(tmp_path):
    a = recording(tmp_path, 'a', {'0':['A','A','B','B','NA','A','C']})
    b = recording(tmp_path, 'b', {'0':['B','B','B']})
    t, s, bouts = build_per_animal_measurements(configs(a,b, fps=10))
    ta = t[t.source_recording == str(a.resolve())]
    assert ta.transition_count.sum() == 2
    assert ta.valid_included_frames.iloc[0] == 6
    assert ta.switching_rate_per_1000_frames.iloc[0] == pytest.approx(2000/6)
    aa = s[(s.source_recording == str(a.resolve())) & (s.behavior == 'A')].iloc[0]
    assert aa.bout_count == 2
    assert aa.total_bout_frames == 3
    assert aa.mean_bout_duration_seconds == pytest.approx(.15)
    assert aa.median_bout_duration_seconds == pytest.approx(.15)
    assert sorted(bouts[bouts.source_recording == str(a.resolve())].bout_length_frames) == [1,1,2,2]


def test_no_pooling_same_local_id(tmp_path):
    a = recording(tmp_path, 'a', {'0':['A','B'], '1':['A','A']})
    b = recording(tmp_path, 'b', {'0':['B','A']})
    t, s, _ = build_per_animal_measurements(configs(a,b, fps=10))
    assert len(s[['source_recording','animal_id']].drop_duplicates()) == 3
    assert t.transition_count.sum() == 2
    assert s[(s.source_recording == str(a.resolve())) & (s.animal_id == '1')].animal_transition_count.eq(0).all()


def test_excluded_and_all_na(tmp_path):
    a = recording(tmp_path, 'a', {'0':['A','C','B'], '1':['NA','NA','NA']})
    b = recording(tmp_path, 'b', {'0':['A','B']})
    t, s, _ = build_per_animal_measurements(configs(a,b, excluded=['C'], fps=10))
    ta = t[t.source_recording == str(a.resolve())]
    assert ta.transition_count.sum() == 0
    assert s[(s.source_recording == str(a.resolve())) & (s.animal_id == '1')].valid_included_frames.eq(0).all()
    assert s[(s.source_recording == str(a.resolve())) & (s.animal_id == '1')].switching_rate_per_1000_frames.isna().all()


def test_mixed_fps_preserves_both_recordings(tmp_path):
    a = recording(tmp_path, 'a', {'0':['A','A','B','B']}, fps=10)
    b = recording(tmp_path, 'b', {'0':['B','B','A','A']}, fps=20)
    out = tmp_path / 'export'
    (a / 'Analysis log.txt').write_text('Video fps: 10\n')
    (b / 'Analysis log.txt').write_text('Video fps: 20\n')
    t, s, bouts = build_per_animal_measurements(configs(a, b), output_folder=out)
    for frame in (t, s, bouts):
        assert set(frame['fps']) == {10.0, 20.0}
        assert set(frame['fps_source']) == {'Analysis log.txt'}
        assert len(set(frame['source_recording'])) == 2
    assert set(pd.read_csv(out / 'per_animal_transitions.csv')['fps']) == {10, 20}
    assert set(pd.read_csv(out / 'per_animal_behaviors.csv')['fps']) == {10, 20}
    assert set(pd.read_csv(out / 'per_animal_bouts.csv')['fps']) == {10, 20}
    a_bout = bouts[bouts.source_recording == str(a.resolve())]
    b_bout = bouts[bouts.source_recording == str(b.resolve())]
    assert a_bout.bout_duration_seconds.iloc[0] == pytest.approx(0.2)
    assert b_bout.bout_duration_seconds.iloc[0] == pytest.approx(0.1)


def test_rounded_30_fps_time_index(tmp_path):
    a = recording(tmp_path, 'a', {'0':['A'] * 120}, fps=30)
    b = recording(tmp_path, 'b', {'0':['B'] * 120}, fps=30)
    for folder in (a, b):
        path = folder / 'all_event_probability.xlsx'
        df = pd.read_excel(path)
        df['time/ID'] = np.round(np.arange(len(df)) / 30, 2)
        df.to_excel(path, index=False)
    t, s, bouts = build_per_animal_measurements(configs(a, b))
    assert set(t.fps) == {30.0}
    assert set(s.fps) == {30.0}
    assert set(bouts.fps) == {30.0}
    assert set(t.fps_source) == {'time/ID (centisecond-rounded)'}
    assert s.mean_bout_duration_seconds.eq(4.0).any()


def test_invalid_time_index_rejected(tmp_path):
    a = recording(tmp_path, 'a', {'0':['A'] * 6}, fps=30)
    b = recording(tmp_path, 'b', {'0':['B'] * 6}, fps=30)
    for times in ([0, 0.03, 0.03, 0.10, 0.13, 0.17],
                  [0, 0.03, 0.02, 0.10, 0.13, 0.17],
                  [0, 1, 2, 3, 4, 5]):
        path = a / 'all_event_probability.xlsx'
        df = pd.read_excel(path)
        df['time/ID'] = times
        df.to_excel(path, index=False)
        with pytest.raises(ValueError):
            build_per_animal_measurements(configs(a, b))


def test_species_and_export(tmp_path):
    a = recording(tmp_path, 'a', {'0':['A','B']}, species='mouse')
    b = recording(tmp_path, 'b', {'0':['B','A']}, species='fly')
    out = tmp_path / 'output'
    t, s, bouts = build_per_animal_measurements(configs(a,b, fps=10), output_folder=out)
    assert set(t.species) == {'mouse','fly'}
    assert all((out / n).exists() for n in ('per_animal_transitions.csv','per_animal_behaviors.csv','per_animal_bouts.csv'))


def test_explicit_fps_overrides_ambiguous_short_index(tmp_path):
    a = recording(tmp_path, 'a', {'0': ['A', 'B']}, fps=10)
    b = recording(tmp_path, 'b', {'0': ['B', 'A']}, fps=10)
    transitions, behaviors, bouts = build_per_animal_measurements(
        configs(a, b, fps=10)
    )
    for frame in (transitions, behaviors, bouts):
        assert set(frame['fps']) == {10.0}
        assert set(frame['fps_source']) == {'explicit'}


def test_analysis_log_fps_preserved(tmp_path):
    a = recording(tmp_path, 'a', {'0': ['A', 'B']}, fps=10)
    b = recording(tmp_path, 'b', {'0': ['B', 'A']}, fps=10)
    for folder in (a, b):
        (folder / 'Analysis log.txt').write_text('Video fps: 10\n')
    transitions, behaviors, bouts = build_per_animal_measurements(configs(a, b))
    for frame in (transitions, behaviors, bouts):
        assert set(frame['fps']) == {10.0}
        assert set(frame['fps_source']) == {'Analysis log.txt'}


@pytest.mark.parametrize("length", [3, 4, 5, 6])
def test_ambiguous_short_rounded_30fps_requires_metadata(tmp_path, length):
    a = recording(tmp_path, 'a', {'0': ['A'] * length}, fps=30)
    b = recording(tmp_path, 'b', {'0': ['B'] * length}, fps=30)
    for folder in (a, b):
        path = folder / 'all_event_probability.xlsx'
        df = pd.read_excel(path)
        df['time/ID'] = np.round(np.arange(length) / 30, 2)
        df.to_excel(path, index=False)
    with pytest.raises(ValueError, match='Ambiguous short rounded time/ID'):
        build_per_animal_measurements(configs(a, b))


def test_seven_rounded_30fps_timestamps_resolve(tmp_path):
    a = recording(tmp_path, 'a', {'0': ['A'] * 7}, fps=30)
    b = recording(tmp_path, 'b', {'0': ['B'] * 7}, fps=30)
    for folder in (a, b):
        path = folder / 'all_event_probability.xlsx'
        df = pd.read_excel(path)
        df['time/ID'] = np.round(np.arange(7) / 30, 2)
        df.to_excel(path, index=False)
    transitions, behaviors, bouts = build_per_animal_measurements(configs(a, b))
    for frame in (transitions, behaviors, bouts):
        assert set(frame['fps']) == {30.0}
        assert set(frame['fps_source']) == {'time/ID (centisecond-rounded)'}
