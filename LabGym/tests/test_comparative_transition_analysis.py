"""Scaffold for comparative-transition C1 workbook loading."""

from pathlib import Path

import pandas as pd

from LabGym.comparative_transition_analysis import (
	build_transition_dataset,  # noqa: F401  -- used by the remaining C1 cases
	load_recording,
)


def write_result_workbook(parent, folder_name, filename, animals):
	"""Write one per-video result folder and an event-probability workbook.

	``animals`` maps each local animal-ID column to exported cell strings,
	for example ``"['Walk', 0.9]"`` and ``"['NA', -1]"``.
	"""
	folder = Path(parent) / folder_name
	folder.mkdir(parents=True)
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
