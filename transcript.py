"""The fixed transcript every system is run against.

One user, a coding agent, two sessions. Session 1 is January. Session 2 is September,
eight months later, with an empty message history. Whatever survives is whatever the
memory system wrote down.

Nothing in this file is system-specific. Every runner imports it unchanged so the
comparison is over one transcript rather than one per library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

USER_ID = "aditya"

SESSION_1_AT = datetime(2026, 1, 10, 10, 0, tzinfo=timezone.utc)
SESSION_2_AT = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Write:
    """A turn that is handed to the memory system to store."""

    id: str
    at: datetime
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class Probe:
    """A search issued with one turn's text, and what the store ought to do with it.

    ``kind`` follows the three-way split in the Memory Weave README:

    - ``answerable``   the store holds the answer and should return it (type 1)
    - ``absent``       the store holds nothing on the subject; empty is correct (type 2)
    - ``ordinary``     nobody asked about the past; empty is correct (type 3)

    Type 3 is the one no public memory benchmark measures, because LoCoMo and
    LongMemEval only ever search on benchmark questions. It is the reason this
    harness exists.
    """

    id: str
    text: str
    kind: str
    note: str
    expect_empty: bool
    stage: str = "after_update"
    dated: bool = False


SESSION_1 = Write(
    id="s1",
    at=SESSION_1_AT,
    messages=[
        {
            "role": "user",
            "content": (
                "I write tests in pytest. Commit messages stay conventional. "
                "Rohan is on-call; he works at Nimbus, on payments."
            ),
        },
        {"role": "assistant", "content": "Okay."},
    ],
)

SESSION_2_UPDATE = Write(
    id="s2-update",
    at=SESSION_2_AT,
    messages=[
        {
            "role": "user",
            "content": "I switched to unittest. Rohan left Nimbus in April and joined Lattice.",
        },
        {"role": "assistant", "content": "Noted."},
    ],
)

PROBES: list[Probe] = [
    Probe(
        id="p1_latency_graph",
        text="The checkout latency graph is red.",
        kind="ordinary",
        note="First September turn. Nobody mentioned pytest, commits, or Rohan.",
        expect_empty=True,
        stage="after_session_1",
    ),
    Probe(
        id="p2_retry_count",
        text="Bump the retry count to 3.",
        kind="ordinary",
        note="The headline turn. A real request with no memory question in it.",
        expect_empty=True,
    ),
    Probe(
        id="p3_test_runner",
        text="What do I write tests in?",
        kind="answerable",
        note="Control. The store holds this and should return it, or the floor is too high.",
        expect_empty=False,
    ),
    Probe(
        id="p4_dog_name",
        text="What is my dog's name?",
        kind="absent",
        note="Control. Nothing in the store is about a dog. Easy empty.",
        expect_empty=True,
    ),
    Probe(
        id="p5_march_employer",
        text="Where was Rohan working in March?",
        kind="answerable",
        note="Dated question. Rohan left Nimbus in April, so March is Nimbus.",
        expect_empty=False,
        dated=True,
    ),
]

PROBES_BY_STAGE = {
    "after_session_1": [p for p in PROBES if p.stage == "after_session_1"],
    "after_update": [p for p in PROBES if p.stage == "after_update"],
}

# What a reader should be able to check the dumps against.
GROUND_TRUTH = {
    "test_runner_january": "pytest",
    "test_runner_september": "unittest",
    "commit_style": "conventional",
    "rohan_employer_january": "Nimbus",
    "rohan_employer_september": "Lattice",
    "rohan_employer_march": "Nimbus",
    "rohan_left_nimbus": "April 2026",
}
