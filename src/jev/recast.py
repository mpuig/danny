"""Recast public classification datasets into the state/question/label schema.

Each record: {"state": str, "question": Question, "label": int, "task": str}
where label indexes the choice options (criteria order), score levels, or 0/1
for noul (0 = no, 1 = yes).

Tasks carry several instruction phrasings; phrasing 0 is canonical and used in
evals, the rest add training diversity. Training tasks: ag_news, dbpedia, imdb,
yelp_stars. Held out for generalization: sst2, tweet_emotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset

from .schema import Question

MAX_STATE_CHARS = 1500


@dataclass(frozen=True)
class Task:
    name: str
    hf_id: str
    text_field: str
    label_field: str
    question_type: str
    phrasings: tuple[str, ...]
    criteria: dict | list | tuple
    eval_split: str
    hf_config: str | None = None
    train_split: str = "train"

    def question(self, phrasing: int = 0) -> Question:
        criteria = list(self.criteria) if isinstance(self.criteria, tuple) else self.criteria
        return Question(
            type=self.question_type,
            instructions=self.phrasings[phrasing % len(self.phrasings)],
            criteria=criteria,
        )


AG_NEWS = Task(
    name="ag_news",
    hf_id="fancyzhx/ag_news",
    text_field="text",
    label_field="label",
    question_type="choice",
    phrasings=(
        "What is the main topic of the news article in the state?",
        "Which category best describes this news story?",
        "What subject area does the article in the state belong to?",
    ),
    criteria={
        "world": "International news, politics, conflicts, and diplomacy.",
        "sports": "Athletes, teams, matches, scores, and competitions.",
        "business": "Companies, markets, economy, deals, and earnings.",
        "science_tech": "Science, technology, software, research, and gadgets.",
    },
    eval_split="test",
)

# criteria order must match fancyzhx/dbpedia_14 label indices (0-13)
DBPEDIA = Task(
    name="dbpedia",
    hf_id="fancyzhx/dbpedia_14",
    text_field="content",
    label_field="label",
    question_type="choice",
    phrasings=(
        "What kind of entity does the encyclopedia text in the state describe?",
        "Which category best fits the subject of the state's text?",
        "What type of thing is the main subject of the text in the state?",
    ),
    criteria={
        "company": "A business or commercial organization.",
        "educational_institution": "A school, college, or university.",
        "artist": "A musician, painter, writer, or other creative person.",
        "athlete": "A sports player or competitor.",
        "office_holder": "A politician or person holding an official position.",
        "transportation": "A vehicle type: ship, train, aircraft, automobile.",
        "building": "A man-made structure: stadium, church, hotel, bridge.",
        "natural_place": "A natural feature: mountain, river, lake.",
        "village": "A village or small settlement.",
        "animal": "A species of animal.",
        "plant": "A species of plant or fungus.",
        "album": "A music album or record.",
        "film": "A movie.",
        "written_work": "A book, novel, journal, or other publication.",
    },
    eval_split="test",
)

# criteria order must match cardiffnlp/tweet_eval emotion label indices:
# anger(0) joy(1) optimism(2) sadness(3)
TWEET_EMOTION = Task(
    name="tweet_emotion",
    hf_id="cardiffnlp/tweet_eval",
    hf_config="emotion",
    text_field="text",
    label_field="label",
    question_type="choice",
    phrasings=(
        "What emotion does the tweet in the state mainly express?",
        "Which feeling best describes the tone of the tweet?",
        "What is the dominant emotion of the message in the state?",
    ),
    criteria={
        "anger": "Irritation, outrage, hostility, or disgust.",
        "joy": "Happiness, excitement, amusement, or celebration.",
        "optimism": "Hope, confidence, or a positive outlook on the future.",
        "sadness": "Grief, disappointment, loneliness, or despair.",
    },
    eval_split="test",
)

SST2 = Task(
    name="sst2",
    hf_id="stanfordnlp/sst2",
    text_field="sentence",
    label_field="label",
    question_type="noul",
    phrasings=(
        "Does the movie review in the state express positive sentiment?",
        "Is the reviewer's overall opinion of the movie favorable?",
        "Does the state contain a positive assessment of the film?",
    ),
    criteria={
        "true": "The reviewer's overall opinion is favorable.",
        "false": "The reviewer's overall opinion is unfavorable.",
    },
    eval_split="validation",
)

IMDB = Task(
    name="imdb",
    hf_id="stanfordnlp/imdb",
    text_field="text",
    label_field="label",
    question_type="noul",
    phrasings=(
        "Does the movie review in the state express positive sentiment?",
        "Is the reviewer's overall opinion of the movie favorable?",
        "Would the reviewer in the state recommend this movie?",
    ),
    criteria={
        "true": "The reviewer's overall opinion is favorable.",
        "false": "The reviewer's overall opinion is unfavorable.",
    },
    eval_split="test",
)

# label 0..4 = 1..5 stars, an ordered spectrum -> score
YELP_STARS = Task(
    name="yelp_stars",
    hf_id="Yelp/yelp_review_full",
    text_field="text",
    label_field="label",
    question_type="score",
    phrasings=(
        "How positive is the customer review in the state?",
        "How satisfied does the customer in the state sound?",
        "Rate the overall sentiment of the review in the state.",
    ),
    criteria=(
        "Terrible experience; strong complaints and would not return.",
        "Poor; mostly negative with significant issues.",
        "Mixed; some good points and some clear problems.",
        "Good; mostly positive with minor complaints.",
        "Excellent; enthusiastic praise with no real complaints.",
    ),
    eval_split="test",
)

TASKS: dict[str, Task] = {
    t.name: t for t in (AG_NEWS, DBPEDIA, TWEET_EMOTION, SST2, IMDB, YELP_STARS)
}
TRAIN_TASKS = ("ag_news", "dbpedia", "imdb", "yelp_stars")
HELD_OUT_TASKS = ("sst2", "tweet_emotion")


def records(
    task: Task,
    split: str | None = None,
    n: int | None = None,
    phrasing: int = 0,
    seed: int = 42,
) -> Iterator[dict]:
    split = split or task.eval_split
    if task.hf_config:
        ds = load_dataset(task.hf_id, task.hf_config, split=split)
    else:
        ds = load_dataset(task.hf_id, split=split)
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    question = task.question(phrasing)
    for row in ds:
        yield {
            "state": row[task.text_field][:MAX_STATE_CHARS],
            "question": question,
            "label": row[task.label_field],
            "task": task.name,
        }
