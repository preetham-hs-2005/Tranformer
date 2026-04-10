import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# Shared model config
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 128
MAX_LEN = 32


# Sentiment task config
SENTIMENT_EPOCHS = 40
SENTIMENT_BATCH_SIZE = 4
SENTIMENT_LR = 1e-3
SENTIMENT_LABELS = {"negative": 0, "positive": 1}
SENTIMENT_ID_TO_LABEL = {0: "negative", 1: "positive"}

SENTIMENT_DATA = [
    ("i love this movie", "positive"),
    ("this film is amazing", "positive"),
    ("what a fantastic story", "positive"),
    ("i enjoyed every minute", "positive"),
    ("the acting was great", "positive"),
    ("i hate this movie", "negative"),
    ("this film is terrible", "negative"),
    ("what a boring story", "negative"),
    ("i disliked every minute", "negative"),
    ("the acting was awful", "negative"),
]


# Summarization task config
SUMMARIZATION_EPOCHS = 80
SUMMARIZATION_BATCH_SIZE = 2
SUMMARIZATION_LR = 1e-3
MAX_SRC_LEN = 24
MAX_TGT_LEN = 12

SUMMARIZATION_DATA = [
    (
        "the weather is sunny and warm today with clear skies",
        "sunny warm weather",
    ),
    (
        "the team won the match after a dramatic final goal",
        "team won match",
    ),
    (
        "a new cafe opened downtown and serves fresh coffee",
        "new downtown cafe",
    ),
    (
        "the company released a new phone with better battery life",
        "company released new phone",
    ),
    (
        "students prepared hard for exams and improved their scores",
        "students improved scores",
    ),
    (
        "heavy rain caused traffic delays across the city this morning",
        "rain caused city delays",
    ),
]
