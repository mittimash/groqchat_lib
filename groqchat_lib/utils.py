import logging
import tiktoken
from pathlib import Path

def get_logger(name, level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger

def get_tokenizer():
    try:
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text))

def count_messages_tokens(tokenizer, messages):
    total = 0
    for msg in messages:
        total += 4 + count_tokens(tokenizer, msg["content"])
    return total + 2

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)