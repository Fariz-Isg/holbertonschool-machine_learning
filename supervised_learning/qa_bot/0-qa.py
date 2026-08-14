#!/usr/bin/env python3
"""0-qa.py
A module that contains a function that answers questions"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """A function that does the trick"""
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors="tf"
    )
    input_ids = inputs["input_ids"]
    input_mask = inputs["attention_mask"]
    segment_ids = inputs["token_type_ids"]

    outputs = model([
        input_ids,
        input_mask,
        segment_ids
    ])

    start_logits, end_logits = outputs

    start = tf.argmax(start_logits, axis=1).numpy()[0]
    end = tf.argmax(end_logits, axis=1).numpy()[0]

    if start > end:
        return None

    tokens = input_ids[0][start:end + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()

    if answer == "":
        return None

    return answer
