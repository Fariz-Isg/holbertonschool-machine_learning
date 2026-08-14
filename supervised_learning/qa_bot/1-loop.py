#!/usr/bin/env python3
"""Interactive question loop."""


if __name__ == "__main__":
    while True:
        question = input("Q: ")

        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        print("A:")
