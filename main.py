import argparse
import json
import queue
import threading
import time
from collections import Counter
from datetime import date, datetime, timezone
from random import choice
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import multiprocessing as mp
import numpy as np
import nltk
import pygame
from numpy import log2, ones
from pandas import DataFrame


def _ensure_words_corpus():
    """Make sure the NLTK words corpus is available without re-downloading."""

    try:
        nltk.data.find("corpora/words")
    except LookupError:
        nltk.download("words", quiet=True)


_ensure_words_corpus()
from nltk.corpus import words as corpus
from wordfreq import word_frequency
from UI import PygameGrid

_entropy_pool_answers_matrix = None


def _entropy_pool_initializer(answers_matrix):
    global _entropy_pool_answers_matrix
    _entropy_pool_answers_matrix = answers_matrix


def _entropy_pool_worker(guess):
    return Entropy.entropy_for_guess(guess, _entropy_pool_answers_matrix)


_BASE3_WEIGHTS_CACHE = {}


def _base3_weights(length):
    weights = _BASE3_WEIGHTS_CACHE.get(length)
    if weights is None:
        weights = np.power(3, np.arange(length - 1, -1, -1), dtype=np.int32)
        _BASE3_WEIGHTS_CACHE[length] = weights
    return weights


def _words_to_matrix(words):
    if not words:
        return np.empty((0, 0), dtype=np.uint8)

    word_length = len(words[0])
    matrix = np.empty((len(words), word_length), dtype=np.uint8)

    for idx, word in enumerate(words):
        if len(word) != word_length:
            raise ValueError("All words must have the same length")
        matrix[idx] = np.frombuffer(word.encode("ascii"), dtype=np.uint8)

    return matrix


def download_wordle_solution(target_date=None, *, timeout=10):
    """Download the official Wordle solution for the given date.

    Parameters
    ----------
    target_date : datetime.date | datetime.datetime | None
        The calendar date for which to fetch the Wordle solution. Defaults to
        the current day in UTC if ``None`` is provided.
    timeout : int | float
        Maximum number of seconds to wait for the remote response.

    Returns
    -------
    str
        The five-letter Wordle solution published for ``target_date``.

    Raises
    ------
    RuntimeError
        If the solution cannot be downloaded or parsed.
    """

    if target_date is None:
        target_date = datetime.now(timezone.utc).date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    elif not isinstance(target_date, date):
        raise TypeError("target_date must be a date, datetime, or None")

    url = f"https://www.nytimes.com/svc/wordle/v2/{target_date.isoformat()}.json"

    try:
        with urlopen(url, timeout=timeout) as response:
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(
                    f"Unexpected HTTP status {response.status} while downloading Wordle solution."
                )
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("Failed to download Wordle solution") from exc

    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Unable to parse Wordle solution response") from exc

    solution = data.get("solution")
    if not solution:
        raise RuntimeError("Wordle response did not include a solution value")

    return solution


def download_today_wordle_word(*, timeout=10):
    """Return today's Wordle solution from the official NYT endpoint."""

    return download_wordle_solution(timeout=timeout)


class WordsObj:
    def __init__(self, word_list):
        self.words_df = DataFrame([list(word) + [word] for word in word_list])
        self.words_df = self.words_df.rename(columns={self.words_df.columns.to_list()[-1]: 'Name'})
        self.words_df['Frequency'] = self.words_df['Name'].map(lambda x: word_frequency(x, 'en', wordlist='best'))
        print('')

    @property
    def words(self):
        return self.words_df['Name'].to_list()

    def return_random_item(self, random=False):
        return choice(self.words)

    def filter_list_by_letter(self, letter):
        return [word for word in self.words if letter in word]



class Entropy:
    @staticmethod
    def system_entropy(words_obj):
        p = 1 / len(words_obj.words)
        return p * log2(1 / p) * len(words_obj.words)

    @staticmethod
    def word_entropy(word, words_obj):
        total_entropy = 0
        for letter in word:
            sub_n = len(words_obj.filter_list_by_letter(letter))
            probability = sub_n / len(words_obj.words)
            i = log2(1 / probability)
            total_entropy += probability * i
        return total_entropy

    @staticmethod
    def calculate_all_entropies(word_list):
        entropies = []
        for word in word_list:
            entropies.append((word, Entropy.word_entropy(word, word_list)))
        return sorted(entropies, key=lambda x: x[1], reverse=True)

    @staticmethod
    def encode_pattern(p):
        v = 0
        for x in p:
            v = v * 3 + x
        return v

    @staticmethod
    def feedback_pattern(guess: str, answer: str) -> tuple[int, ...]:
        # returns a 5-tuple of 0/1/2
        res = [0] * len(guess)
        ans = list(answer)

        # pass 1: greens
        for i, (g, a) in enumerate(zip(guess, ans)):
            if g == a:
                res[i] = 2
                ans[i] = None  # consume

        # pass 2: yellows with multiplicity control
        leftover = Counter(ch for ch in ans if ch is not None)
        for i, g in enumerate(guess):
            if res[i] == 0 and leftover.get(g, 0) > 0:
                res[i] = 1
                leftover[g] -= 1
        return tuple(res)

    @staticmethod
    def entropy_for_guess(guess, answers_matrix):
        if not isinstance(answers_matrix, np.ndarray):
            answers_matrix = _words_to_matrix(answers_matrix)

        if answers_matrix.size == 0:
            return 0.0

        guess_vector = np.frombuffer(guess.encode("ascii"), dtype=np.uint8)
        word_length = answers_matrix.shape[1]

        if guess_vector.shape[0] != word_length:
            raise ValueError("Guess length does not match answer length")

        greens = answers_matrix == guess_vector
        patterns = greens.astype(np.uint8) * 2

        if not greens.all():
            leftover = np.zeros((answers_matrix.shape[0], 256), dtype=np.int8)

            for pos in range(word_length):
                mask = ~greens[:, pos]
                if not mask.any():
                    continue

                rows = np.nonzero(mask)[0]
                cols = answers_matrix[rows, pos]
                np.add.at(leftover, (rows, cols), 1)

            for pos in range(word_length):
                mask = ~greens[:, pos]
                if not mask.any():
                    continue

                rows = np.nonzero(mask)[0]
                if rows.size == 0:
                    continue

                letter_code = guess_vector[pos]
                available_rows = rows[leftover[rows, letter_code] > 0]
                if available_rows.size == 0:
                    continue

                patterns[available_rows, pos] = 1
                leftover[available_rows, letter_code] -= 1

        weights = _base3_weights(word_length)
        encoded = patterns.dot(weights).astype(np.int32)
        pattern_space = 3 ** word_length
        counts = np.bincount(encoded, minlength=pattern_space)
        probs = counts[counts > 0] / answers_matrix.shape[0]
        return float((probs * log2(1.0 / probs)).sum())

    @staticmethod
    def compute_entropies_parallel(guesses, answers, processes=None):
        if not guesses:
            return []

        answers_matrix = _words_to_matrix(answers)

        max_workers = processes or (mp.cpu_count() or 1)
        max_workers = min(max_workers, len(guesses))

        if max_workers <= 1:
            return Entropy.compute_entropies_bulk(guesses, answers, answers_matrix)

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        with ctx.Pool(
                processes=max_workers,
                initializer=_entropy_pool_initializer,
                initargs=(answers_matrix,),
        ) as pool:
            chunk = max(1, len(guesses) // (max_workers * 4))
            return pool.map(_entropy_pool_worker, guesses, chunksize=chunk)

    @staticmethod
    def compute_entropies_bulk(guesses, answers, answers_matrix=None):
        matrix = answers_matrix if answers_matrix is not None else _words_to_matrix(answers)
        return [Entropy.entropy_for_guess(g, matrix) for g in guesses]

    @staticmethod
    def entropy(word: str, word_list) -> float:
        answers = word_list  # current candidate answers
        N = len(answers)
        # Count how many answers yield each feedback pattern
        counts = Counter(Entropy.feedback_pattern(word, a) for a in answers)

        # Expected info: sum p * log2(1/p) over patterns with nonzero mass
        H = 0.0
        for c in counts.values():
            p = c / N
            H += p * log2(1.0 / p)
        return H


class Actions:
    @staticmethod
    def refresh_answer(words_obj):
        return words_obj.return_random_item()


    @staticmethod
    def update_candidates(guess_scores, words_obj):
        df = words_obj.words_df
        if df.empty:
            return

        letter_columns = [col for col in df.columns if isinstance(col, int)]
        if not letter_columns:
            return

        letter_columns.sort()
        letters_matrix = df[letter_columns].to_numpy(copy=False)
        mask = ones(len(df), dtype=bool)

        column_positions = {col: pos for pos, col in enumerate(letter_columns)}
        unique_letters = {letter for letter, _ in guess_scores.values()}
        letter_matches = {
            letter: letters_matrix == letter for letter in unique_letters
        }

        letter_feedback = {}

        for idx, (letter, score) in guess_scores.items():
            info = letter_feedback.setdefault(letter, {"positives": 0, "zeros": 0})
            col_pos = column_positions.get(idx)
            if col_pos is None:
                continue

            matches = letter_matches[letter][:, col_pos]
            if score == 2:
                mask &= matches
                info["positives"] += 1
            elif score == 1:
                mask &= ~matches
                info["positives"] += 1
            else:
                mask &= ~matches
                info["zeros"] += 1

            if not mask.any():
                words_obj.words_df = df.iloc[0:0].copy()
                return

        for letter, info in letter_feedback.items():
            positives = info["positives"]
            zero_count = info["zeros"]
            occurrences = letter_matches[letter].sum(axis=1)

            if positives == 0:
                condition = occurrences == 0
            elif zero_count:
                condition = occurrences == positives
            else:
                condition = occurrences >= positives

            mask &= condition

            if not mask.any():
                words_obj.words_df = df.iloc[0:0].copy()
                return

        words_obj.words_df = df.loc[mask].copy()




class Round:
    def __init__(self, state, action_engine, entropy_engine):
        self.state = state
        self.actionGenerator = action_engine
        self.entropyGenerator = entropy_engine

    def make_guess(self, guess: str, feedback=None):
        normalized_guess = guess.strip().lower()

        word_length = self.state.word_length
        if word_length <= 0:
            self.state.invalid_guess = True
            return self.state

        if len(normalized_guess) != word_length:
            self.state.invalid_guess = True
            return self.state

        if normalized_guess not in self.state.originalWordsObj.words:
            self.state.invalid_guess = True
            return self.state

        if feedback is None:
            if self.state.manual_feedback:
                self.state.invalid_guess = True
                return self.state
            pattern = Entropy.feedback_pattern(normalized_guess, self.state.answer)
        else:
            try:
                pattern = tuple(int(value) for value in feedback)
            except (TypeError, ValueError):
                self.state.invalid_guess = True
                return self.state

            if len(pattern) != word_length or any(value not in (0, 1, 2) for value in pattern):
                self.state.invalid_guess = True
                return self.state

        self.state.invalid_guess = False
        self.state.scores = self.evaluate(normalized_guess, pattern)
        self.state.round_number += 1
        self.update_possibilities()

        if all(score == 2 for _, score in self.state.scores.values()):
            self.state.won = True

        return self.state

    def get_entropy(self, guess):
        return self.entropyGenerator.word_entropy(guess, self.state.wordsObj)

    def evaluate(self, guess, pattern=None):
        if pattern is None:
            pattern = Entropy.feedback_pattern(guess, self.state.answer)
        return {
            position: (letter, score)
            for position, (letter, score) in enumerate(zip(guess, pattern))
        }

    def update_possibilities(self):
        self.actionGenerator.update_candidates(self.state.scores, self.state.wordsObj)


class State:
    def __init__(
        self,
        words_list,
        round_number=0,
        won=False,
        answer=None,
        scores=None,
        manual_feedback=False,
        answer_source="dictionary",
    ):
        self.round_number = round_number
        self.won = won
        self.scores = scores or {}
        self.invalid_guess = False
        self.manual_feedback = manual_feedback
        self.answer_source = answer_source

        self.originalWordsObj = WordsObj(words_list)
        self.wordsObj = WordsObj(words_list)

        self.word_length = len(words_list[0]) if words_list else 0

        if answer is not None:
            self.answer = answer
        elif manual_feedback:
            self.answer = None
        else:
            self.answer = self._select_answer(answer_source)

    def _select_answer(self, answer_source):
        if answer_source == "today":
            try:
                today_answer = download_today_wordle_word()
            except Exception as exc:  # noqa: BLE001 - surface to stdout and fallback
                print(
                    "Failed to download today's Wordle solution; "
                    f"falling back to a dictionary word. ({exc})"
                )
            else:
                normalized = today_answer.strip().lower()
                if len(normalized) != self.word_length:
                    print(
                        "Today's Wordle solution length does not match the configured word length; "
                        "falling back to a dictionary word."
                    )
                else:
                    corpus_words = self.originalWordsObj.words
                    if normalized not in corpus_words:
                        print(
                            "Today's Wordle solution was not found in the local corpus; "
                            "falling back to a dictionary word."
                        )
                    else:
                        print("Using today's official Wordle solution as the answer.")
                        return normalized

        return self.originalWordsObj.return_random_item()




class Game:
    def __init__(self, word_length, *, manual_feedback=False, answer_source="dictionary"):
        self.state = State(
            self.word_corpus(word_length),
            manual_feedback=manual_feedback,
            answer_source=answer_source,
        )
        self.round = Round(self.state, Actions, Entropy)


    @staticmethod
    def word_corpus(word_length):
        return [word for word in corpus.words() if (len(word) == word_length) and (word[0].islower())]

    def calculate_entropies(self):
        df = self.state.wordsObj.words_df.iloc[:2000]
        start = time.perf_counter()
        answers = df['Name'].to_list()
        guesses = df['Name'].to_list() # or a shortlist
        ent = Entropy.compute_entropies_parallel(guesses, answers)
        df.loc[df['Name'].isin(guesses), 'Expected_Entropy'] = ent
        #df['Expected_Entropy'] = df['Name'].map(lambda x: Entropy.entropy(x, words_list))
        df = df.sort_values(by='Expected_Entropy', ascending=False)
        print(df.set_index('Name').head(50))
        elapsed = time.perf_counter() - start
        print(elapsed, "s")


    def new_round(self, guess, feedback=None):
        self.state = self.round.make_guess(guess, feedback=feedback)
        return self.state


    @staticmethod
    def new_game(word_length, manual_feedback=False, answer_source="dictionary"):
        return Game(
            word_length=word_length,
            manual_feedback=manual_feedback,
            answer_source=answer_source,
        )


def word_corpus(word_length):
    return [word for word in corpus.words() if (len(word)==word_length) and (word[0].islower())]


def run_game(
    word_length,
    guess_queue,
    state_queue,
    manual_feedback=False,
    answer_source="dictionary",
):
    game_session = Game(
        word_length=word_length,
        manual_feedback=manual_feedback,
        answer_source=answer_source,
    )

    while True:
        game_session.calculate_entropies()
        payload = guess_queue.get()
        if payload is None:
            break
        if isinstance(payload, dict):
            guess = payload.get("guess", "")
            feedback = payload.get("feedback")
        else:
            guess = payload
            feedback = None

        state = game_session.new_round(guess, feedback)
        state_queue.put(state)

        if not state.invalid_guess and (state.won or state.round_number >= 6):
            break

    state_queue.put(None)  # sentinel


def parse_args():
    parser = argparse.ArgumentParser(description="Wordle Solver")
    parser.add_argument(
        "--word-length",
        type=int,
        default=5,
        help="Length of the target words (default: 5).",
    )
    parser.add_argument(
        "--manual-feedback",
        action="store_true",
        help="Manually enter Wordle feedback colors in the UI instead of using a hidden answer.",
    )
    parser.add_argument(
        "--answer-source",
        choices=("dictionary", "today"),
        default="dictionary",
        help=(
            "Choose the hidden answer source when not using manual feedback. "
            "'dictionary' selects a random word from the local corpus, while 'today' "
            "downloads today's official Wordle solution."
        ),
    )
    return parser.parse_args()


def main():
    print("Welcome to Wordle Solver")
    args = parse_args()
    word_length = max(1, args.word_length)
    manual_feedback = args.manual_feedback
    answer_source = args.answer_source
    if manual_feedback:
        if answer_source != "dictionary":
            print(
                "Manual feedback mode enabled; ignoring --answer-source and "
                "using a dictionary word."
            )
        answer_source = "dictionary"
    state_queue = queue.Queue()
    guess_queue = queue.Queue()

    thread = threading.Thread(
        target=run_game,
        args=(word_length, guess_queue, state_queue, manual_feedback, answer_source),
        daemon=True,
    )
    thread.start()

    grid = PygameGrid()
    grid.draw()

    current_guess = ""
    current_row = 0
    awaiting_result = False
    editing_feedback = False
    active_feedback_row = None
    pending_guess = ""
    preview_dirty = False
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                guess_queue.put(None)
            elif event.type == pygame.KEYDOWN:
                if awaiting_result and event.key != pygame.K_ESCAPE:
                    continue

                if event.key == pygame.K_RETURN:
                    if manual_feedback and editing_feedback and active_feedback_row is not None:
                        feedback = grid.get_row_scores(active_feedback_row, word_length)
                        guess_queue.put({"guess": pending_guess, "feedback": feedback})
                        awaiting_result = True
                        editing_feedback = False
                        active_feedback_row = None
                        pending_guess = ""
                    elif len(current_guess) == word_length:
                        if manual_feedback:
                            grid.set_row(current_row, current_guess, [0] * word_length)
                            editing_feedback = True
                            active_feedback_row = current_row
                            pending_guess = current_guess
                            preview_dirty = True
                        else:
                            guess_queue.put({"guess": current_guess, "feedback": None})
                            awaiting_result = True
                    else:
                        continue
                elif event.key == pygame.K_ESCAPE:
                    if manual_feedback and editing_feedback and active_feedback_row is not None:
                        grid.set_row(active_feedback_row, "", [])
                        editing_feedback = False
                        active_feedback_row = None
                        pending_guess = ""
                        current_guess = ""
                        preview_dirty = True
                    elif current_guess:
                        current_guess = ""
                        preview_dirty = True
                elif event.key == pygame.K_BACKSPACE:
                    if manual_feedback and editing_feedback:
                        continue
                    if current_guess:
                        current_guess = current_guess[:-1]
                        preview_dirty = True
                else:
                    if manual_feedback and editing_feedback:
                        continue
                    if (
                        event.unicode
                        and event.unicode.isalpha()
                        and len(current_guess) < word_length
                    ):
                        current_guess += event.unicode.lower()
                        preview_dirty = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (
                    manual_feedback
                    and editing_feedback
                    and event.button == 1
                    and active_feedback_row is not None
                ):
                    cell = grid.cell_from_point(event.pos)
                    if cell and cell[0] == active_feedback_row:
                        grid.cycle_cell(*cell)
                        preview_dirty = True

        try:
            while True:
                state = state_queue.get_nowait()
                if state is None:
                    running = False
                    break

                if getattr(state, "invalid_guess", False):
                    print("Invalid guess, try again.")
                    current_guess = ""
                    current_row = state.round_number
                    awaiting_result = False
                    editing_feedback = False
                    active_feedback_row = None
                    pending_guess = ""
                    preview_dirty = True
                    if current_row < grid.ROWS:
                        grid.set_row(current_row, "", [])
                    continue

                colors, word = [], ""
                for idx, (letter, score) in sorted(state.scores.items()):
                    colors.append(score)
                    word += letter

                grid.set_row(state.round_number - 1, word, colors)
                current_row = state.round_number
                current_guess = ""
                awaiting_result = False
                editing_feedback = False
                active_feedback_row = None
                pending_guess = ""
                preview_dirty = True

                if state.won or state.round_number >= grid.ROWS:
                    running = False
                    guess_queue.put(None)
                    break
        except queue.Empty:
            pass

        if preview_dirty:
            if (
                running
                and current_row < grid.ROWS
                and not (manual_feedback and editing_feedback)
            ):
                preview_colors = [0] * len(current_guess)
                grid.set_row(current_row, current_guess, preview_colors)
            grid.draw()
            preview_dirty = False

        clock.tick(60)

    grid.close()
    thread.join(timeout=0.5)





if __name__ == "__main__":
    main()

