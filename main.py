from numpy import log2, bincount, array, int16, ones
from pandas import DataFrame
from random import choice

import threading
import queue
from itertools import product
from collections import Counter
import time

import pygame
import multiprocessing as mp

import nltk
nltk.download('words')
from nltk.corpus import words as corpus
from wordfreq import word_frequency
from UI import PygameGrid

_entropy_pool_answers = None


def _entropy_pool_initializer(answers):
    global _entropy_pool_answers
    _entropy_pool_answers = answers


def _entropy_pool_worker(guess):
    return Entropy.entropy_for_guess(guess, _entropy_pool_answers)



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
    def entropy_for_guess(guess, answers):
        # produce encoded patterns for this guess vs every answer
        ids = [Entropy.encode_pattern(Entropy.feedback_pattern(guess, a)) for a in answers]
        counts = bincount(array(ids, dtype=int16), minlength=3 ** 5)
        probs = counts[counts > 0] / len(answers)
        return float((probs * log2(1.0 / probs)).sum())

    @staticmethod
    def compute_entropies_parallel(guesses, answers, processes=None):
        if not guesses:
            return []

        max_workers = processes or (mp.cpu_count() or 1)
        max_workers = min(max_workers, len(guesses))

        if max_workers <= 1:
            return Entropy.compute_entropies_bulk(guesses, answers)

        ctx = mp.get_context("spawn")

        with ctx.Pool(
                processes=max_workers,
                initializer=_entropy_pool_initializer,
                initargs=(answers,),
        ) as pool:
            chunk = max(1, len(guesses) // (max_workers * 4))
            return pool.map(_entropy_pool_worker, guesses, chunksize=chunk)

    @staticmethod
    def compute_entropies_bulk(guesses, answers):
        # pure Python loop; much faster than df['Name'].map(...)
        ent = [Entropy.entropy_for_guess(g, answers) for g in guesses]
        return ent

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

    def make_guess(self, guess: str):
        normalized_guess = guess.strip().lower()

        word_length = len(self.state.answer)
        print(self.state.answer)
        if len(normalized_guess) != word_length:
            self.state.invalid_guess = True
            return self.state

        if normalized_guess not in self.state.originalWordsObj.words:
            self.state.invalid_guess = True
            return self.state

        self.state.invalid_guess = False
        #self.entropyGenerator.entropy(normalized_guess, self.state.wordsObj.words)
        self.state.scores = self.evaluate(normalized_guess)
        self.state.round_number += 1
        self.update_possibilities()

        if all(score == 2 for _, score in self.state.scores.values()):
            self.state.won = True

        return self.state

    def get_entropy(self, guess):
        return self.entropyGenerator.word_entropy(guess, self.state.wordsObj)

    def evaluate(self, guess):
        pattern = Entropy.feedback_pattern(guess, self.state.answer)
        return {
            position: (letter, score)
            for position, (letter, score) in enumerate(zip(guess, pattern))
        }

    def update_possibilities(self):
        self.actionGenerator.update_candidates(self.state.scores, self.state.wordsObj)


class State:
    def __init__(self, words_list, round_number=0, won=False, answer=None, scores=None):
        self.round_number = round_number
        self.won = won
        self.scores = scores
        self.invalid_guess = False

        self.originalWordsObj = WordsObj(words_list)
        self.wordsObj = WordsObj(words_list)

        self.answer = answer if answer is not None else self.originalWordsObj.return_random_item()




class Game:
    def __init__(self, word_length):
        self.state  = State(self.word_corpus(word_length))
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


    def new_round(self, guess):
        self.state = self.round.make_guess(guess)
        return self.state


    @staticmethod
    def new_game(word_length):
        return Game(word_length=word_length)


def word_corpus(word_length):
    return [word for word in corpus.words() if (len(word)==word_length) and (word[0].islower())]


def run_game(word_length, guess_queue, state_queue):
    game_session = Game(word_length=word_length)

    while True:
        game_session.calculate_entropies()
        guess = guess_queue.get()
        if guess is None:
            break
        state = game_session.new_round(guess)
        state_queue.put(state)

        if not state.invalid_guess and (state.won or state.round_number >= 6):
            break

    state_queue.put(None)  # sentinel

def main():
    print("Welcome to Wordle Solver")
    word_length = 5
    state_queue = queue.Queue()
    guess_queue = queue.Queue()

    thread = threading.Thread(
        target=run_game,
        args=(word_length, guess_queue, state_queue),
        daemon=True,
    )
    thread.start()

    grid = PygameGrid()
    grid.draw()

    current_guess = ""
    current_row = 0
    preview_dirty = False
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                guess_queue.put(None)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if current_guess:
                        guess_queue.put(current_guess)
                elif event.key == pygame.K_BACKSPACE:
                    if current_guess:
                        current_guess = current_guess[:-1]
                        preview_dirty = True
                else:
                    if (
                        event.unicode
                        and event.unicode.isalpha()
                        and len(current_guess) < word_length
                    ):
                        current_guess += event.unicode.lower()
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
                preview_dirty = True

                if state.won or state.round_number >= grid.ROWS:
                    running = False
                    guess_queue.put(None)
                    break
        except queue.Empty:
            pass

        if preview_dirty:
            if running and current_row < grid.ROWS:
                preview_colors = [0] * len(current_guess)
                grid.set_row(current_row, current_guess, preview_colors)
            grid.draw()
            preview_dirty = False

        clock.tick(60)

    grid.close()
    thread.join(timeout=0.5)





if __name__ == "__main__":
    main()

