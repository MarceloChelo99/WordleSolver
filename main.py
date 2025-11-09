from numpy import log2
from pandas import DataFrame
from random import choice
import string
import threading
import queue
from itertools import product
from collections import Counter
import time

import pygame

import nltk
nltk.download('words')
from nltk.corpus import words as corpus
from wordfreq import word_frequency
from UI import PygameGrid



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




    # @staticmethod
    # def entropy(word, words_obj):
    #     patterns = list(product((0, 1, 2), repeat=5))
    #     expected_entropy = 0
    #     original_length = words_obj.words_df.shape[0]
    #     for code in patterns:
    #         df = words_obj.words_df.copy()
    #         for idx, (letter, value) in enumerate(zip(list(word), code)):
    #             if value == 2:
    #                 df = df[df[idx] == letter]
    #             if value == 1:
    #                 df = df[df[idx] != letter]
    #             if value == 0:
    #                 df = df[~df['Name'].str.contains(letter)]
    #         new_length = df.shape[0]
    #         p = new_length / original_length
    #         i = log2(1 / p)
    #         expected_entropy += p * i
    #     return expected_entropy

class Actions:
    @staticmethod
    def refresh_answer(words_obj):
        return words_obj.return_random_item()

    @staticmethod
    def evaluate_guess(guess, answer) -> dict:
        guess_array = list(guess)
        actual_array = list(answer)
        result = {position: (letter, 0) for position, letter in enumerate(guess_array)}
        for position, (guess, actual) in enumerate(zip(guess_array, actual_array)):
            if guess in actual_array:
                current_value = result[position][1]
                result[position] = (guess, current_value + 1)
            if guess == actual:
                current_value = result[position][1]
                result[position] = (guess, current_value + 1)
        return result

    @staticmethod
    def refresh_words_list(positions, words_obj):
        for idx, letters in positions.items():
            words_obj.words_df = words_obj.words_df[words_obj.words_df[idx].isin(letters)]

    @staticmethod
    def filter_possible_patterns(positions, words_obj):
        for idx, letters in positions.items():
            positions[idx] = [letter for letter in letters if letter in list(words_obj.words_df[idx].unique())]

    @staticmethod
    def update_positions(guess_scores, positions):
        for idx, (letter, score) in guess_scores.items():
            print(idx, letter, score)
            if score == 0:
                for position in positions:
                    current_values = positions[position]
                    positions[position] = [item for item in current_values if item != letter]

            if score == 1:
                current_values = positions[idx]
                positions[idx] = [item for item in current_values if item != letter]

            if score == 2:
                positions[idx] = [letter]



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
        self.entropyGenerator.entropy(normalized_guess, self.state.wordsObj)
        self.state.scores = self.evaluate(normalized_guess)
        self.state.round_number += 1
        self.update_possibilities()

        if all(score == 2 for _, score in self.state.scores.values()):
            self.state.won = True

        return self.state

    def get_entropy(self, guess):
        return self.entropyGenerator.word_entropy(guess, self.state.wordsObj)

    def evaluate(self, guess):
        return self.actionGenerator.evaluate_guess(guess, self.state.answer)

    def update_possibilities(self):
        self.actionGenerator.update_positions(self.state.scores, self.state.positions)
        self.actionGenerator.filter_possible_patterns(self.state.positions, self.state.wordsObj)


class State:
    def __init__(self, words_list, round_number=0, won=False, answer=None, positions=None, scores=None):
        alphabet = string.ascii_lowercase
        self.round_number = round_number
        self.won = won
        self.scores = scores
        self.invalid_guess = False

        self.originalWordsObj = WordsObj(words_list)
        self.wordsObj = WordsObj(words_list)

        self.answer = answer if answer is not None else self.originalWordsObj.return_random_item()

        self.positions = positions if positions is not None else {position: list(alphabet)
                                                                  for position in list(range(len(self.wordsObj.words[0])))}



class Game:
    def __init__(self, word_length):
        self.state  = State(self.word_corpus(word_length))
        self.round = Round(self.state, Actions, Entropy)


    @staticmethod
    def word_corpus(word_length):
        return [word for word in corpus.words() if (len(word) == word_length) and (word[0].islower())]

    def calculate_entropies(self):
        df = self.state.wordsObj.words_df
        words_list = self.state.wordsObj.words[:1000]
        start = time.perf_counter()
        df['Expected_Entropy'] = df['Name'].map(lambda x: Entropy.entropy(x, words_list))
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

