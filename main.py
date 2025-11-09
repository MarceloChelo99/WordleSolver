from numpy import log2
from pandas import DataFrame
from random import choice
import string
import threading
import queue

import pygame

import nltk
nltk.download('words')
from nltk.corpus import words as corpus

from UI import PygameGrid



class WordsObj:
    def __init__(self, word_list):
        self.words_df = DataFrame([list(word) + [word] for word in word_list])
        self.words_df = self.words_df.rename(columns={self.words_df.columns.to_list()[-1]: 'Name'})

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
    def calculate_all_entropies(words_obj):
        entropies = []
        for word in words_obj.words:
            entropies.append((word, Entropy.word_entropy(word, words_obj)))
        return sorted(entropies, key=lambda x: x[1], reverse=True)


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
        if len(normalized_guess) != word_length:
            self.state.invalid_guess = True
            return self.state

        if normalized_guess not in self.state.originalWordsObj.words:
            self.state.invalid_guess = True
            return self.state

        self.state.invalid_guess = False
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
