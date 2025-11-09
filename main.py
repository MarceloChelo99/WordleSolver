from numpy import log2
from pandas import DataFrame
from random import choice
import string
import threading
import queue

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
    def make_guess(words_obj, random=False):
        if random:
            return words_obj.return_random_item()
        while len((guess := input("Guess: ").strip().lower())) != len(words_obj.words[0]):
            print("Sorry, try again.")
        return guess

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

    def make_guess(self, random=False):
        return self.actionGenerator.make_guess(self.state.wordsObj)

    def get_entropy(self, guess):
        return self.entropyGenerator.word_entropy(guess, self.state.wordsObj)

    def evaluate(self, guess):
        return self.actionGenerator.evaluate_guess(guess, self.state.answer)

    def update_possibilities(self):
        self.actionGenerator.update_positions(self.state.scores, self.state.positions)
        self.actionGenerator.filter_possible_patterns(self.state.positions, self.state.wordsObj)

    def run(self):
        while (guess := self.make_guess()) not in self.state.originalWordsObj.words:
            continue
        self.state.scores = self.evaluate(guess)
        self.update_possibilities()
        return self.state


class State:
    def __init__(self, words_list, round_number=0, won=False, answer=None, positions=None, scores=None):
        alphabet = string.ascii_lowercase
        self.round_number = round_number
        self.won = won
        self.scores = scores

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

    def new_round(self):
        self.state.round_number += 1
        print(self.state.answer)
        self.state = self.round.run()
        return self.state


    @staticmethod
    def new_game(word_length):
        return Game(word_length=word_length)


def word_corpus(word_length):
    return [word for word in corpus.words() if (len(word)==word_length) and (word[0].islower())]


def game(word_length):
    game_session = Game(word_length=word_length)
    for _ in range(6):
        state = game_session.new_round()
        yield state

def run_game(word_length, state_queue):
    for state in game(word_length):
        state_queue.put(state)
    state_queue.put(None)  # sentinel

def main():
    print("Welcome to Wordle Solver")
    state_queue = queue.Queue()
    thread = threading.Thread(target=run_game, args=(5, state_queue), daemon=True)
    thread.start()

    grid = PygameGrid()
    grid.draw()

    running = True
    while running:
        try:
            state = state_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            if state is None:
                running = False
            else:
                # use state.scores (note the plural) when pulling letter/score pairs
                colors, word = [], ""
                for idx, (letter, score) in state.scores.items():
                    colors.append(score)
                    word += letter
                grid.set_row(state.round_number - 1, word, colors)
                grid.draw()

    print("Game over")





if __name__ == "__main__":
    main()
