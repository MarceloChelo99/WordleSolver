from numpy import log2
from pandas import DataFrame
from random import choice
import string

import nltk
nltk.download('words')
from nltk.corpus import words as corpus

from UI import PygameGrid


class WordsObj:
    def __init__(self, word_list):
        self.words_df = DataFrame([list(word) + [word] for word in word_list])
        self.words_df = self.words_df.rename(columns={self.words_df.columns.to_list()[-1]: 'Name'})

        print('hi')
    @property
    def words(self):
        return self.words_df['Name'].to_list()

    def return_random_item(self, random=False):
        return choice(self.words)

    def filter_list_by_letter(self, letter):
        return [word for word in self.words if letter in word]



class Entropy:
    def __init__(self, words: WordsObj):
        self.wordsObj = words
        self.n = len(self.wordsObj.words)

    def system_entropy(self):
        p = 1 / self.n
        return p * log2(1 / p) * self.n

    def word_entropy(self, word):
        total_entropy = 0
        for letter in word:
            sub_n = len(self.wordsObj.filter_list_by_letter(letter))
            probability = sub_n / self.n
            i = log2(1 / probability)
            total_entropy += probability * i
        return total_entropy



class Actions:
    def __init__(self, words_list):
        self.wordsObj = WordsObj(words_list)
        self.entropy = Entropy(self.wordsObj)

    def refresh_answer(self):
        return self.wordsObj.return_random_item()

    def make_guess(self, random=False):
        if random:
            return self.wordsObj.return_random_item()
        while len((guess := input("Guess: ").strip().lower())) != 5:
            print("Sorry, try again.")
        return guess


    def generate_possible_patterns(self, ):
        pass

    @staticmethod
    def evaluate_guess(guess, answer):
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

    def calculate_all_entropies(self):
        entropies = []
        for word in self.wordsObj.words:
            entropies.append((word, self.entropy.word_entropy(word)))
        return sorted(entropies, key=lambda x: x[1], reverse=True)

    def get_system_entropy(self):
        return self.entropy.system_entropy()

    def get_word_entropy(self, word):
        return self.entropy.word_entropy(word)

    def refresh_words_list(self, positions):
        for idx, letters in positions.items():
            self.wordsObj.words_df = self.wordsObj.words_df[self.wordsObj.words_df[idx].isin(letters)]
        return Actions(self.wordsObj.words)

    def filter_possible_patterns(self, positions):
        for idx, letters in positions.items():
            positions[idx] = [letter for letter in letters if letter in list(self.wordsObj.words_df[idx].unique())]
        return positions

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

        return positions

class Round:
    def __init__(self, actions, answer, positions):
        self.actions = actions
        self.answer = answer
        self.positions = positions

    def make_guess(self, random=False):
        return self.actions.make_guess(random)

    def get_entropy(self, guess):
        return self.actions.get_word_entropy(guess)

    def evaluate(self, guess):
        return self.actions.evaluate_guess(guess, self.answer)

    def update_possibilities(self, scores):
        self.positions = self.actions.update_positions(scores, self.positions)
        self.actions = self.actions.refresh_words_list(self.positions)
        self.positions = self.actions.filter_possible_patterns(self.positions)

    def run(self):
        guess = self.make_guess()
        #guess_entropy = self.get_entropy(guess)
        evaluation = self.evaluate(guess)
        self.update_possibilities(evaluation)
        return self.actions, self.positions, evaluation


class Game:
    def __init__(self, word_length):
        alphabet = string.ascii_lowercase
        self.round_number = 0
        self.won = False
        self.word_length = word_length
        self.positions = {position: list(alphabet) for position in list(range(self.word_length))}

        self.words_list = self.word_corpus()
        self.actions = Actions(self.words_list)
        self.answer = self.actions.refresh_answer()
        self.round = Round(self.actions, self.answer, self.positions)
        self.grid = PygameGrid()
        self.grid.draw()



    def word_corpus(self):
        return [word for word in corpus.words() if (len(word) == self.word_length) and (word[0].islower())]

    def new_round(self):
        self.round_number += 1
        print(self.answer)
        self.actions, self.positions, score = self.round.run()
        colors = []
        word = ""
        row_number = self.round_number - 1
        for idx, (letter, score) in score.items():
            colors.append(score)
            word += letter

        self.grid.set_row(row_number, word, colors)
        self.grid.draw()


    def check_win(self):
        return self.won

    def new_game(self):
        pass

    def refresh_answer(self):
        self.answer = self.actions.refresh_answer()
        return self




def word_corpus(word_length):
    return [word for word in corpus.words() if (len(word)==word_length) and (word[0].islower())]


def main():
    print("Welcome to Wordle Solver")
    words_list  = word_corpus(5)
    game = Game(word_length=5)
    for round_number in range(6):
        print(round_number)
        game.new_round()

    print("Game over")





if __name__ == "__main__":
    main()
