import os
import pickle


def LoadResults(filename):
    with open(
        os.path.join(os.path.dirname(__file__), "../results", filename), "rb"
    ) as file:
        return pickle.load(file)
