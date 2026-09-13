import pickle

if __name__ == "__main__":
    with open("artifacts_tiny/vocab.pkl", "rb") as file:
        data: dict[int, bytes] = pickle.load(file)
        longest_len = 0
        longest_item = 0

        for value in data.values():
            if len(value) > longest_len:
                longest_len = len(value)
                longest_item = value
        print(longest_len, longest_item)
