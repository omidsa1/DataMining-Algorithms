from csv import reader
from random import randrange, shuffle
from itertools import chain


def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):

        try:
            val = example[self.column]
        except:
            print(example, self.column)

        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def prune(gain, to_prune):
    if to_prune:
        return gain >= 0.1
    else:
        return True


def find_best_split(rows, to_prune=False):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):

        values = set([row[col] for row in rows])

        for val in values:

            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain and prune(gain, to_prune=to_prune):
                best_gain, best_question = gain, question
    return best_gain, best_question


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows, to_prune=False):

    gain, question = find_best_split(rows, to_prune=to_prune)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def train_test_evaluate(tree, test, tp, fp, fn):

    for row in test:

        counted_class = classify(row, tree)
        prediction = sorted(counted_class.items(), key=lambda x: x[1], reverse=True)[0][
            0
        ]

        if prediction == row[-1]:
            tp[str(prediction)] += 1
        else:
            fp[str(prediction)] += 1
            fn[str(row[-1])] += 1
    return tp, fp, fn


if __name__ == "__main__":

    with open("hayes_roth.data") as f:
        csv_reader = reader(f)
        dataset_ = list(csv_reader)

    lamb = lambda x: list(map(int, x))
    dataset = []
    for row in dataset_:
        dataset.append(lamb(row))

    shuffle(dataset)

    n_fold = 3
    header = ["f_A", "f_B", "f_C", "f_D", "f_E", "label"]

    split_dataset = cross_validation_split(dataset, n_fold)

    tp = {"1": 0, "2": 0, "3": 0}
    fp = {"1": 0, "2": 0, "3": 0}
    fn = {"1": 0, "2": 0, "3": 0}

    for index in range(n_fold):
        split_copy = split_dataset.copy()
        test = split_copy.pop(index)
        train = list(chain(*split_copy))
        tree = build_tree(train, to_prune=True)
        tp, fp, fn = train_test_evaluate(tree, test, tp, fp, fn)


    for i in range(1, 4):
        i = str(i)
        precision = tp[i] / (tp[i] + fp[i])
        recall = tp[i] / (tp[i] + fn[i])
        f_measure = 2 * (precision * recall) / (precision + recall)
        print(
            f"{i}'s precisson: {precision}, recall: {recall} , f_measure: {f_measure}"
        )
