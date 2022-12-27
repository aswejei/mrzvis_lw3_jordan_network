import json
import numpy as np


def activation_function(val, alpha=1):
    return val if val >= 0 else alpha * (np.exp(val) - 1)


def activation_function_derivative(val, alpha=1):
    return 1 if val > 0 else alpha * np.exp(val)


def activate(matrix):
    for i, element in np.ndenumerate(matrix):
        matrix[i] = activation_function(element)
    return matrix


def json_to_ndarray(raw_json):
    arr = np.array(raw_json[:4])
    for i in range(1, 5):
        arr = np.append(arr, raw_json[i:i + 4])
    arr = arr.reshape((5, 4))
    return arr


def load_from_file(row):
    with open(f'rows/{choose(row)}.json', 'r') as f:
        data = json.load(f)
    return json_to_ndarray(data)


def choose(c):
    match c:
        case 1: return 'fibonacci'
        case 2: return 'periodical'
        case 3: return 'demonstration'
        case 4: return 'factorial'
    # if c == 1:
    #     return 'fibonacci'
    # elif c == 2:
    #     return 'periodical'
    # elif c == 3:
    #     return 'demonstration'
    # elif c == 4:
    #     return 'factorial'
