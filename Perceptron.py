#import scikit-learn
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

#Code sourced from: https://www.python-course.eu/neural_networks.php
#https://www.youtube.com/watch?v=262XJe2I2D0

class Perceptron:
    def __init__(self, input_length, weights=None):
        if weights is None:
            self.weights = (np.random.random((input_length)) * 2 ) - 1
            '''
            self.weights = np.ones(input_length) + 0.5 #Simple weights of 1.5
            else:
            self.weights = weights
            '''
        self.learning_rate = 0.1

    @staticmethod
    def unit_step_function(x):
        if x > 0.5:
            return 1
        return 0

    def __call__(self, in_data):
        weighted_input = self.weights*in_data
        weighted_sum = weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)

    def adjust(self, target_results, calculated_result, in_data):
        error = target_results - calculated_result
        for i in range(len(in_data)):
            correction = error * in_data[i] * self.learning_rate
            self.weights[i] += correction

def above_line(point, line_func):
    x,y = point
    if y > line_func(x):
        return 1
    else:
        return 0

points = np.random.randint(1, 100 , (100,2))
p = Perceptron(2)
def lin1(x):
    return x+4
'''
#Simple AND function where x are 2 input matrixes
for x in [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]:
    y = p(np.array(x))
    print(x,y)
'''
for point in points:
    p.adjust(above_line(point, lin1), p(point), point)
evaluation = Counter()
for point in points:
    if p(point) == above_line(point, lin1):
        evaluation["correct"] += 1
    else:
        evaluation["wrong"] += 1
print(evaluation.most_common())
