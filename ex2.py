import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import genfromtxt
import math
from easygui import multenterbox

nLines = 25
nColumns = nLines
SIZE = nLines*nColumns
nPopulation = 500
nIterations = 500
constraints = 8
percent_mutation = 0.05
percent_children = 0.5
percent_accelerate_mutation = 0.35
Category = ["Original", "Darwin", "Lamarck"]
mode = 1
MODE = Category[mode]
NAME_FILE = "fish.txt"
run_time = 0

# input user box(1):
text = "The board size is " + str(nLines) +"X" + str(nColumns) + " squares.\n" \
        "IMPORTANT!\n" \
        "1. Put the file text in the folder where the script is located.\n" \
        "2. The text file should contain SPACES between the constraints - as defined in the exercise.\n"\
        "3. Input the name file with extension .txt for example: fish.txt\n" \
        "4. The running may take several minutes! (5-10 minutes) - please be patient.\n"\
        "5. select MODE for the algorithm: for regular insert 0, for Darwin insert 1, for Lamarck insert 2.\n"
title = ""
# parameters explanation and default values
initial_values_box = ["Text file name", "Mode algorithm"]
initial_values = ["fish.txt", 0]
input = multenterbox(text, title, initial_values_box, initial_values)
# convert to str to int
input[1] = int(input[1])
NAME_FILE, mode = input

def read_csv_file():
    my_data = genfromtxt(NAME_FILE, delimiter=' ')
    # save first half lines in rows
    rows = my_data[0:nLines]
    # save second half lines in columns
    columns = my_data[nLines:nLines*2]
    return my_data, rows, columns


class Solution:
    def __init__(self, _rows, _columns):
        self.constraints_rows = _rows
        self.constraints_columns = _columns
        self.grid_solution = np.random.randint(2, size=(nLines, nColumns))
        self.fitness_score = fitness_function(self.grid_solution, self.constraints_rows,
                                              self.constraints_columns)
        self.before_opt_grid_solution = self.grid_solution


def random_solutions(_rows, _columns):
    solutions = []
    for num in range(nPopulation):
        solutions.append(Solution(_rows, _columns))
    return solutions


def fitness_function(grid_solution, _rows, _columns):
    # for mode 2
    count = 0
    counter_until_50_opt = 0
    # calculate the difference between constraints to current status in the rows
    for line_index in range(nLines):
        rules_num = constraints
        column_index = 0
        rule_index = 0
        flag = 0
        begin_sequence = 0
        remember_index = 0
        while column_index < nColumns or rule_index < rules_num:
            count_sequence = 0
            current_rule = _rows[line_index][rule_index] if rule_index < rules_num else 0
            while column_index < nColumns and grid_solution[line_index][column_index] == 0:
                column_index += 1
            while column_index < nColumns and grid_solution[line_index][column_index] == 1:
                if flag == 0:
                    begin_sequence = column_index
                    flag = 1
                count_sequence += 1
                column_index += 1
                remember_index = column_index
            mode_counter = abs(count_sequence - current_rule)
            # Lamarck -optimization
            if MODE == Category[1] or MODE == Category[2] and counter_until_50_opt < 50:
                if mode_counter == 1:
                    # positive means we need to remove one block in order to get the sequence
                    if count_sequence - current_rule > 0 and remember_index < nColumns - 1:
                        grid_solution[line_index][remember_index + 1] = 0
                        mode_counter = 0
                    # if we at the end of row we need to remove the block from the other side (start)
                    elif count_sequence - current_rule > 0 and remember_index == nColumns - 1:
                        grid_solution[line_index][begin_sequence - 1] = 0
                        mode_counter = 0
                    # else negative we need to add one block in order to get the sequence
                    elif remember_index < nColumns - 1:
                        grid_solution[line_index][remember_index + 1] = 1
                        mode_counter = 0
                    elif remember_index == nColumns - 1:
                        grid_solution[line_index][begin_sequence - 1] = 1
                        mode_counter = 0
                    counter_until_50_opt += 1
            count -= mode_counter
            rule_index += 1
    # calculate the difference between constraints to current status in the columns
    for column_index in range(nColumns):
        rules_num = constraints
        line_index = 0
        rule_index = 0
        flag = 0
        begin_sequence = 0
        remember_index = 0
        while line_index < nLines or rule_index < rules_num:
            count_sequence = 0
            current_rule = _columns[column_index][rule_index] if rule_index < rules_num else 0
            while line_index < nLines and grid_solution[line_index][column_index] == 0:
                line_index += 1
            while line_index < nLines and grid_solution[line_index][column_index] == 1:
                if flag == 0:
                    begin_sequence = line_index
                    flag = 1
                count_sequence += 1
                line_index += 1
                remember_index = line_index
            mode_counter = abs(count_sequence - current_rule)
            # Lamarck -optimization
            if MODE == Category[1] or MODE == Category[2] and counter_until_50_opt < 50:
                if mode_counter == 1:
                    # positive means we need to remove one block in order to get the sequence
                    if count_sequence - current_rule > 0 and remember_index < nColumns-1:
                        grid_solution[remember_index+1][column_index] = 0
                        mode_counter = 0
                    # if we at the end of row we need to remove the block from the other side (start)
                    elif count_sequence - current_rule > 0 and remember_index == nColumns-1:
                        grid_solution[begin_sequence-1][column_index] = 0
                        mode_counter = 0
                    # else negative we need to add one block in order to get the sequence
                    elif remember_index < nColumns-1:
                        grid_solution[remember_index+1][column_index] = 1
                        mode_counter = 0
                    elif remember_index == nColumns -1 and begin_sequence!=0:
                        grid_solution[begin_sequence-1][column_index] = 1
                        mode_counter = 0
                    counter_until_50_opt += 1
            count -= mode_counter
            rule_index += 1
    return count


def crossover(list_solutions, _rows, _columns):
    number_children = math.ceil(percent_children*nPopulation) # num of children
    list_children_solution = []
    # sort by fitness score - last one is that with the bigger fitness score
    list_solutions.sort(key=lambda solution: (solution.fitness_score, random.random()))
    # calculate probability to select - higher score higher probability
    probability = [index / (nPopulation*(nPopulation+1)) for index in range(1, nPopulation + 1)]
    for i in range(number_children):
        # select index parents in random way which solutions with higher fitness score gets higher chance to be parents
        parent1, parent2 = random.choices(list_solutions, weights=probability, k=2)
        # create child solution - child.grid_solution all zeros at the beginning
        child_solution = Solution(_rows, _columns)
        for line in range(nLines):
            for column in range(nColumns):
                if random.random() <= 0.5:
                    child_solution.grid_solution[line][column] = parent1.grid_solution[line][column]
                else:
                    child_solution.grid_solution[line][column] = parent2.grid_solution[line][column]
        # update fitness score to child after give him a "genetic" grid
        child_solution.fitness_score = fitness_function(child_solution.grid_solution,
                                                        child_solution.constraints_rows,
                                                        child_solution.constraints_columns)
        list_children_solution.append(child_solution)
    return list_children_solution


def mutation_children(list_children_solution):
    num_children_with_mutation_sol = math.ceil(percent_mutation*len(list_children_solution))
    # select children
    select_children = random.choices(list_children_solution, k=num_children_with_mutation_sol)
    for child in select_children:
        # select point to change
        random_line = random.choice(range(nLines))
        random_column = random.choice(range(nColumns))
        if child.grid_solution[random_line][random_column] == 1:
            child.grid_solution[random_line][random_column] = 0
        else:
            child.grid_solution[random_line][random_column] = 1
    # add children after mutation to children list
    list_children_solution += select_children
    return list_children_solution


def next_generation(list_solutions, list_children_solution):
    list_solutions += list_children_solution
    top_90_percent_best_pop = math.ceil(nPopulation*0.9)
    # regular and lamarck
    if MODE != Category[1]:
        # select best solution - higher fitness score - THIS IS AFTER OPTIMIZATION
        list_solutions.sort(key=lambda solution: (solution.fitness_score, random.random()), reverse=True)
    # darwin - sort before optimization score
    else:
        list_solutions.sort(key=lambda solution: (fitness_function(solution.before_opt_grid_solution,
                                                                   solution.constraints_rows,
                                                                   solution.constraints_columns)), reverse=True)
        for solution in list_solutions:
            solution.before_opt_grid_solution = solution.grid_solution
    # select 90% top solution
    new_list_solutions = list_solutions[0:top_90_percent_best_pop]
    other_10_percent = nPopulation - top_90_percent_best_pop
    # other 10% select by random
    for i in range(other_10_percent):
        new_list_solutions.append(random.choice(list_solutions))
    # sort again
    if MODE != Category[2]:
        list_solutions.sort(key=lambda solution: (solution.fitness_score, random.random()), reverse=True)
    else:
        list_solutions.sort(key=lambda solution: (fitness_function(solution.before_opt_grid_solution,
                                                                   solution.constraints_rows,
                                                                   solution.constraints_columns)), reverse=True)
    sol_with_max_fitness_score = list_solutions[0].fitness_score
    sol_with_min_fitness_score = list_solutions[-1].fitness_score
    return new_list_solutions, sol_with_max_fitness_score, sol_with_min_fitness_score


def accelerate_mutations_rate(list_solutions):
    print("accelerate_mutations_rate")
    # select random indexes
    random_indexes = random.choices(range(nPopulation), k=math.ceil(nPopulation*percent_accelerate_mutation))
    # take list solutions and for each solution in given index make some mutation
    for index in random_indexes:
        # 3 mutation at one solution
        for step in range(0, 3):
            random_line = random.choice(range(nLines))
            random_column = random.choice(range(nColumns))
            if list_solutions[index].grid_solution[random_line][random_column] == 1:
                list_solutions[index].grid_solution[random_line][random_column] = 0
            else:
                list_solutions[index].grid_solution[random_line][random_column] = 1
    return list_solutions


def genetic_algo(_rows, _columns):
    converge_counter = 0
    last_max = 0
    # create random solutions (as size of nPopulation)
    list_solutions = random_solutions(_rows, _columns)
    for iteration in range(nIterations):
        print("Num of iteration is:", iteration)
        list_children_solution = crossover(list_solutions, _rows, _columns)
        list_children_solution = mutation_children(list_children_solution)
        list_solutions, max_value, min_value = next_generation(list_solutions, list_children_solution)
        print("Max: ", max_value)
        print("Min: ", min_value)
        print(converge_counter)
        if last_max == max_value:
            converge_counter += 1
        else:
            converge_counter = 0
        if max_value == min_value or converge_counter == 7:
            list_solutions = accelerate_mutations_rate(list_solutions)
            converge_counter = 0
        last_max = max_value
    list_solutions.sort(key=lambda solution: (solution.fitness_score, random.random()), reverse=True)
    solution = list_solutions[0]
    print(solution.fitness_score)
    show_solution(solution.grid_solution)


def show_solution(solution):
    plt.imshow(solution, interpolation='nearest', cmap=cm.Greys)
    plt.show()


if __name__ == '__main__':
    print("Mode algorithm is: ", mode)
    print("File name is: ", NAME_FILE)
    my_data1, rows1, columns1 = read_csv_file()
    genetic_algo(rows1, columns1)
    # show_solution(my_data)