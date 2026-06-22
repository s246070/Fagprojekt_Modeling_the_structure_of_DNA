import os

with open(os.path.join(os.path.dirname(__file__), "cell_types_subset_1.txt"), "r") as f:
    cell_types = [line.strip() for line in f]


cell_list = {i: 0 for i in cell_types}
for i in range(len(cell_types)):
    cell_list[cell_types[i]] += 1

print(cell_list)
    