# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    l = [[1,2],[3,4]]
    l.append([4,5])
    l.append([5,6])
    l = np.array(l)
    t = torch.from_numpy(l)
    print(t.size())
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
