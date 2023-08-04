
import matplotlib.pyplot as plt
import numpy

def sum_abs_difference_to_mean(lst):
    if not lst:
        return 0
    
    # Step 1: Find the mean value of the list
    mean_value = sum(lst) / len(lst)
    
    # Step 2: Calculate the absolute difference between each element and the mean
    abs_differences = [abs(x - mean_value) for x in lst]
    
    # Step 3: Sum up all the absolute differences
    sum_abs_diff = sum(abs_differences)
    
    return sum_abs_diff



def sum_abs_difference_to_prior_element(lst):
    if not lst or len(lst) < 2:
        return 0
    
    # Initialize the sum of absolute differences to 0
    sum_abs_diff = 0
    
    # Iterate through the list starting from the second element (index 1)
    for i in range(1, len(lst)):
        # Calculate the absolute difference between the current element and its prior element
        abs_diff = abs(lst[i] - lst[i - 1])
        # Add the absolute difference to the running sum
        sum_abs_diff += abs_diff
    
    return sum_abs_diff


x = [0.5, 0.1, 0.3, 0.5, 0.2, 0.4, 0.1, 0.4, 0.5, 0.2, 0.3]
y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
plt.plot(x)
plt.plot(y)


sum_abs_difference_to_mean(x)
sum_abs_difference_to_mean(y)   


sum_abs_difference_to_prior_element(x)
sum_abs_difference_to_prior_element(y)