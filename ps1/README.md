
# Table of Contents

1.  [Simple python and numpy function](#orgb7dd75a)
2.  [Linear regression with one variable](#org64bf370)
    1.  [Plotting the Data](#orgbfd24cb)
    2.  [Gradient Descent](#orgaedc327)

In this exercise, you will implement linear regression and get to see it work on data.

All the information you need for solving this assignment is in directory.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<tbody>
<tr>
<td class="org-right">Section</td>
<td class="org-left">Part</td>
<td class="org-left">Submitted Function</td>
<td class="org-right">Points</td>
</tr>


<tr>
<td class="org-right">1</td>
<td class="org-left">Warm up exercise</td>
<td class="org-left">warmUpExercise</td>
<td class="org-right">10</td>
</tr>


<tr>
<td class="org-right">2</td>
<td class="org-left">Compute cost for one variable</td>
<td class="org-left">computeCost</td>
<td class="org-right">40</td>
</tr>


<tr>
<td class="org-right">3</td>
<td class="org-left">Gradient descent for one variable</td>
<td class="org-left">gradientDescent</td>
<td class="org-right">50</td>
</tr>


<tr>
<td class="org-right">4</td>
<td class="org-left">Feature normalization</td>
<td class="org-left">featureNormalize</td>
<td class="org-right">10</td>
</tr>


<tr>
<td class="org-right">5</td>
<td class="org-left">Compute Cost for multiple variables</td>
<td class="org-left">computeCostMulti</td>
<td class="org-right">20</td>
</tr>


<tr>
<td class="org-right">6</td>
<td class="org-left">Gradient descent for multiple variables</td>
<td class="org-left">gradientDescentMulti</td>
<td class="org-right">20</td>
</tr>


<tr>
<td class="org-right">&#xa0;</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">Total Points</td>
<td class="org-right">130</td>
</tr>
</tbody>
</table>


<a id="orgb7dd75a"></a>

# Simple python and numpy function

The first part of this assignment gives you practice with python and numpy syntax and the homework submission process. In the next code block, you will find the outline of a python function. Modify it to return a 5 x 5 identity matrix by filling in the following code:

A = np.eye(5)

    
    def warmUpExercise(
            matrix
    ):
        """
        Example function in Python which computes the identity matrix.
    
        Returns
        -------
        matrix : array_like
            The 5x5 identity matrix.
    
        Instructions
        ------------
        Return the 5x5 identity matrix.
        """
        # ======== YOUR CODE HERE ======
        identity_matrix =   # modify this line
    
        # ==============================
        return identity

The previous code block only defines the function warmUpExercise. Run it by executing it in your own script. You should see output similar to the following for a 5x5 input matrix:

    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.]])


<a id="org64bf370"></a>

# Linear regression with one variable

Now you will implement linear regression with one variable to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities. You would like to use this data to help you select which city to expand to next.

The file Data/ex1data1.txt contains the dataset for our linear regression problem. The first column is the population of a city (in 10,000s) and the second column is the profit of a food truck in that city (in $10,000s). A negative value for profit indicates a loss.

You are provided with the code needed to load this data. The dataset is loaded from the data file into the variables x and y:

    import numpy as np
    import os
    # Read comma separated data
    data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
    X, y = data[:, 0], data[:, 1]
    
    m = y.size  # number of training examples


<a id="orgbfd24cb"></a>

## Plotting the Data

Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population). Many other problems that you will encounter in real life are multi-dimensional and cannot be plotted on a 2-d plot. There are many plotting libraries in python (see this [blog](https://mode.com/blog/python-data-visualization-libraries/) post for a good summary of the most popular ones).

In this course, we will be mostly be using matplotlib to do all our plotting. matplotlib is one of the most popular scientific plotting libraries in python and has extensive tools and functions to make beautiful plots. pyplot is a module within matplotlib which provides a simplified interface to matplotlib&rsquo;s most common plotting tasks, mimicking MATLAB&rsquo;s plotting interface.

    
    from matplotlib.pyplot as plt
    def plotData(x, y):
        """
        Plots the data points x and y into a new figure. Plots the data
        points and gives the figure axes labels of population and profit.
    
        Parameters
        ----------
        x : array_like
            Data point values for x-axis.
    
        y : array_like
            Data point values for y-axis. Note x and y should have the same size.
    
        Instructions
        ------------
        Plot the training data into a figure using the "figure" and "plot"
        functions. Set the axes labels using the "xlabel" and "ylabel" functions.
        Assume the population and revenue data have been passed in as the x
        and y arguments of this function.
    
        Hint
        ----
        You can use the 'ro' option with plot to have the markers
        appear as red circles. Furthermore, you can make the markers larger by
        using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
        can also set the marker edge color using the `mec` property.
        """
        fig = plt.figure()  # open a new figure
    
        # ====================== YOUR CODE HERE =======================
    
    
        # =============================================================


<a id="orgaedc327"></a>

## Gradient Descent

In this part, you will fit the linear regression parameters \(\theta\) to our dataset using gradient descent.

2.2.1 Update Equations
The objective of linear regression is to minimize the cost function

\[ J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( h_{\theta}(x^{(i)}) - y^{(i)}\right)^2\]
where the hypothesis \(h_\theta(x)\) is given by the linear model\[ h_\theta(x) = \theta^Tx = \theta_0 + \theta_1 x_1\]

Recall that the parameters of your model are the \(\theta_j\) values. These are the values you will adjust to minimize cost \(J(\theta)\). One way to do this is to use the batch gradient descent algorithm. In batch gradient descent, each iteration performs the update

\[ \theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)} \qquad \text{simultaneously update } \theta_j \text{ for all } j\]
With each step of gradient descent, your parameters \(\theta_j\) come closer to the optimal values that will achieve the lowest cost J(\(\theta\)).

