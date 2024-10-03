#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#                                   MATPLOTLIB ASSIGNMENT:

#                     (Use Matplotlib for the visualization of the given questions)
                     

# 1. Create a scatter plot using Matplotlib to visualize the relationship between two arrays, x and y for the given
# data.
#         x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
#         y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]

# 2. Generate a line plot to visualize the trend of values for the given data.

#        data = np.array([3, 7, 9, 15, 22, 29, 35])

# 3. Display a bar chart to represent the frequency of each item in the given array categories.
  
#        categories = ['A', 'B', 'C', 'D', 'E'] 
#         values = [25, 40, 30, 35, 20]

# 4. Create a histogram to visualize the distribution of values in the array data.
  
#       data = np.random.normal(0, 1, 1000)
        
# 5. Show a pie chart to represent the percentage distribution of different sections in the array `sections`.

#      sections = ['Section A', 'Section B', 'Section C', 'Section D'] 
#      sizes = [25, 30, 15, 30]



# In[1]:


#Ques 1

#Create a scatter plot using Matplotlib to visualize the relationship between two arrays, x and y for the given
# data.
#         x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
#         y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]

import matplotlib.pyplot as plt

# Given data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]

# Create a scatter plot
plt.scatter(x, y, color='blue', label='Data Points')

# Add labels and title
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Scatter Plot of X vs Y')

# Add grid
plt.grid(True)

# Display the legend
plt.legend()

# Show the plot
plt.show()



# In[2]:


#Ques 2

#Generate a line plot to visualize the trend of values for the given data.

#        data = np.array([3, 7, 9, 15, 22, 29, 35])

import matplotlib.pyplot as plt
import numpy as np

# Data array
data = np.array([3, 7, 9, 15, 22, 29, 35])

# Generate the line plot
plt.plot(data, marker='o', color='blue', label='Value Trend')

# Add labels for axes and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Trend of Values')

# Display a grid for better readability
plt.grid(True)

# Show the legend
plt.legend()

# Display the plot
plt.show()


# In[3]:


#Ques 3

#Display a bar chart to represent the frequency of each item in the given array categories.
  
#        categories = ['A', 'B', 'C', 'D', 'E'] 
#         values = [25, 40, 30, 35, 20]

import matplotlib.pyplot as plt

# Data arrays
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 35, 20]

# Generate the bar chart
plt.bar(categories, values, color='skyblue')

# Add labels for axes and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Frequency of Each Item in Categories')

# Display the plot
plt.show()


# In[4]:


# Ques 4

# Create a histogram to visualize the distribution of values in the array data.
  
#       data = np.random.normal(0, 1, 1000)

import matplotlib.pyplot as plt
import numpy as np

# Generate a random array with normal distribution
data = np.random.normal(0, 1, 1000)

# Create the histogram
plt.hist(data, bins=30, color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normally Distributed Data')

# Display the plot
plt.show()


# In[5]:


# Ques 5

# Show a pie chart to represent the percentage distribution of different sections in the array `sections`.

#      sections = ['Section A', 'Section B', 'Section C', 'Section D'] 
#      sizes = [25, 30, 15, 30]

import matplotlib.pyplot as plt

# Data arrays
sections = ['Section A', 'Section B', 'Section C', 'Section D']
sizes = [25, 30, 15, 30]

# Create the pie chart
plt.pie(sizes, labels=sections, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'gold'])

# Add a title
plt.title('Percentage Distribution of Sections')

# Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.axis('equal')

# Display the plot
plt.show()


# In[ ]:


#                                 SEABORN ASSIGNMENT:
#                  (Use Seaborn for the visualization of the given questions)
            
# 1. Create a scatter plot to visualize the relationship between two variables, by generating a synthetic
# dataset.


# 2. Generate a dataset of random numbers. Visualize the distribution of a numerical variable.


# 3. Create a dataset representing categories and their corresponding values. Compare different categories
# based on numerical values.


# 4. Generate a dataset with categories and numerical values. Visualize the distribution of a numerical
# variable across different categories.


# 5. Generate a synthetic dataset with correlated features. Visualize the correlation matrix of a dataset using a
# heatmap


# In[6]:


# Ques 1

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)  # For reproducibility
x = np.random.rand(100) * 100  # Random values for x between 0 and 100
y = 2.5 * x + np.random.randn(100) * 10  # Linear relation with some noise

# Create a scatter plot
plt.scatter(x, y, color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Scatter Plot: Relationship between X and Y')

# Show the plot
plt.show()


# In[7]:


# Ques 2

#Generate a dataset of random numbers. Visualize the distribution of a numerical variable

import numpy as np
import matplotlib.pyplot as plt

# Generate a dataset of random numbers (normally distributed)
np.random.seed(42)  # For reproducibility
data = np.random.randn(1000)  # 1000 random numbers from a standard normal distribution

# Create a histogram to visualize the distribution
plt.hist(data, bins=30, color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram: Distribution of Random Numbers')

# Show the plot
plt.show()


# In[8]:


# Ques 3

# Create a dataset representing categories and their corresponding values. Compare different categories
# based on numerical values.

import matplotlib.pyplot as plt

# Dataset representing categories and corresponding values
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
values = [45, 30, 60, 25, 50]

# Create a bar chart to compare categories based on values
plt.bar(categories, values, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Comparison of Categories Based on Values')

# Show the plot
plt.show()


# In[9]:


# Ques 4

# Generate a dataset with categories and numerical values. Visualize the distribution of a numerical
# variable across different categories.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Step 1: Generate a dataset
np.random.seed(42)  # For reproducibility
categories = ['Category A', 'Category B', 'Category C']
values_A = np.random.normal(50, 10, 100)  # 100 values around 50 with std of 10
values_B = np.random.normal(70, 15, 100)  # 100 values around 70 with std of 15
values_C = np.random.normal(30, 5, 100)   # 100 values around 30 with std of 5

# Combine the data into a pandas DataFrame
df = pd.DataFrame({
    'Category': np.repeat(categories, 100),
    'Value': np.concatenate([values_A, values_B, values_C])
})

# Step 2: Visualize the distribution using a box plot
plt.figure(figsize=(8, 6))
df.boxplot(by='Category', column='Value', grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

# Add labels and title
plt.title('Distribution of Numerical Values Across Categories')
plt.suptitle('')  # Suppress the automatic title to keep only the custom one
plt.xlabel('Category')
plt.ylabel('Value')

# Show the plot
plt.show()


# In[10]:


# Ques 5

# Generate a synthetic dataset with correlated features. Visualize the correlation matrix of a dataset using a
# heatmap.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Generate a synthetic dataset with correlated features
np.random.seed(42)  # For reproducibility

# Generate a random dataset
n_samples = 100
feature_1 = np.random.rand(n_samples)  # First feature
feature_2 = feature_1 + np.random.normal(0, 0.1, n_samples)  # Second feature correlated with the first
feature_3 = feature_1 * 0.5 + np.random.rand(n_samples) * 0.5  # Third feature correlated with the first

# Combine features into a DataFrame
data = pd.DataFrame({
    'Feature 1': feature_1,
    'Feature 2': feature_2,
    'Feature 3': feature_3
})

# Step 2: Compute the correlation matrix
correlation_matrix = data.corr()

# Step 3: Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar=True)

# Add labels and title
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[ ]:


#                              PLOTLY ASSIGNMENT:

#             (Use Plotly for the visualization of the given questions)
            
            
# 1. Using the given dataset, to generate a 3D scatter plot to visualize the distribution of data points in a threedimensional space.


#         np.random.seed(30) 
#         data = { 
#             'X': np.random.uniform(-10, 10, 300), 
#             'Y': np.random.uniform(-10, 10, 300), 
#             'Z': np.random.uniform(-10, 10, 300) 
#         } 
#         df = pd.DataFrame(data)

        
# 2. Using the Student Grades, create a violin plot to display the distribution of scores across different grade
# categories.

#         np.random.seed(15) 
#         data = { 
#             'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200), 
#             'Score': np.random.randint(50, 100, 200) 
#         } 
#         df = pd.DataFrame(data
#          Using the sales data, generate a heatmap to visualize the variation in sales across
#         different months and days.

#         np.random.seed(20) 
#         data = { 
#             'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100), 
#             'Day': np.random.choice(range(1, 31), 100), 
#             'Sales': np.random.randint(1000, 5000, 100) 
#         } 
#         df = pd.DataFrame(data)
                          

# 3. Using the sales data, generate a heatmap to visualize the variation in sales across different months and
# days.
                          
                          
#         np.random.seed(20) 
#         data = { 
#             'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100), 
#             'Day': np.random.choice(range(1, 31), 100), 
#             'Sales': np.random.randint(1000, 5000, 100) 
#         } 
#         df = pd.DataFrame(data)
                        
                    
# 4. Using the given x and y data, generate a 3D surface plot to visualize the function z=sin(√x^2+y^2)
                          
                        
#         x = np.linspace(-5, 5, 100) 
#         y = np.linspace(-5, 5, 100) 
#         x, y = np.meshgrid(x, y) 
#         z = np.sin(np.sqrt(x**2 + y**2))  
#         data = { 
#             'X': x.flatten(), 
#             'Y': y.flatten(), 
#             'Z': z.flatten() 
#         } 
#         df = pd.DataFrame(data)  
  
    
# 5. Using the given dataset, create a bubble chart to represent each country's population (y-axis), GDP (xaxis), and bubble size proportional to the population.
                          
#         np.random.seed(25) 
#         data = { 
#             'Country': ['USA', 'Canada', 'UK',
#         'Germany', 'France'], 
#             'Population':
#         np.random.randint(100, 1000, 5), 
#             'GDP': np.random.randint(500, 2000,
#         5) 
#         } 
#         df = pd.DataFrame(data)                          
              
                          
                          


# In[1]:


# Ques 1

# Using the given dataset, to generate a 3D scatter plot to visualize the distribution of data points in a threedimensional space.


#         np.random.seed(30) 
#         data = { 
#             'X': np.random.uniform(-10, 10, 300), 
#             'Y': np.random.uniform(-10, 10, 300), 
#             'Z': np.random.uniform(-10, 10, 300) 
#         } 
#         df = pd.DataFrame(data)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate the dataset
np.random.seed(30)
data = {
    'X': np.random.uniform(-10, 10, 300),
    'Y': np.random.uniform(-10, 10, 300),
    'Z': np.random.uniform(-10, 10, 300)
}
df = pd.DataFrame(data)

# Step 2: Create the 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Step 3: Plot the points
ax.scatter(df['X'], df['Y'], df['Z'], c='b', marker='o')

# Step 4: Set labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Scatter Plot of Data Points')

# Show the plot
plt.show()



# In[62]:


# Ques 2

#  Using the Student Grades, create a violin plot to display the distribution of scores across different grade
# categories.

#         np.random.seed(15) 
#         data = { 
#             'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200), 
#             'Score': np.random.randint(50, 100, 200) 
#         } 
#         df = pd.DataFrame(data
#          Using the sales data, generate a heatmap to visualize the variation in sales across
#         different months and days.

#         np.random.seed(20) 
#         data = { 
#             'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100), 
#             'Day': np.random.choice(range(1, 31), 100), 
#             'Sales': np.random.randint(1000, 5000, 100) 
#         } 
#         df = pd.DataFrame(data)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Generate the dataset
np.random.seed(15)
data = {
    'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200),
    'Score': np.random.randint(50, 100, 200)
}
df = pd.DataFrame(data)

# Step 2: Create the violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x='Grade', y='Score', data=df, palette='Set3')

# Step 3: Set labels and title
plt.xlabel('Grade')
plt.ylabel('Score')
plt.title('Distribution of Scores Across Grade Categories')

# Display the plot
plt.show()



# In[5]:


# Ques 3

#  Using the sales data, generate a heatmap to visualize the variation in sales across different months and
# days.
                          
                          
#         np.random.seed(20) 
#         data = { 
#             'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100), 
#             'Day': np.random.choice(range(1, 31), 100), 
#             'Sales': np.random.randint(1000, 5000, 100) 
#         } 
#         df = pd.DataFrame(data)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Set the seed and generate the dataset
np.random.seed(20)
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df = pd.DataFrame(data)

# Step 2: Pivot the data for the heatmap
sales_pivot = df.pivot_table(index='Day', columns='Month', values='Sales', aggfunc='mean')

# Step 3: Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(sales_pivot, cmap='coolwarm', annot=True, fmt=".1f")

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Day')
plt.title('Sales Variation Across Different Months and Days')

# Display the heatmap
plt.show()


# In[10]:


# Ques 4

# Using the given x and y data, generate a 3D surface plot to visualize the function z=sin(√x^2+y^2)
                          
                        
#         x = np.linspace(-5, 5, 100) 
#         y = np.linspace(-5, 5, 100) 
#         x, y = np.meshgrid(x, y) 
#         z = np.sin(np.sqrt(x**2 + y**2))  
#         data = { 
#             'X': x.flatten(), 
#             'Y': y.flatten(), 
#             'Z': z.flatten() 
#         } 
#         df = pd.DataFrame(data)  

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate x, y, and z data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Create a 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface
surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Adding labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Surface Plot: z = sin(sqrt(x^2 + y^2))')

# Adding color bar for the surface
fig.colorbar(surf)

# Show plot
plt.show()


# In[11]:


# Ques 5

# Using the given dataset, create a bubble chart to represent each country's population (y-axis), GDP (xaxis), and bubble size proportional to the population.
                          
#         np.random.seed(25) 
#         data = { 
#             'Country': ['USA', 'Canada', 'UK',
#         'Germany', 'France'], 
#             'Population':
#         np.random.randint(100, 1000, 5), 
#             'GDP': np.random.randint(500, 2000,
#         5) 
#         } 
#         df = pd.DataFrame(data) 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create the dataset
np.random.seed(25)
data = {
    'Country': ['USA', 'Canada', 'UK', 'Germany', 'France'],
    'Population': np.random.randint(100, 1000, 5),
    'GDP': np.random.randint(500, 2000, 5)
}
df = pd.DataFrame(data)

# Create the bubble chart
plt.figure(figsize=(10, 6))

# Scatter plot with bubble sizes proportional to population
plt.scatter(df['GDP'], df['Population'], s=df['Population']*10, alpha=0.6, edgecolors="w", linewidth=2)

# Adding labels
for i, country in enumerate(df['Country']):
    plt.text(df['GDP'][i], df['Population'][i], country, fontsize=12)

# Label the axes
plt.xlabel('GDP (in billions)')
plt.ylabel('Population (in millions)')

# Title of the plot
plt.title('Bubble Chart: Population vs GDP of Countries')

# Show the plot
plt.show()



# In[ ]:


#                                         BOKEH ASSIGNMENT:

#                          (Use Bokeh for the visualization of the given questions)
            
            
# 1.Create a Bokeh plot displaying a sine wave. Set x-values from 0 to 10 and y-values as the sine of x


# 2.Create a Bokeh scatter plot using randomly generated x and y values. Use different sizes and colors for the
# markers based on the 'sizes' and 'colors' columns.


# 3. Generate a Bokeh bar chart representing the counts of different fruits using the following dataset.

#         fruits = ['Apples', 'Oranges', 'Bananas', 'Pears'] 
#         counts = [20, 25, 30, 35]
        
# 4. Create a Bokeh histogram to visualize the distribution of the given data.

#         data_hist = np.random.randn(1000) 
#         hist, edges = np.histogram(data_hist, bins=30)
        
# 5. Create a Bokeh heatmap using the provided dataset.

#         data_heatmap = np.random.rand(10, 10) 
#         x = np.linspace(0, 1, 10) 
#         y = np.linspace(0, 1, 10) 
#         xx, yy = np.meshgrid(x, y)


# In[54]:


# Ques 1

# Create a Bokeh plot displaying a sine wave. Set x-values from 0 to 10 and y-values as the sine of x


import numpy as np
from bokeh.plotting import figure, show

# Generate x and y values for the sine function
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a Bokeh figure
p = figure(title="Sine Wave", x_axis_label='x', y_axis_label='sin(x)')

# Add a line plot to the figure
p.line(x, y, line_width=2)

# Show the plot
show(p)



# In[55]:


# Ques 2

# Create a Bokeh scatter plot using randomly generated x and y values. Use different sizes and colors for the
# markers based on the 'sizes' and 'colors' columns.

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import numpy as np

import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

# Generate random x and y values
N = 100
x = np.random.rand(N)
y = np.random.rand(N)

# Generate random sizes and colors
sizes = np.random.randint(10, 50, size=N)
colors = ["#%06x" % np.random.randint(0, 0xFFFFFF) for _ in range(N)]

# Create a ColumnDataSource to hold the data
source = ColumnDataSource(data={'x': x, 'y': y, 'sizes': sizes, 'colors': colors})

# Create a Bokeh figure
p = figure(title="Scatter Plot with Random Data", x_axis_label='x', y_axis_label='y')

# Add a scatter plot to the figure, using the ColumnDataSource
p.scatter(x='x', y='y', size='sizes', fill_color='colors', source=source)

# Show the plot
show(p)


# In[56]:


# Ques 3

# Generate a Bokeh bar chart representing the counts of different fruits using the following dataset.

#         fruits = ['Apples', 'Oranges', 'Bananas', 'Pears'] 
#         counts = [20, 25, 30, 35]

import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

# Enable output in the notebook (if running in a notebook environment)
output_notebook()

# Sample dataset
fruits = ['Apples', 'Oranges', 'Bananas', 'Pears']
counts = [20, 25, 30, 35]

# Create a Bokeh figure with the x_range set to the list of fruits
p = figure(x_range=fruits, x_axis_label='Fruits', y_axis_label='Count', title='Fruit Counts')

# Add a bar chart to the figure
p.vbar(x=fruits, top=counts, width=0.9)

# Show the plot
show(p)



# In[53]:


# Ques 4

#  Create a Bokeh histogram to visualize the distribution of the given data.

#         data_hist = np.random.randn(1000) 
#         hist, edges = np.histogram(data_hist, bins=30)

import numpy as np
from bokeh.plotting import figure, show

# Generate random data
data_hist = np.random.randn(1000)

# Calculate histogram
hist, edges = np.histogram(data_hist, bins=30)

# Create Bokeh figure
p = figure(title="Histogram of Random Data", x_axis_label='Value', y_axis_label='Frequency')

# Add histogram bars
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color='gray')

# Show the plot
show(p)


# In[51]:


# Ques 5

# Create a Bokeh heatmap using the provided dataset.

#         data_heatmap = np.random.rand(10, 10) 
#         x = np.linspace(0, 1, 10) 
#         y = np.linspace(0, 1, 10) 
#         xx, yy = np.meshgrid(x, y)

from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.io import output_file
import numpy as np

# Generate the heatmap data
data_heatmap = np.random.rand(10, 10)
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
xx, yy = np.meshgrid(x, y)

# Create a Bokeh plot for heatmap
p = figure(title="Heatmap", tools="hover", tooltips=[("Value", "@image")],
           x_axis_label='X-axis', y_axis_label='Y-axis')

# Set up color mapper
color_mapper = LinearColorMapper(palette="Viridis256", low=data_heatmap.min(), high=data_heatmap.max())

# Add the heatmap image
p.image(image=[data_heatmap], x=0, y=0, dw=1, dh=1, color_mapper=color_mapper)

# Add color bar to show the scale
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0, 0))
p.add_layout(color_bar, 'right')

# Save and display the plot
output_file("heatmap.html")
show(p)



# In[ ]:





# In[ ]:




