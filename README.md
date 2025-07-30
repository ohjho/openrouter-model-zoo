# OpenRouter Model Zoo
this project compares the different models available on OpenRouter, showcasing their performance and capabilities.

# How It's Made
1. load the data from `https://openrouter.ai/api/v1/models` into a Pandas DataFrame
2. Have dropdown box for users to select columns from the dataframe to use as:
  - x-axis
  - y-axis
  - size
3. Create a scatter plot with Plotly Express, using the selected columns as x-axis, y-axis, and size.
  - make sure that each point on hover displays the model name, description, and other relevant information.
