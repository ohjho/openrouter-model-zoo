# OpenRouter Model Zoo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://openrouter-model-zoo.streamlit.app)

This project compares the different models available on OpenRouter, showcasing their performance and capabilities.

# How the App works?
1. load the data from `https://openrouter.ai/api/v1/models` into a Pandas DataFrame
2. Have dropdown box for users to select columns from the dataframe to use as:
  * x-axis
  * y-axis
  * size
3. Create a scatter plot with Plotly Express, using the selected columns as x-axis, y-axis, and size.
  * make sure that each point on hover displays the model name, description, and other relevant information.

# How it's [vibe coded](https://simonwillison.net/2025/Mar/19/vibe-coding/)?
using [zed](https://zed.dev/agentic) and Claude Sonnet 4, I provided the instructions in the [above section](#how-the-app-works) and just watched it go to work.
