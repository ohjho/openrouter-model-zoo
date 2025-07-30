import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from typing import Dict, Any, List
import numpy as np
import re

# Set page config
st.set_page_config(
    page_title="OpenRouter Model Zoo",
    page_icon="https://unpkg.com/@lobehub/icons-static-png@latest/light/openrouter.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_openrouter_data() -> pd.DataFrame:
    """Load data from OpenRouter API and convert to pandas DataFrame"""
    try:
        url = "https://openrouter.ai/api/v1/models"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        models = data.get("data", [])

        # Flatten the nested data structure
        flattened_models = []
        for model in models:
            flattened_model = {
                "id": model.get("id", ""),
                "name": model.get("name", ""),
                "description": model.get("description", ""),
                "created": model.get("created", 0),
                "context_length": model.get("context_length", 0),
                "modality": model.get("architecture", {}).get("modality", ""),
                "tokenizer": model.get("architecture", {}).get("tokenizer", ""),
                "instruct_type": model.get("architecture", {}).get("instruct_type", ""),
                "prompt_price": float(model.get("pricing", {}).get("prompt", 0)),
                "completion_price": float(
                    model.get("pricing", {}).get("completion", 0)
                ),
                "request_price": float(model.get("pricing", {}).get("request", 0)),
                "image_price": float(model.get("pricing", {}).get("image", 0)),
                "max_completion_tokens": model.get("top_provider", {}).get(
                    "max_completion_tokens", 0
                ),
                "is_moderated": model.get("top_provider", {}).get(
                    "is_moderated", False
                ),
                "hugging_face_id": model.get("hugging_face_id", ""),
                "input_modalities": len(
                    model.get("architecture", {}).get("input_modalities", [])
                ),
                "output_modalities": len(
                    model.get("architecture", {}).get("output_modalities", [])
                ),
                "supported_parameters_count": len(
                    model.get("supported_parameters", [])
                ),
                "supported_parameters": model.get("supported_parameters", []),
            }

            # Add derived fields
            flattened_model["total_price"] = (
                flattened_model["prompt_price"] + flattened_model["completion_price"]
            )
            flattened_model["price_ratio"] = (
                flattened_model["completion_price"] / flattened_model["prompt_price"]
                if flattened_model["prompt_price"] > 0
                else 0
            )
            flattened_model["is_free"] = flattened_model["total_price"] == 0
            flattened_model["context_length_log"] = np.log10(
                max(flattened_model["context_length"], 1)
            )
            flattened_model["provider"] = (
                flattened_model["id"].split("/")[0]
                if "/" in flattened_model["id"]
                else "Unknown"
            )
            flattened_model["creator"] = (
                flattened_model["id"].split("/")[0]
                if "/" in flattened_model["id"]
                else "Unknown"
            )

            # Extract model size from name using regex
            model_name = flattened_model["name"]
            flattened_model["estimated_size"] = 0

            # Look for pattern: number followed by 'B' or 'b', preceded by space or non-digit
            size_pattern = r"(?:^|[^\d])(\d+(?:\.\d+)?)[Bb](?:\s|$|[^\w])"
            match = re.search(size_pattern, model_name)
            if match:
                flattened_model["estimated_size"] = float(match.group(1))

            flattened_models.append(flattened_model)

        df = pd.DataFrame(flattened_models)

        # Convert timestamps to datetime
        df["created_date"] = pd.to_datetime(df["created"], unit="s")

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns suitable for plotting"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove columns that might not be meaningful for visualization
    exclude_cols = ["created"]
    return [col for col in numeric_cols if col not in exclude_cols]


def create_hover_data(
    df: pd.DataFrame, selected_columns: List[str], include_description: bool = False
) -> Dict[str, str]:
    """Create hover data configuration for plotly"""
    hover_data = {}

    # Always include basic info with name first
    hover_data["name"] = True
    hover_data["provider"] = True

    if include_description:
        hover_data["description"] = ":.100"  # Truncate long descriptions

    # Include selected columns if they're not already in basic info
    for col in selected_columns:
        if col not in ["name", "provider", "description"]:
            hover_data[col] = True

    return hover_data


def main():
    st.logo(
        "https://unpkg.com/@lobehub/icons-static-png@latest/light/openrouter.png",
        size="large",
    )
    st.title("OpenRouter Model Zoo")
    st.markdown(
        "Compare different models available on OpenRouter showcasing their performance and capabilities."
    )
    # Inject custom CSS to set the width of the sidebar
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 500px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load data
    with st.spinner("Loading model data from OpenRouter API..."):
        df = load_openrouter_data()

    if df.empty:
        st.error("Failed to load data. Please try again later.")
        return

    st.success(f"Successfully loaded {len(df)} models!")

    # Sidebar controls
    st.sidebar.header("📊 Plot Configuration")

    # Get numeric columns
    numeric_columns = get_numeric_columns(df)
    all_columns = df.columns.tolist()

    # Column selection
    col1, col2 = st.sidebar.columns(2)

    with col1:
        x_axis = st.selectbox(
            "X-Axis:",
            options=numeric_columns,
            index=(
                numeric_columns.index("context_length")
                if "context_length" in numeric_columns
                else 0
            ),
        )

    with col2:
        y_axis = st.selectbox(
            "Y-Axis:",
            options=numeric_columns,
            index=(
                numeric_columns.index("total_price")
                if "total_price" in numeric_columns
                else 1
            ),
        )

    # Size options with better labeling
    enable_bubble_size = st.sidebar.checkbox("Enable bubble size", value=False)

    size_column = None
    if enable_bubble_size:
        size_column = st.sidebar.selectbox(
            "Size (bubble size):",
            options=numeric_columns,
            index=(
                numeric_columns.index("estimated_size")
                if "estimated_size" in numeric_columns
                else 0
            ),
        )

    # Color options
    color_options = [
        "None",
        "provider",
        "modality",
        "is_free",
        "is_moderated",
        "tokenizer",
    ]
    color_column = st.sidebar.selectbox(
        "Color by:", options=color_options, index=1  # Default to 'provider'
    )

    # Hover data options
    st.sidebar.header("🎯 Hover Options")
    include_description = st.sidebar.checkbox(
        "Include description in hover", value=False
    )

    # Filters
    st.sidebar.header("🔍 Filters")

    # Provider filter
    providers = df["provider"].unique()
    selected_providers = st.sidebar.multiselect(
        "Select Providers:", options=sorted(providers), default=sorted(providers)
    )

    # Creator filter
    creators = df["creator"].unique()
    selected_creators = st.sidebar.multiselect(
        "Select Creators:", options=sorted(creators), default=sorted(creators)
    )

    # Modality filter
    modalities = df["modality"].unique()
    selected_modalities = st.sidebar.multiselect(
        "Select Modalities:", options=sorted(modalities), default=sorted(modalities)
    )

    # Price filter with better scaling
    max_price = df["total_price"].max()
    if max_price > 0:
        # Use log scale for better price range handling
        price_range = st.sidebar.slider(
            "Price Range (total, as % of 1-cent):",
            min_value=0.0,
            max_value=float(max_price * 100),
            value=(0.0, float(max_price * 100)),
            step=0.001,
            format="%.3f",
        )
    else:
        price_range = (0.0, 0.0)
    price_range = [i / 100 for i in price_range]

    # Context length filter with 4k increments
    max_context = df["context_length"].max()
    # Ensure minimum is 4k and maximum is at least 2M
    min_context = 4000
    max_context_display = max(int(max_context), 2000000)

    context_range = st.sidebar.slider(
        "Context Length Range:",
        min_value=min_context,
        max_value=max_context_display,
        value=(min_context, max_context_display),
        step=4000,
        format="%d",
    )

    # Supported parameters filter
    if not df.empty:
        # Get all unique supported parameters
        all_params = set()
        for params_list in df["supported_parameters"]:
            if isinstance(params_list, list):
                all_params.update(params_list)

        if all_params:
            selected_params = st.sidebar.multiselect(
                "Supported Parameters:",
                options=sorted(all_params),
                default=[],
                help="Filter models that support specific parameters",
            )
        else:
            selected_params = []
    else:
        selected_params = []

    # Free/Paid model filters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        free_only = st.checkbox("Only free")
    with col2:
        skip_free = st.checkbox("Skip free")

    # Apply filters
    filtered_df = df[
        (df["provider"].isin(selected_providers))
        & (df["creator"].isin(selected_creators))
        & (df["modality"].isin(selected_modalities))
        & (df["total_price"] >= price_range[0])
        & (df["total_price"] <= price_range[1])
        & (df["context_length"] >= context_range[0])
        & (df["context_length"] <= context_range[1])
    ]

    # Apply supported parameters filter
    if selected_params:

        def has_required_params(params_list):
            if not isinstance(params_list, list):
                return False
            return all(param in params_list for param in selected_params)

        filtered_df = filtered_df[
            filtered_df["supported_parameters"].apply(has_required_params)
        ]

    if free_only:
        filtered_df = filtered_df[filtered_df["is_free"] == True]
    elif skip_free:
        filtered_df = filtered_df[filtered_df["is_free"] == False]

    if filtered_df.empty:
        st.warning("No models match the current filters. Please adjust your selection.")
        return

    # Main plot
    st.header("📈 Model Comparison Scatter Plot")

    # Create the plot
    plot_kwargs = {
        "data_frame": filtered_df,
        "x": x_axis,
        "y": y_axis,
        "title": f'{y_axis.replace("_", " ").title()} vs {x_axis.replace("_", " ").title()}',
        "labels": {
            x_axis: x_axis.replace("_", " ").title(),
            y_axis: y_axis.replace("_", " ").title(),
        },
    }

    # Add size if specified
    if size_column is not None:
        plot_kwargs["size"] = size_column
        plot_kwargs["size_max"] = 50

    # Add color if specified
    if color_column != "None":
        plot_kwargs["color"] = color_column

    # Create hover data
    hover_columns = [x_axis, y_axis]
    if size_column is not None:
        hover_columns.append(size_column)
    if color_column != "None" and color_column not in hover_columns:
        hover_columns.append(color_column)

    plot_kwargs["hover_data"] = create_hover_data(
        filtered_df, hover_columns, include_description
    )

    # Create the figure
    fig = px.scatter(**plot_kwargs)

    # Update layout and hover template to show name first
    fig.update_layout(height=600, hovermode="closest", showlegend=True)

    # Customize hover template to show name first
    # fig.update_traces(
    #     hovertemplate="<b>%{customdata[0]}</b><br>"
    #     + "Provider: %{customdata[1]}<br>"
    #     + f"{x_axis.replace('_', ' ').title()}: %{{x}}<br>"
    #     + f"{y_axis.replace('_', ' ').title()}: %{{y}}<br>"
    #     + "<extra></extra>",
    #     customdata=filtered_df[["name", "provider"]].values,
    # )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Models", len(filtered_df))

    with col2:
        free_models = len(filtered_df[filtered_df["is_free"] == True])
        st.metric("Free Models", free_models)

    with col3:
        paid_models = len(filtered_df[filtered_df["is_free"] == False])
        st.metric("Paid Models", paid_models)

    # Data table
    st.header("📋 Model Details")

    # Select columns to display
    display_columns = st.multiselect(
        "Select columns to display:",
        options=all_columns,
        default=[
            "name",
            "provider",
            "context_length",
            "total_price",
            "modality",
            "estimated_size",
        ],
    )

    if display_columns:
        st.dataframe(
            filtered_df[display_columns].sort_values(by=display_columns[0]),
            use_container_width=True,
            hide_index=True,
        )

    # Additional insights
    st.header("💡 Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Providers by Model Count")
        provider_counts = filtered_df["provider"].value_counts().head(10)
        st.bar_chart(provider_counts)

    with col2:
        st.subheader("Modality Distribution")
        modality_counts = filtered_df["modality"].value_counts()
        fig_pie = px.pie(
            values=modality_counts.values,
            names=modality_counts.index,
            title="Models by Modality",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Total Price Distribution")
        # Filter out zero prices for better visualization if there are paid models
        price_data = (
            filtered_df[filtered_df["total_price"] > 0]
            if len(filtered_df[filtered_df["total_price"] > 0]) > 0
            else filtered_df
        )
        fig_hist = px.histogram(
            price_data,
            x="total_price",
            title="Price Distribution",
            nbins=20,
            labels={"total_price": "Total Price", "count": "Number of Models"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        st.subheader("Models by Creator")
        creator_counts = filtered_df["creator"].value_counts().head(15)
        fig_creator = px.bar(
            x=creator_counts.values,
            y=creator_counts.index,
            orientation="h",
            title="Top Creators by Model Count",
            labels={"x": "Number of Models", "y": "Creator"},
        )
        fig_creator.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_creator, use_container_width=True)


if __name__ == "__main__":
    main()
