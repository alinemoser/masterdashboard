# Importing Python Libraries
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import date, datetime
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots 

# Setting Streamlit page configurations: title, icon, and layout
st.set_page_config(page_title="US Health App", page_icon="📊", layout="wide")

very_top_left, very_top_center, very_top_right = st.columns([1, 1.9, 1], vertical_alignment="bottom")

# Dashboard title and description
very_top_center.title("Cybersecurity Breaches Dashboard 🔐🕵️‍♂️")
very_top_center.markdown("_Master's degree in Digital Leadership | Technology - Complementary Assignment | Aline Moser_")

# Sidebar file uploader for CSV file
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file")

# If no file is uploaded, stop execution
if uploaded_file is None:
    st.info(" Upload a file through config", icon="ℹ️")
    st.stop()

#######################################
# DATA LOADING
#######################################

@st.cache_data
def load_data(path: str):
    """Load data from an uploaded Excel file"""
    df = pd.read_excel(path)
    return df

# Load data using the uploaded file
df = load_data(uploaded_file)

#######################################
# DATA PREPARATION
#######################################

# Convert 'Individuals_Affected' to integer type
df['Individuals_Affected'] = df['Individuals_Affected'].astype(int)

# Remove any extra spaces in 'Breach_Date'
df['Breach_Date'] = df['Breach_Date'].str.strip()

# Convert 'Breach_Date' to datetime format, handling errors as NaT
df['Breach_Date'] = pd.to_datetime(df['Breach_Date'], infer_datetime_format=True, errors='coerce')

# Create new 'Month' and 'Year' columns from 'Breach_Date'
df['Month'] = df['Breach_Date'].dt.month.astype(int)
df['Year'] = df['Breach_Date'].dt.year.astype(int)

# Fill missing 'Description' values with 'Not Informed'
df['Description'] = df['Description'].fillna('Not Informed')

# Adding Latitude and Longitude based on the state abbreviation
state_coords = {
    'AL': {'latitude': 32.806671, 'longitude': -86.791130},
    'AK': {'latitude': 61.370716, 'longitude': -152.404419},
    'AZ': {'latitude': 33.729759, 'longitude': -111.431221},
    'AR': {'latitude': 34.969704, 'longitude': -92.373123},
    'CA': {'latitude': 36.116203, 'longitude': -119.681564},
    'CO': {'latitude': 39.059811, 'longitude': -105.311104},
    'CT': {'latitude': 41.597782, 'longitude': -72.755371},
    'DC': {'latitude': 38.89511, 'longitude': -77.03637},
    'DE': {'latitude': 39.318523, 'longitude': -75.507141},
    'FL': {'latitude': 27.766279, 'longitude': -81.686783},
    'GA': {'latitude': 33.040619, 'longitude': -83.643074},
    'HI': {'latitude': 21.094318, 'longitude': -157.498337},
    'ID': {'latitude': 44.240459, 'longitude': -114.478828},
    'IL': {'latitude': 40.349457, 'longitude': -88.986137},
    'IN': {'latitude': 39.849426, 'longitude': -86.258278},
    'IA': {'latitude': 42.011539, 'longitude': -93.210526},
    'KS': {'latitude': 38.526600, 'longitude': -96.726486},
    'KY': {'latitude': 37.668140, 'longitude': -84.670067},
    'LA': {'latitude': 31.169546, 'longitude': -91.867805},
    'ME': {'latitude': 44.693947, 'longitude': -69.381927},
    'MD': {'latitude': 39.063946, 'longitude': -76.802101},
    'MA': {'latitude': 42.230171, 'longitude': -71.530106},
    'MI': {'latitude': 43.326618, 'longitude': -84.536095},
    'MN': {'latitude': 45.694454, 'longitude': -93.900192},
    'MS': {'latitude': 32.741646, 'longitude': -89.678696},
    'MO': {'latitude': 38.456085, 'longitude': -92.288368},
    'MT': {'latitude': 46.921925, 'longitude': -110.454353},
    'NE': {'latitude': 41.125370, 'longitude': -98.268082},
    'NV': {'latitude': 38.313515, 'longitude': -117.055374},
    'NH': {'latitude': 43.452492, 'longitude': -71.563896},
    'NJ': {'latitude': 40.298904, 'longitude': -74.521011},
    'NM': {'latitude': 34.840515, 'longitude': -106.248482},
    'NY': {'latitude': 42.165726, 'longitude': -74.948051},
    'NC': {'latitude': 35.630066, 'longitude': -79.806419},
    'ND': {'latitude': 47.528912, 'longitude': -99.784012},
    'OH': {'latitude': 40.388783, 'longitude': -82.764915},
    'OK': {'latitude': 35.565342, 'longitude': -96.928917},
    'OR': {'latitude': 44.572021, 'longitude': -122.070938},
    'PA': {'latitude': 40.590752, 'longitude': -77.209755},
    'PR': {'latitude': 18.220833, 'longitude': -66.590149},
    'RI': {'latitude': 41.680893, 'longitude': -71.511780},
    'SC': {'latitude': 33.856892, 'longitude': -80.945007},
    'SD': {'latitude': 44.299782, 'longitude': -99.438828},
    'TN': {'latitude': 35.747845, 'longitude': -86.692345},
    'TX': {'latitude': 31.054487, 'longitude': -97.563461},
    'UT': {'latitude': 40.150032, 'longitude': -111.862434},
    'VT': {'latitude': 44.045876, 'longitude': -72.710686},
    'VA': {'latitude': 37.769337, 'longitude': -78.169968},
    'WA': {'latitude': 47.400902, 'longitude': -121.490494},
    'WV': {'latitude': 38.491226, 'longitude': -80.954456},
    'WI': {'latitude': 44.268543, 'longitude': -89.616508},
    'WY': {'latitude': 42.755966, 'longitude': -107.302490}
}

# Map state abbreviations to their corresponding latitude and longitude
df['lat'] = df['State'].map(lambda x: state_coords[x]['latitude'])
df['lon'] = df['State'].map(lambda x: state_coords[x]['longitude'])
# df['lat'], df['lon'] = zip(*df['State'].map(state_coords).fillna((np.nan, np.nan)))

#######################################
# VISUALIZATION METHODS
#######################################

def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value= value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        height=130,
    )

    st.plotly_chart(fig, use_container_width=True)

##################################################################

# ----- Auxiliar Dataframes -----

# Split and explode 'Breach_Type' into individual entries in an new auxiliar df
df['Breach_Type'] = df['Breach_Type'].str.split(',').apply(lambda x: [i.strip() for i in x])
df_breach_types_exploded = df.explode('Breach_Type')
# Convert list columns back to strings
df['Breach_Type'] = df['Breach_Type'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Split and explode 'Breach_Type' into individual entries in an new auxiliar df
df['Breach_Location_Information'] = df['Breach_Location_Information'].str.split(',').apply(lambda x: [i.strip() for i in x])
df_breach_locations_exploded = df.explode('Breach_Location_Information')
# Convert lists to strings using apply() and join() to transform lists into a single string
df['Breach_Location_Information'] = df['Breach_Location_Information'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Calculate the sum of 'Individuals_Affected' for different breach types
sum_individuals_per_breach = df_breach_types_exploded.groupby('Breach_Type')['Individuals_Affected'].sum().reset_index()
df_affected_by_type = df_breach_types_exploded.groupby('Breach_Type').agg({'Individuals_Affected': 'sum'}).reset_index()
total_hacking_affected = df_breach_types_exploded[df_breach_types_exploded['Breach_Type'] == 'Hacking/IT Incident']['Individuals_Affected'].sum()
total_theft_affected = df_breach_types_exploded[df_breach_types_exploded['Breach_Type'] == 'Theft']['Individuals_Affected'].sum()
total_loss_affected = df_breach_types_exploded[df_breach_types_exploded['Breach_Type'] == 'Loss']['Individuals_Affected'].sum()
total_unauthorized_affected = df_breach_types_exploded[df_breach_types_exploded['Breach_Type'] == 'Unauthorized Access/Disclosure']['Individuals_Affected'].sum()

# Total individuals affected
total_affected = df['Individuals_Affected'].sum()
total_cases = df.shape[0]

# Count of unique types
unique_types = sorted(df_breach_types_exploded['Breach_Type'].unique())
unique_types_count = len(unique_types)

# Count of unique locations
unique_locations = sorted(df_breach_locations_exploded['Breach_Location_Information'].unique())
unique_locations_count = len(unique_locations)

# Count of unique states
unique_states = sorted(df['State'].unique())
unique_states_count = len(unique_states)

oldest_date = df['Breach_Date'].min()
newest_date = df['Breach_Date'].max()

# Display data with number formatting in a Streamlit Expander
with st.expander("Data Preview"):
    st.dataframe(
        df,
        column_config={
             "Individuals_Affected": st.column_config.NumberColumn(format="%d"),
             "Month": st.column_config.NumberColumn(format="%d"),
             "Year": st.column_config.NumberColumn(format="%d")
           })

# Function to apply various filters to the DataFrame
def apply_filters(df, selected_state=None, selected_min_date=None, selected_max_date=None,selected_type=None):
    """Apply filters to the DataFrame based on selected state, year, and breach type"""
    if selected_state:
        df = df[df['State'].isin(selected_state)]
    
    if selected_min_date:
        selected_min_date = pd.to_datetime(selected_min_date)
        df = df[df['Breach_Date'] >= selected_min_date]
    
    if selected_max_date:
        selected_max_date = pd.to_datetime(selected_max_date)
        df = df[df['Breach_Date'] <= selected_max_date]
    
    if selected_type:
        df = df[df['Breach_Type'].isin(selected_type)]
    
    return df

# Function to get individuals affected per state with latitude and longitude
def get_individuals_affected_per_state(df):
    
    # Group data by state and sum affected individuals
    individuals_affected = df.groupby('State')['Individuals_Affected'].sum().reset_index()

    # Bring in lat/lon coordinates and merge with affected individuals
    return df.drop_duplicates(subset='State')[['State', 'lat', 'lon']].merge(
        individuals_affected, on='State', how='inner'
    ).sort_values('Individuals_Affected', ascending=False)

# function to get Individuals Affected per Date
def get_individuals_affected_per_date(df):
    
    # Group data by state and sum affected individuals
    individuals_affected = df.set_index('Breach_Date') \
        .groupby(pd.Grouper(freq = 'M'))['Individuals_Affected'] \
            .sum().reset_index()

    individuals_affected['Year'] = individuals_affected['Breach_Date'].dt.year
    individuals_affected['Month'] = individuals_affected['Breach_Date'].dt.month_name()
    
    return individuals_affected

# function to get Individuals Affected per Type
def get_individuals_affected_per_type(df):
    
    # Group data by state and sum affected individuals
    individuals_affected = df.groupby('Breach_Type')['Individuals_Affected'] \
        .sum().reset_index()
    
    return individuals_affected

# Function to Plot the Line Chart - Individuals Affected per Month each selected Year
def plot_line_char(df): 

    fig = px.line(get_individuals_affected_per_date(df),
                                            x = 'Month',
                                            y = 'Individuals_Affected',
                                            markers = True,
                                            range_y = (0,get_individuals_affected_per_date(df).max()),
                                            color = 'Year',
                                            line_dash = 'Year')

    fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'), yaxis_title = 'Affected Individuals', width=1200, height=600)

    fig.update_traces(marker=dict(size=4, symbol='circle', color='black'))

    st.plotly_chart(fig, use_container_width=True)

   

# Map Chart -> Build a map chart based on breach type
def plot_map_chart(df):
    
    # Build Map Chart/Graph
    fig = px.scatter_geo(get_individuals_affected_per_state(df),
                         lat='lat',
                         lon='lon',
                         scope='usa',
                         size='Individuals_Affected',
                         template='seaborn',
                         hover_name='State',
                         hover_data={'lat': False, 'lon': False})
    
    # Layout settings
    fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'), width=1200, height=600)

    # Plot the chart on Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Bar Chart 1
def plot_affected_per_type_bar(df):

    fig = px.bar(
        get_individuals_affected_per_type(df),
        x="Breach_Type",
        y="Individuals_Affected",
        color="Individuals_Affected",
    ) 

    fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'), yaxis_title = 'Affected Individuals', width=1200, height=600)
    st.plotly_chart(fig, use_container_width=True)

# Streamlit Visualization
left, right = st.columns(2, vertical_alignment="bottom")

selected_state = left.multiselect('Select the State', unique_states)
selected_type = right.multiselect('Select the Type of Breach', unique_types)

selected_min_date = st.date_input(
     'First Date', value=oldest_date, 
     min_value=oldest_date, max_value=newest_date, format="YYYY/MM/DD", disabled=False, label_visibility="visible")

selected_max_date = st.date_input(
     'Last Date', value=newest_date, 
     min_value=oldest_date, max_value=newest_date, format="YYYY/MM/DD", disabled=False, label_visibility="visible")

middle_column_1, middle_column_2, middle_column_3, middle_column_4 = st.columns(4)

with middle_column_1:
        
        # Metric: Total number of Individuals Affected
        plot_metric(
            "Affected Ever",
            total_affected,
            prefix="",
            suffix="",
            show_graph=True,
            color_graph="rgba(0, 104, 201, 0.2)",
        )

with middle_column_2:

        # Metric: Total Case in the US
        plot_metric(
            "Cases in the US",
            total_cases,
            prefix="",
            suffix="",
            show_graph=True,
            color_graph="rgba(0, 104, 201, 0.2)",
        )

with middle_column_3:

        # Metric: Count of Breach Types
        plot_metric(
            "Breach Types",
            unique_types_count,
            prefix="",
            suffix="",
            show_graph=True,
            color_graph="rgba(0, 104, 201, 0.2)",
        )

with middle_column_4:

        # Metric: Count of Breach Locations
        plot_metric(
            "Breach Locations",
            unique_locations_count,
            prefix="",
            suffix="",
            show_graph=True,
            color_graph="rgba(0, 104, 201, 0.2)",
        )

bottom_left_column, bottom_middle_column, bottom_right_column = st.columns(3)

filtered_df = apply_filters(df, selected_state, selected_min_date, selected_max_date, selected_type)
filtered_types_exploded_df = apply_filters(df_breach_types_exploded, selected_state, selected_min_date, selected_max_date, selected_type)

with bottom_left_column:
    st.header("Affected Individuals per Year", divider=True)
    plot_line_char(filtered_df)
    #st.caption("Optional text")

with bottom_middle_column:
    st.header("Affected Individuals per State", divider=True)
    plot_map_chart(filtered_df)
    #st.caption("Optional text")

with bottom_right_column:
    st.header("Affected Individuals by Type", divider=True)
    plot_affected_per_type_bar(filtered_types_exploded_df)
    #st.caption("Optional text")