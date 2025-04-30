#!/usr/bin/env python
# coding: utf-8

# ## NULLCLASS Training Project

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import webbrowser
import os


# In[2]:


# Load the Dataset
apps_df = pd.read_csv('Play Store Data.csv')
reviews_df = pd.read_csv('User Reviews.csv')


# In[3]:


apps_df.head()


# In[4]:


reviews_df.head()


# In[5]:


# Data Cleaning
apps_df = apps_df.dropna(subset=['Rating'])

for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0], inplace=True)

apps_df.drop_duplicates(inplace=True)
apps_df = apps_df[apps_df['Rating'] <= 5]
reviews_df.dropna(subset=['Translated_Review'], inplace=True)


# In[6]:


# Merge datasets on 'App' and handle non-matching apps
merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')


# In[7]:


apps_df.info()


# In[8]:


# Step 3: Data Transformation
apps_df['Reviews'] = apps_df['Reviews'].astype(int)
apps_df['Installs'] = apps_df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)


# In[9]:


# Converting the units for size to MegaBytes
def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    else:
        return np.nan
apps_df['Size'] = apps_df['Size'].apply(convert_size)


# In[10]:


# Add log_installs and log_reviews columns
apps_df['Log_Installs'] = np.log1p(apps_df['Installs'])
apps_df['Log_Reviews'] = np.log1p(apps_df['Reviews'])


# In[11]:


# Add Rating Group column
def rating_group(rating):
    if rating >= 4:
        return 'Top rated'
    elif rating >= 3:
        return 'Above average'
    elif rating >= 2:
        return 'Average'
    else:
        return 'Below average'

apps_df['Rating_Group'] = apps_df['Rating'].apply(rating_group)


# In[12]:


# Add Revenue column
apps_df['Revenue'] = apps_df['Price'] * apps_df['Installs']


# In[13]:


# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
reviews_df['Sentiment_Score'] = reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# In[14]:


# Extract year from 'Last Updated' and create 'Year' column
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df['Year'] = apps_df['Last Updated'].dt.year


# In[15]:


import plotly.express as px

# Define the path for HTML files
html_files_path = "./"

if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)

# Initialize plot_containers
plot_containers = ""

# Save each Plotly figure to an HTML file
def save_plot_as_html(fig, filename, insight, plot_width = 400, plot_height = 300):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    # Inject dynamic style for width and height
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')" style="width: {plot_width}px; height: {plot_height}px;">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath, full_html=False, include_plotlyjs='cdn')

# Define your plots
plot_width = 400
plot_height = 300
plot_bg_color = 'black'
text_color = 'white'
title_font = {'size': 16}
axis_font = {'size': 12}


# In[16]:


# Defining All Plots
# Category Analysis Plot
category_counts = apps_df['Category'].value_counts().nlargest(10)
fig1 = px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x': 'Category', 'y': 'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=plot_width,
    height=plot_height
)
fig1.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig1.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig1, "Category Analysis 1.html", "The top categories on the Play Store are dominated by tools, entertainment, and productivity apps. This suggests users are looking for apps that either provide utility or offer leisure activities.")

# Type Analysis Plot
type_counts = apps_df['Type'].value_counts()
fig2 = px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=plot_width,
    height=plot_height
)
fig2.update_traces(textposition='inside', textinfo='percent+label')
fig2.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig2, "Type Analysis 2.html", "Most apps on the Play Store are free, indicating a strategy to attract users first and monetize through ads or in-app purchases.")

# Rating Distribution Plot
fig3 = px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=plot_width,
    height=plot_height
)
fig3.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig3, "Rating Distribution 3.html", "Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users.")

sentiment_counts = reviews_df['Sentiment_Score'].value_counts()
fig4 = px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x': 'Sentiment Score', 'y': 'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=plot_width,
    height=plot_height
)
fig4.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig4.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig4, "Sentiment Distribution 4.html", "Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments.")

# Installs by Category Plot
installs_by_category = apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5 = px.bar(
    x=installs_by_category.values,
    y=installs_by_category.index,
    orientation='h',
    labels={'x': 'Installs', 'y': 'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=plot_width,
    height=plot_height
)
fig5.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig5.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig5, "Installs by Category 5.html", "The categories with the most installs are social and communication apps, which reflects their broad appeal and daily usage.")

# Updates Per Year Plot
updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig6, "Updates Per Year 6.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.")

# Revenue by Category Plot
revenue_by_category = apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7 = px.bar(
    x=revenue_by_category.index,
    y=revenue_by_category.values,
    labels={'x': 'Category', 'y': 'Revenue'},
    title='Revenue by Category',
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=plot_width,
    height=plot_height
)
fig7.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig7.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig7, "Revenue by Category 7.html", "Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential.")

# Genre Count Plot
genre_counts = apps_df['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)
fig8 = px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x': 'Genre', 'y': 'Count'},
    title='Top Genres',
    color=genre_counts.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=plot_width,
    height=plot_height
)
fig8.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig8.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig8, "Genres Counts 8.html", "Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games.")

# Impact of Last Update on Rating
fig9 = px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=plot_width,
    height=plot_height
)
fig9.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig9, "Update on Rating 9.html", "The scatter plot shows a weak correlation between the last update date and ratings, suggesting that more frequent updates don't always result in better ratings.")

# Ratings for Paid vs Free Apps
fig10 = px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Ratings for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=plot_width,
    height=plot_height
)
fig10.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig10, "Ratings paid vs free 10.html", "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for.")


# ## NULLCLASS INTERNSHIP TASKS (1-04-2025 to 1-05-2025)

# #### Task 1: 
# Visualize the sentiment distribution (positive, neutral, negative) of user reviews using a stacked bar chart, segmented by rating groups (e.g., 1-2 stars, 3-4 stars, 4-5 stars). Include only apps with more than 1,000 reviews and group by the top 5 categories.
# 

# In[17]:


# Data Preparation for Task 1
# Merge the two datasets on 'App' column
merged_data = pd.merge(apps_df, reviews_df, on='App')

# Filter apps with more than 1,000 reviews
filtered_data = merged_data[merged_data['Reviews'] > 1000]

# Select the top 5 categories based on the number of reviews
top_5_categories = filtered_data['Category'].value_counts().nlargest(5).index
filtered_data = filtered_data[filtered_data['Category'].isin(top_5_categories)]

# Group by Rating_Group, Category, and Sentiment, then count the reviews
grouped_data = filtered_data.groupby(['Rating_Group', 'Category', 'Sentiment']).size().reset_index(name='TotalReviews')

# Changing long category names or cleaner visibility
grouped_data['Category'] = grouped_data['Category'].replace({
    'HEALTH_AND_FITNESS': 'Health',
    'TRAVEL_AND_LOCAL': 'Travel',
    'PHOTOGRAPHY': 'Photo',
})


# In[18]:


# Creating the Stacked bar chart
plot_width = 800
plot_height = 300

fig11 = px.bar(
    grouped_data,
    x='Rating_Group',
    y='TotalReviews',
    color='Sentiment',
    facet_col = "Category",
    title='Sentiment Distribution of User Reviews (Top 5 Categories) by Rating Groups',
    labels={'TotalReviews': 'Total Reviews', 'Rating_Group': 'Rating Group'},
    barmode='stack',  # Stacked bar for each sentiment
    color_discrete_map={'Positive': 'green', 'Neutral': 'yellow', 'Negative': 'red'}  # Custom colors for sentiments
)

fig11.update_layout(
    width = 800,
    height = 300,
    xaxis_tickangle=-45,
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    title_x = 0.5,
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)

save_plot_as_html(fig11,"Stacked Bar Chart 11.html","Task 1", plot_width = 800, plot_height = 300)


# #### Task 2:
# Use a grouped bar chart to compare the average rating and total review count for the top 10 app categories by number of installs. Filter out any categories where the average rating is below 4.0 and size below 10 M and last update should be Jan month . this graph should work only between 3PM IST to 5 PM IST apart from that time we should not show this graph in dashboard itself.
# 

# In[19]:


# Data Preparation for Task 2
apps_df['Month'] = apps_df['Last Updated'].dt.month
# Filter the data based on the specified conditions
filtered_data = apps_df[
    (apps_df['Rating'] >= 4.0) &  # Average rating >= 4.0
    (apps_df['Installs'] >= 10000000) &  # Size >= 10M installs
    (apps_df['Month'] == 1)  # Last update in January
]

category_grouped = filtered_data.groupby('Category').agg(
    Average_Rating=('Rating', 'mean'),
    Total_Reviews=('Reviews', 'sum'),
    Total_Installs=('Installs', 'sum')
).reset_index()


# In[20]:


# Sort by installs and pick the top 10 categories
top_10_categories = category_grouped.sort_values(by='Total_Installs', ascending=False).head(10)


# In[21]:


# setting up the time for the plot to only show between 3 PM to 5 PM
from datetime import datetime, time

start_time = time(15, 0)
end_time = time(17, 0)

current_time = datetime.utcnow()
current_time = (current_time+pd.Timedelta(hours = 5, minutes = 30)).time()


# In[22]:


import plotly.graph_objects as go

plot_width = 800
plot_height = 300
# Ensure the graph only displays between 10 AM to 5 PM
if start_time <= current_time <= end_time:
    fig12 = go.Figure()

    # Bar for Total Reviews
    fig12.add_trace(
        go.Bar(x=top_10_categories['Category'], y=top_10_categories['Total_Reviews'], 
               name="Total Reviews", yaxis='y', marker_color='yellow')
    )

    # Line for Average Rating (scaled differently)
    fig12.add_trace(
        go.Scatter(x=top_10_categories['Category'], y=top_10_categories['Average_Rating'], 
                   name="Average Rating", yaxis='y2', marker=dict(color='cyan'), mode='lines+markers')
    )

    # Update layout for dual y-axes
    fig12.update_layout(
        width = 800,
        height = 300,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title="Average Rating and Total Review Count for Top 10 App Categories by Installs",
        xaxis=dict(title="Category"),
        yaxis=dict(title="Total Reviews", side='left'),
        yaxis2=dict(title="Average Rating", overlaying='y', side='right', range=[0, 5]),  # Scaling for rating (0-5)
        legend=dict(x=1.05, y=1, xanchor='left')
    )
    save_plot_as_html(fig12,"Grouped Bar Chart 12.html","Task 2", plot_width = 800, plot_height = 300)
else:
    print("The graph can only be displayed between 3 PM and 5 PM.")


# #### Task 3:
# Plot a time series line chart to show the trend of total installs over time, segmented by app category. Highlight periods of significant growth by shading the areas under the curve where the increase in installs exceeds 20% month-over-month and app name should not starts with x, y ,z and app category should start with letter " E " or " C " or " B " and reviews should be more than 500 as well as this graph should work only between 6 PM IST to 9 PM IST apart from that time we should not show this graph in dashboard itself.

# In[23]:


# Data Preparation for Task 3
filtered_data_3 = apps_df[(apps_df['Content Rating'] == 'Teen')]

filtered_data_3 = filtered_data_3[(filtered_data_3['App'].str.startswith('E'))]

filtered_data_3 = filtered_data_3[(filtered_data_3['Installs'] > 10000)]

filtered_data_3['YearMonth'] = filtered_data_3['Last Updated'].dt.to_period('M')

filtered_data_3


# In[24]:


installs = (
    filtered_data_3.groupby(['YearMonth', 'Category'])['Installs']
    .sum()
    .reset_index()
    .sort_values('YearMonth')
)


# In[25]:


installs['Pct_Change'] = (
    installs.groupby('Category')['Installs']
    .pct_change() * 100
)


# In[26]:


installs['Significant Growth'] = installs['Pct_Change'] > 20
installs


# In[27]:


#Before creating the chart, converting the YearMonth column to a string format first
installs['YearMonth'] = installs['YearMonth'].astype(str)
installs.info()


# In[28]:


print(installs['YearMonth'].unique())


# In[29]:


installs['YearMonth'] = pd.to_datetime(installs['YearMonth'], format='%Y-%m', errors='coerce')
installs.info()


# In[30]:


# Setting up the time for the Time Series Chart
current_time = datetime.now().time()
start_time = datetime.strptime("18:00", "%H:%M").time()
end_time = datetime.strptime("21:00", "%H:%M").time()


# In[31]:


if start_time <= current_time <= end_time:
    # Creating the line chart with Plotly
    fig13 = px.line(
        installs,
        x='YearMonth',
        y='Installs',
        color='Category',
        line_group='Category',
        title="Trend of Total Installs Over Time (Teen, Apps Starting with 'E')",
        labels={'YearMonth': 'Month-Year', 'Installs': 'Total Installs', 'Category': 'App Category'},
    )

    # Highlighting significant growth areas by adding filled areas
    for category in installs['Category'].unique():
        category_data = installs[(installs['Category'] == category) & (installs['Significant Growth'])]
        fig13.add_scatter(
            x=category_data['YearMonth'],
            y=category_data['Installs'],
            fill='tozeroy',
            mode='lines',
            name=f"Significant Growth: {category}",
            opacity=0.3
        )

    # Updating the layout for better visualization
    fig13.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="Total Installs",
        legend_title="App Categories",
    )
    fig13.update_layout(
        width = 800,
        height = 300,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title_font={'size':16},
        title_x = 0.5,
        xaxis=dict(title_font={'size':12}),
        yaxis=dict(title_font={'size':12}),
        margin=dict(l=10,r=10,t=30,b=10)
    )
    save_plot_as_html(fig13,"Time Series 13.html","Task 3", plot_width = 800, plot_height = 300)
else:
    print("Time Series Chart is not available outside the time range (6 PM - 9 PM IST).")


# In[32]:


# DASHBOARD DESIGNING
# Split plot_containers to handle the last plot properly
plot_containers_split = plot_containers.split('</div>')

if len(plot_containers_split) > 1:
    final_plot = plot_containers_split[-2] + '</div>'
else:
    final_plot = plot_containers  # Using plot_containers as default if splitting isn't sufficient

# HTML template for the dashboard
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Reviews Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container:hover .insights {{
            display: block;
        }}
        .notice {{
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            background-color: #555;
            border-radius: 10px;
            font-size: 16px;
        }}
        .notice p {{
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .notice ul {{
            list-style: none;
            padding: 0;
        }}
        .notice li {{
            margin: 5px 0;
        }}
    </style>
    <script>
        function openPlot(filename) {{
            window.open(filename, '_blank');
        }}
    </script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
    </div>
    <div class="container">
        {plots}
    </div>
    <div class="notice">
    <p>Note:</p>
    <ul>
        <li><strong>Task 2</strong> (Grouped Bar Chart) is available only between <strong>3 PM - 5 PM IST</strong>.</li>
        <li><strong>Task 3</strong> (Time Series Chart) is available only between <strong>6 PM - 9 PM IST</strong>.</li>
    </ul>
    </div>
</body>
</html>
"""

# Use these containers to fill in the dashboard HTML
final_html = dashboard_html.format(plots=plot_containers, plot_width=plot_width, plot_height=plot_height)

# Save the final dashboard to an HTML file
dashboard_path = os.path.join(html_files_path, "Final Dashboard.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)

# Automatically open the generated HTML file in a web browser
webbrowser.open('file://' + os.path.realpath(dashboard_path))

