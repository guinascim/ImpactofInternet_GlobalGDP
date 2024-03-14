import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
import plotly.express as px

# Load CSV files
df1 = pd.read_csv('gdp.csv')
df2 = pd.read_csv('users.csv')

# Title
st.title('Global GDP and Internet Users Analysis')

# Sidebar - Add interactive elements
selected_analysis = st.sidebar.selectbox('Select Analysis', ['Overview', 'GDP and Users Predictions', 'Correlation Analysis', 'Heatmap', 'Regression Models','Growth Prediction - GDP only'])

if selected_analysis == 'GDP and Users Predictions':
    # Plot GDP and Users Predictions
    st.subheader('GDP and Users Predictions')
    
    fig = go.Figure()

    all_countries_checkbox = st.sidebar.checkbox('Select All Countries')
    if all_countries_checkbox:
        selected_countries = df1['Country'].unique()
    else:
        selected_countries = st.sidebar.multiselect('Select Countries', df1['Country'].unique())

    for country in selected_countries:
        gdp_data = df1[df1['Country'] == country].drop(columns=['Country']).values.flatten()
        years = np.arange(2010, 2010+len(gdp_data))

        users_data = df2[df2['Country'] == country].drop(columns=['Country']).values.flatten()

        nan_indices_gdp = pd.isnull(gdp_data)
        nan_indices_users = pd.isnull(users_data)

        if nan_indices_gdp.all() or nan_indices_users.all():
            continue

        nan_indices_combined = nan_indices_gdp | nan_indices_users
        gdp_data = gdp_data[~nan_indices_combined]
        users_data = users_data[~nan_indices_combined]
        years = years[~nan_indices_combined]

        assert gdp_data.shape == users_data.shape == years.shape, f"GDP, Users, and Years data shapes don't match for {country}"

        gdp_model = LinearRegression()
        gdp_model.fit(years.reshape(-1, 1), gdp_data)

        users_model = LinearRegression()
        users_model.fit(years.reshape(-1, 1), users_data)

        future_years = np.arange(2021, 2031)
        future_gdp = gdp_model.predict(future_years.reshape(-1, 1))
        future_users = users_model.predict(future_years.reshape(-1, 1))

        fig.add_trace(go.Scatter(x=years, y=gdp_data, mode='lines', name=f'{country} - GDP'))
        fig.add_trace(go.Scatter(x=future_years, y=future_gdp, mode='lines', line=dict(dash='dash'), name=f'{country} - Predicted GDP'))

        fig.add_trace(go.Scatter(x=years, y=users_data, mode='lines', name=f'{country} - Users'))
        fig.add_trace(go.Scatter(x=future_years, y=future_users, mode='lines', line=dict(dash='dash'), name=f'{country} - Predicted Users'))

    fig.update_layout(title='GDP and Users Predictions', xaxis_title='Year', yaxis_title='Value', 
    xaxis=dict(tickmode='linear'), legend=dict(orientation="h", yanchor="bottom", y=-1.15, xanchor="center", x=0.5))

    st.plotly_chart(fig)

elif selected_analysis == 'Correlation Analysis':
    # Plot Correlation Analysis
    st.subheader('Correlation Analysis')
    merged_df = pd.merge(df1, df2, on='Country', suffixes=('_GDP', '_Users'))

    gdp_data = merged_df.iloc[:, 1:12].values.flatten()
    users_data = merged_df.iloc[:, 12:].values.flatten()

    correlation_coefficient = np.corrcoef(gdp_data, users_data)[0, 1]

    plt.scatter(users_data, gdp_data, alpha=0.5)
    plt.title('Correlation between GDP and Users')
    plt.xlabel('Users')
    plt.ylabel('GDP')
    plt.text(0.1, 0.9, f'Correlation coefficient: {correlation_coefficient:.2f}', transform=plt.gca().transAxes)
    st.pyplot(plt)

    explanation = """
    **Correlation Analysis: GDP vs. Users**

    This chart visualizes the correlation between Gross Domestic Product (GDP) and Internet Users across different countries. Each point on the scatter plot represents a country's GDP and the corresponding number of internet users. The correlation coefficient, which quantifies the strength and direction of the relationship between GDP and users, is displayed on the chart. A positive correlation coefficient indicates a direct relationship, while a negative coefficient implies an inverse relationship. A value close to 1 or -1 suggests a strong correlation, while a value close to 0 indicates a weak correlation.
    """
    st.write(explanation)

elif selected_analysis == 'Heatmap':
    # Plot Heatmap
    st.subheader('Heatmap')

    merged_df = pd.merge(df1, df2, on='Country', suffixes=('_GDP', '_Users'))  # Define merged_df here

    # Drop non-numeric columns like 'Country'
    merged_df_numeric = merged_df.drop(columns=['Country'])

    correlation_matrix = merged_df_numeric.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap between GDP and Users')
    plt.tight_layout()  # Adjust subplot parameters
    st.pyplot(plt)
    
elif selected_analysis == 'Regression Models':
    # Regression Models
    st.subheader('Regression Models')
    # Add code for regression models analysis here

    # Random Forest Regression
    st.subheader('RandomForestRegressor')
    # Add code for RandomForestRegressor analysis here
    # Initialize lists to store results
    error_margin_list = []
    accuracy_list = []
    growth_2010_to_2020_list = []
    growth_prediction_2021_to_2031_list = []
    mse_list = []

    for country in df1['Country'].unique():
        # Prepare data for the specific country
        gdp_data = df1[df1['Country'] == country].drop(columns=['Country']).values.flatten()
        users_data = df2[df2['Country'] == country].drop(columns=['Country']).values.flatten()

        # Check if both GDP and users data exist for the country
        if len(gdp_data) == 0 or len(users_data) == 0:
            continue

        years = np.arange(2010, 2010 + len(gdp_data))

        # Remove NaN values
        nan_indices_gdp = pd.isnull(gdp_data)
        nan_indices_users = pd.isnull(users_data)
        nan_indices_combined = nan_indices_gdp | nan_indices_users
        gdp_data = gdp_data[~nan_indices_combined]
        users_data = users_data[~nan_indices_combined]
        years = years[~nan_indices_combined]

        # Check if there are enough samples for fitting the model
        if len(gdp_data) < 2 or len(users_data) < 2:
            continue

        # Random Forest Regression for GDP
        gdp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gdp_model.fit(years.reshape(-1, 1), gdp_data)

        # Predictions for the next ten years (2021-2031)
        future_years = np.arange(2021, 2031)
        future_gdp = gdp_model.predict(future_years.reshape(-1, 1))

        # Calculate error margin and accuracy
        predictions = gdp_model.predict(years.reshape(-1, 1))
        mse = mean_squared_error(gdp_data, predictions)
        rmse = np.sqrt(mse)
        accuracy = 100 - (rmse / np.mean(gdp_data) * 100)

        # Append results to lists
        error_margin_list.append(rmse)
        accuracy_list.append(accuracy)
        mse_list.append(mse)

        # Calculate growth percentage from 2010 to 2020
        growth_percentage_2010_to_2020 = ((gdp_data[-1] - gdp_data[0]) / gdp_data[0]) * 100
        growth_2010_to_2020_list.append(growth_percentage_2010_to_2020)

        # Calculate growth prediction from 2021 to 2031
        growth_prediction_2021_to_2031_list.append((future_gdp[-1] / future_gdp[0]) * 100)

    # Compute total values
    total_error_margin = np.mean(error_margin_list)
    total_accuracy = np.mean(accuracy_list)
    total_growth_2010_to_2020 = np.mean(growth_2010_to_2020_list)
    total_growth_prediction_2021_to_2031 = np.mean(growth_prediction_2021_to_2031_list)
    total_mse = np.mean(mse_list)

    # Print the results
    st.write("Total Error Margin (RMSE):", total_error_margin)
    st.write("Total Accuracy (%):", total_accuracy)
    st.write("Total Growth Percentage 2010-2020:", total_growth_2010_to_2020)
    st.write("Total Growth Prediction 2021-2031:", total_growth_prediction_2021_to_2031)
    st.write("Total Mean Squared Error (MSE):", total_mse)

    # Gradient Boosting Regressor
    st.subheader('Gradient Boosting Regressor')

    error_margin_list = []
    accuracy_list = []
    growth_2010_to_2020_list = []
    growth_prediction_2021_to_2031_list = []
    mse_list = []  # Store MSE for each country
    r2_list = []   # Store R-squared for each country

    for country in df1['Country'].unique():
        # Prepare data for the specific country
        gdp_data = df1[df1['Country'] == country].drop(columns=['Country']).values.flatten()
        users_data = df2[df2['Country'] == country].drop(columns=['Country']).values.flatten()

        # Check if both GDP and users data exist for the country
        if len(gdp_data) == 0 or len(users_data) == 0:
            continue

        years = np.arange(2010, 2010 + len(gdp_data))

        # Remove NaN values
        nan_indices_gdp = pd.isnull(gdp_data)
        nan_indices_users = pd.isnull(users_data)
        nan_indices_combined = nan_indices_gdp | nan_indices_users
        gdp_data = gdp_data[~nan_indices_combined]
        users_data = users_data[~nan_indices_combined]
        years = years[~nan_indices_combined]

        # Check if there are enough samples for fitting the model
        if len(gdp_data) < 2 or len(users_data) < 2:
            continue

        # Gradient Boosting Regression for GDP
        gdp_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gdp_model.fit(years.reshape(-1, 1), gdp_data)

        # Predictions for the next ten years (2021-2031)
        future_years = np.arange(2021, 2031)
        future_gdp = gdp_model.predict(future_years.reshape(-1, 1))

        # Calculate error margin and accuracy
        predictions = gdp_model.predict(years.reshape(-1, 1))
        mse = mean_squared_error(gdp_data, predictions)
        rmse = np.sqrt(mse)
        accuracy = 100 - (rmse / np.mean(gdp_data) * 100)

        # Append results to lists
        error_margin_list.append(rmse)
        accuracy_list.append(accuracy)

        # Calculate growth percentage from 2010 to 2020
        growth_percentage_2010_to_2020 = ((gdp_data[-1] - gdp_data[0]) / gdp_data[0]) * 100
        growth_2010_to_2020_list.append(growth_percentage_2010_to_2020)

        # Calculate growth prediction from 2021 to 2031
        growth_prediction_2021_to_2031_list.append((future_gdp[-1] / future_gdp[0]) * 100)

        # Calculate R-squared
        r2 = r2_score(gdp_data, predictions)
        r2_list.append(r2)
        mse_list.append(mse)

    # Compute total values
    total_error_margin = np.mean(error_margin_list)
    total_accuracy = np.mean(accuracy_list)
    total_growth_2010_to_2020 = np.mean(growth_2010_to_2020_list)
    total_growth_prediction_2021_to_2031 = np.mean(growth_prediction_2021_to_2031_list)
    total_mse = np.mean(mse_list)
    total_r2 = np.mean(r2_list)

    # Print the results
    st.write("Total Error Margin (RMSE):", total_error_margin)
    st.write("Total Accuracy (%):", total_accuracy)
    st.write("Total Growth Percentage 2010-2020:", total_growth_2010_to_2020)
    st.write("Total Growth Prediction 2021-2031:", total_growth_prediction_2021_to_2031)
    st.write("Total Mean Squared Error (MSE):", total_mse)
    st.write("Total R-squared (R2):", total_r2)

elif selected_analysis == 'Growth Prediction - GDP only':
    # Plot Heatmap
    st.subheader('Growth Prediction')
    # Reshape DataFrame from wide to long format
    melted_data = df1.melt(id_vars=['Country'], var_name='Year', value_name='GDP')
    merged_df = pd.merge(df1, df2, on='Country', suffixes=('_GDP', '_Users'))

    melted_data['Year'] = pd.to_numeric(melted_data['Year'])

    # Filter data for years 2010 to 2020
    filtered_data = melted_data[(melted_data['Year'] >= 2010) & (melted_data['Year'] <= 2020)]

    # Calculate total GDP for each year
    total_gdp_by_year = filtered_data.groupby('Year').agg({'GDP': 'sum'})

    # Calculate GDP growth from 2010 to 2020
    initial_gdp = total_gdp_by_year.loc[2010, 'GDP']
    final_gdp = total_gdp_by_year.loc[2020, 'GDP']
    total_growth = ((final_gdp - initial_gdp) / initial_gdp) * 100

    # Train a simple linear regression model to predict total GDP growth for the next ten years
    X_train = total_gdp_by_year.index.values.reshape(-1, 1)
    y_train = total_gdp_by_year['GDP'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict total GDP for the next ten years
    future_years = [year for year in range(2021, 2031)]
    X_future = pd.DataFrame({'Year': future_years})
    predicted_gdp = model.predict(X_future)

    # Display total GDP growth from 2010 to 2020 and predicted total GDP for the next ten years
    st.write("Total GDP growth from 2010 to 2020: {:.2f}%".format(total_growth))

    st.write("\nPredicted total GDP for the next ten years:")
    predicted_data = pd.DataFrame({'Year': future_years, 'Predicted GDP': predicted_gdp})
    st.write(predicted_data)

    st.write(f"The prediction for the next 10 years is of:", predicted_data["Predicted GDP"].sum())

    # Calculate the total predicted GDP growth for the next 10 years
    predicted_growth_percentage = ((predicted_gdp[-1] - predicted_gdp[0]) / predicted_gdp[0]) * 100
    rounded_predicted_growth_percentage = round(predicted_growth_percentage, 2)

    # Display the total predicted GDP growth for the next 10 years as a rounded percentage
    st.write(f"The predicted total GDP growth for the next 10 years is: {rounded_predicted_growth_percentage}%")

    # assuming cleaned_gdp2 is a DataFrame with columns: 'Country', '2010', '2011', ..., '2023'
    # You need to have the cleaned_gdp2 DataFrame loaded or defined properly before running this code

    # Reshape DataFrame from wide to long format
    melted_data = df1.melt(id_vars=['Country'], var_name='Year', value_name='GDP')

    # Convert 'Year' column to numeric
    melted_data['Year'] = pd.to_numeric(melted_data['Year'])

    # Filter data for years 2010 to 2023
    filtered_data = melted_data[(melted_data['Year'] >= 2010) & (melted_data['Year'] <= 2023)]

    # Calculate total GDP for each year
    total_gdp_by_year = filtered_data.groupby('Year').agg({'GDP': 'sum'})

    # Train a simple linear regression model to predict total GDP growth for the next seven years (2024 to 2030)
    X_train = total_gdp_by_year.index.values.reshape(-1, 1)
    y_train = total_gdp_by_year['GDP'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict total GDP for the next seven years (2024 to 2030)
    future_years = [year for year in range(2024, 2031)]
    X_future = pd.DataFrame({'Year': future_years})
    predicted_gdp = model.predict(X_future)

    # Display predicted total GDP for the next seven years
    predicted_data = pd.DataFrame({'Year': future_years, 'Predicted GDP': predicted_gdp})
    st.write("Predicted total GDP for the years 2024 to 2030:")
    st.write(predicted_data)

    # Calculate the total predicted GDP growth for the next 10 years
    predicted_growth_percentage = ((predicted_gdp[-1] - predicted_gdp[0]) / predicted_gdp[0]) * 100
    rounded_predicted_growth_percentage = round(predicted_growth_percentage, 2)

    # Display the total predicted GDP growth for the next 6 years as a rounded percentage
    st.write(f"The predicted total GDP growth for the next 6 years is: {rounded_predicted_growth_percentage}%")

    total_gdp_2010 = merged_df['2010_GDP'].sum()
    total_gdp_2020 = merged_df['2020_GDP'].sum()
    percentage_growth = ((total_gdp_2020 - total_gdp_2010) / total_gdp_2010) * 100

    st.write(f"The estimated percentage growth in total GDP from 2010 to 2020 is: {percentage_growth:.2f}%")
    
elif selected_analysis == 'Overview':
    # Plot Correlation Analysis
    st.subheader('Countries by users and GDP')

# Define the URL of the Tableau visualization hosted on Tableau Public
twbx_url = "https://public.tableau.com/views/usersbycountry/Dashboard1?:language=pt-BR&:display_count=n&:origin=viz_share_link"

# Generate HTML code to embed the Tableau visualization
html_code = f"""
<div class='tableauPlaceholder' id='viz_placeholder' style='position: relative'>
    <noscript><a href='#'><img alt='Dashboard 1' src='https://public.tableau.com/static/images/us/usersbycountry/Dashboard1/1_rss.png' style='border: none' /></a></noscript>
    <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='usersbycountry/Dashboard1' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/us/usersbycountry/Dashboard1/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='pt-BR' />
    </object>
</div>
<script type='text/javascript'>
    var placeholderDiv = document.getElementById('viz_placeholder');
    var url = '{twbx_url}';
    var options = {{
        hideTabs: true,
        onFirstInteractive: function () {{
            console.log('Tableau visualization is loaded.');
        }}
    }};
    var viz = new tableau.Viz(placeholderDiv, url, options);
</script>
"""

# Display the HTML code
st.markdown(html_code, unsafe_allow_html=True)
