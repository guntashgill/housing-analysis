import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

class PredictiveModels:
    def __init__(self, housing_df, job_df):
        self.housing_df = housing_df
        self.job_df = job_df
        
    def train_housing_models(self):
        """Train models to predict future housing prices"""
        # Prepare data for time series forecasting
        county_models = {}
        appreciation_models = {}
        
        # Train models for each county
        for county_id, group in tqdm(self.housing_df.groupby('CountyRegionID'), 
                                    desc="Training housing models"):
            # Time series model for price prediction
            price_data = group.set_index('Date')['MedianListingPrice']
            try:
                # Use a simpler model for stability
                model = ARIMA(price_data, order=(1,1,0))
                model_fit = model.fit()
                county_models[county_id] = model_fit
            except:
                # Fallback to simple average if ARIMA fails
                county_models[county_id] = None
            
            # Appreciation rate model
            features = ['MedianSquareFeet', 'MedianRentalPrice', 'ActiveListingCount']
            X = group[features].fillna(group[features].mean())
            y = group['AnnualAppreciation'].fillna(0)
            
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                rf.fit(X_train, y_train)
                appreciation_models[county_id] = rf
        
        return county_models, appreciation_models
    
    def train_job_market_models(self):
        """Train models to predict future job market conditions"""
        # Prepare job data
        job_geo = self.job_df.groupby(['City', 'Experience', 'Industry']).agg(
            avg_salary=('Salary', 'mean'),
            job_count=('Salary', 'count')
        ).reset_index()
        
        # Add historical trend data (simulated)
        years = [2018, 2019, 2020, 2021, 2022, 2023]
        full_data = []
        
        for (city, exp, industry), group in job_geo.groupby(['City', 'Experience', 'Industry']):
            base_salary = group['avg_salary'].values[0]
            base_count = group['job_count'].values[0]
            
            for year in years:
                growth = np.random.uniform(0.03, 0.08) * (year - 2018)
                salary = base_salary * (1 + growth)
                count = base_count * (1 + growth * 1.5)  # Job count grows faster
                
                full_data.append({
                    'City': city,
                    'Experience': exp,
                    'Industry': industry,
                    'Year': year,
                    'avg_salary': salary,
                    'job_count': count
                })
        
        trend_df = pd.DataFrame(full_data)
        
        # Train prediction models
        salary_models = {}
        job_count_models = {}
        
        for (exp, industry), group in trend_df.groupby(['Experience', 'Industry']):
            # Salary growth model
            X = group[['Year']]
            y = group['avg_salary']
            
            if len(X) > 3:
                rf_salary = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=3)
                rf_salary.fit(X, y)
                salary_models[(exp, industry)] = rf_salary
            
            # Job count model
            y_count = group['job_count']
            rf_count = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=3)
            rf_count.fit(X, y_count)
            job_count_models[(exp, industry)] = rf_count
        
        return salary_models, job_count_models, trend_df
    
    def predict_future_housing(self, county_id, future_date):
        """Predict housing prices for a future date"""
        model = self.county_models.get(county_id)
        if not model:
            # Fallback to simple growth model
            county_data = self.housing_df[self.housing_df['CountyRegionID'] == county_id]
            if len(county_data) > 1:
                growth_rate = county_data['AnnualAppreciation'].mean()
                current_price = county_data['MedianListingPrice'].iloc[-1]
                months = (future_date - county_data['Date'].max()).days // 30
                return current_price * (1 + growth_rate/12)**months, growth_rate
            return None, None
        
        # Predict price
        try:
            forecast = model.get_forecast(steps=(future_date - self.housing_df['Date'].max()).days // 30)
            pred_price = forecast.predicted_mean.iloc[-1]
        except:
            # Fallback if forecast fails
            pred_price = self.housing_df[self.housing_df['CountyRegionID'] == county_id]['MedianListingPrice'].mean()
        
        # Predict appreciation rate
        rf_model = self.appreciation_models.get(county_id)
        if rf_model:
            latest = self.housing_df[self.housing_df['CountyRegionID'] == county_id].iloc[-1]
            features = latest[['MedianSquareFeet', 'MedianRentalPrice', 'ActiveListingCount']].fillna(0).values.reshape(1, -1)
            appreciation_rate = rf_model.predict(features)[0]
            return pred_price, appreciation_rate
        
        return pred_price, 0.05  # Default 5% appreciation
    
    def predict_future_jobs(self, city, experience, industry, year):
        """Predict job market conditions for future year"""
        # Predict salary
        salary_model = self.salary_models.get((experience, industry))
        job_count_model = self.job_count_models.get((experience, industry))
        
        if not salary_model or not job_count_model:
            return None, None
        
        salary_pred = salary_model.predict([[year]])[0]
        job_count_pred = job_count_model.predict([[year]])[0]
        
        return salary_pred, job_count_pred