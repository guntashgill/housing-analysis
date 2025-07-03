import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import zipfile
import shutil
import time

# Set professional style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# ======================
# KAGGLE DATA DOWNLOADER
# ======================

class KaggleDataLoader:
    @staticmethod
    def download_dataset(handle, filename=None):
        """Download dataset from Kaggle Hub and return path"""
        try:
            print(f"Downloading dataset: {handle}")
            path = kagglehub.dataset_download(handle)
            print(f"Dataset downloaded to: {path}")
            return path
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return None

    @staticmethod
    def get_file_path(handle, filename):
        """Get path to specific file in downloaded dataset"""
        base_path = KaggleDataLoader.download_dataset(handle)
        if not base_path:
            return None
            
        # Search for file in directory
        for root, dirs, files in os.walk(base_path):
            if filename in files:
                return os.path.join(root, filename)
        
        print(f"File {filename} not found in {base_path}")
        return None

# ======================
# DATA LOADING & PREPROCESSING
# ======================

class DataLoader:
    @staticmethod
    def load_housing_data():
        """Load and preprocess Zillow housing data from Kaggle"""
        try:
            # Download dataset
            file_path = KaggleDataLoader.get_file_path(
                "zillow/zecon", 
                "County_time_series.csv"
            )
            
            if not file_path:
                raise FileNotFoundError("Housing data not available")
            
            # Load core housing data
            df = pd.read_csv(file_path)
            
            # Load metadata
            meta_path = KaggleDataLoader.get_file_path(
                "zillow/zecon", 
                "County_crosswalk.csv"
            )
            if meta_path:
                meta = pd.read_csv(meta_path)
                # Merge with main dataset
                df = df.merge(meta[['CountyRegionID', 'CountyName', 'StateName']], 
                              on='CountyRegionID', how='left')
            
            # Convert to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Check for required columns
            required_columns = ['MedianListingPrice', 'MedianSquareFeet', 'MedianRentalPrice']
            for col in required_columns:
                if col not in df.columns:
                    # Try to find similar columns
                    matching_cols = [c for c in df.columns if col.lower() in c.lower()]
                    if matching_cols:
                        df.rename(columns={matching_cols[0]: col}, inplace=True)
                    else:
                        print(f"Column {col} not found, using placeholder values")
                        df[col] = np.nan
            
            # Feature engineering
            df['PricePerSqft'] = df['MedianListingPrice'] / df['MedianSquareFeet']
            df['RentRatio'] = df['MedianRentalPrice'] / df['MedianListingPrice']
            df['AnnualAppreciation'] = df.groupby('CountyRegionID')['MedianListingPrice'].pct_change() * 12
            
            return df.dropna(subset=['MedianListingPrice'])
        except Exception as e:
            print(f"Error loading housing data: {str(e)}")
            print("Using generated housing data")
            return DataLoader.generate_housing_data()

    @staticmethod
    def generate_housing_data():
        """Generate realistic housing data for demonstration"""
        counties = ['Santa Clara', 'San Mateo', 'Alameda', 'Contra Costa', 'San Francisco']
        states = ['CA'] * 5
        dates = pd.date_range('2010-01-01', '2023-01-01', freq='MS')
        
        data = []
        for i, county in enumerate(counties):
            base_price = np.random.uniform(300000, 800000)
            for date in dates:
                # Appreciation model
                years = (date - pd.Timestamp('2010-01-01')).days / 365
                appreciation = 0.05 * years + 0.02 * np.sin(years)
                price = base_price * (1 + appreciation) * np.random.uniform(0.98, 1.02)
                
                # Related metrics
                sqft = np.random.uniform(1200, 2500)
                rent = price * 0.0045 * np.random.uniform(0.9, 1.1)
                
                data.append({
                    'CountyRegionID': i+1,
                    'CountyName': county,
                    'StateName': states[i],
                    'Date': date,
                    'MedianListingPrice': price,
                    'MedianSquareFeet': sqft,
                    'MedianRentalPrice': rent,
                    'ActiveListingCount': np.random.randint(50, 500)
                })
        
        df = pd.DataFrame(data)
        df['PricePerSqft'] = df['MedianListingPrice'] / df['MedianSquareFeet']
        df['RentRatio'] = df['MedianRentalPrice'] / df['MedianListingPrice']
        df['AnnualAppreciation'] = df.groupby('CountyRegionID')['MedianListingPrice'].pct_change() * 12
        return df.dropna()

    @staticmethod
    def load_job_data():
        """Load and preprocess job data from Kaggle"""
        try:
            # Download dataset
            file_path = KaggleDataLoader.get_file_path(
                "PromptCloudHQ/us-jobs-on-monstercom", 
                "monster_com-job_sample.csv"
            )
            
            if not file_path:
                raise FileNotFoundError("Job data not available")
            
            # Try different encodings if necessary
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except:
                df = pd.read_csv(file_path, encoding='latin1')
            
            # Create consistent column names
            col_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'title' in col_lower:
                    col_map[col] = 'job_title'
                elif 'company' in col_lower:
                    col_map[col] = 'company'
                elif 'location' in col_lower:
                    col_map[col] = 'location'
                elif 'description' in col_lower:
                    col_map[col] = 'job_description'
                elif 'experience' in col_lower:
                    col_map[col] = 'job_experience'
            
            df = df.rename(columns=col_map)
            
            # Ensure we have required columns
            required_columns = ['job_title', 'company', 'location', 'job_description', 'job_experience']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Column {col} not found, creating placeholder")
                    df[col] = np.nan
            
            # Clean and transform
            df = df[required_columns]
            df = df.dropna(subset=['location', 'job_experience'])
            
            # Extract state and city
            df['State'] = df['location'].apply(
                lambda x: x.split(',')[-1].strip() if isinstance(x, str) and ',' in x else 'Unknown'
            )
            df['City'] = df['location'].apply(
                lambda x: x.split(',')[0].strip() if isinstance(x, str) and ',' in x else 'Unknown'
            )
            
            # Extract salary information
            def extract_salary(desc):
                if pd.isna(desc) or not isinstance(desc, str):
                    return np.nan
                if '$' in desc:
                    # Find all numbers with dollar signs
                    nums = []
                    for word in desc.split():
                        clean_word = word.replace(',', '').replace('$', '')
                        if clean_word.replace('.', '').isdigit():
                            nums.append(float(clean_word))
                    if nums:
                        return np.mean(nums)
                return np.nan
            
            df['Salary'] = df['job_description'].apply(extract_salary)
            
            # If no salaries found, use industry averages
            if df['Salary'].isna().all():
                print("No salaries extracted, using industry averages")
                industry_salaries = {
                    'Technology': 120000,
                    'Healthcare': 90000,
                    'Finance': 110000,
                    'Education': 70000,
                    'Manufacturing': 80000
                }
                # Add industry classification
                industries = list(industry_salaries.keys())
                df['Industry'] = np.random.choice(industries, size=len(df))
                df['Salary'] = df['Industry'].map(industry_salaries)
            else:
                # Add industry classification
                industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing']
                df['Industry'] = np.random.choice(industries, size=len(df))
                # Fill missing salaries with industry averages
                industry_avg = df.groupby('Industry')['Salary'].transform('mean')
                df['Salary'] = df['Salary'].fillna(industry_avg)
            
            # Categorize experience
            def map_experience(exp):
                if pd.isna(exp):
                    return 'Not Specified'
                exp = str(exp).lower()
                if 'entry' in exp or 'junior' in exp or '0-2' in exp:
                    return 'Entry-level'
                elif 'senior' in exp or 'lead' in exp or '8+' in exp:
                    return 'Senior'
                elif 'mid' in exp or '3-5' in exp or '5-7' in exp:
                    return 'Mid-level'
                return 'Not Specified'
            
            df['Experience'] = df['job_experience'].apply(map_experience)
            df = df[df['Experience'] != 'Not Specified']
            
            return df
        except Exception as e:
            print(f"Error loading job data: {str(e)}")
            print("Using generated job data")
            return DataLoader.generate_job_data()

    @staticmethod
    def generate_job_data():
        """Generate realistic job data for demonstration"""
        cities = ['San Jose', 'Palo Alto', 'Mountain View', 'Sunnyvale', 'San Francisco']
        states = ['CA'] * 5
        titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 
                 'UX Designer', 'DevOps Engineer', 'Marketing Specialist']
        companies = ['TechCorp', 'DataSystems', 'InnovateCo', 'FutureTech', 'DigitalSolutions']
        
        data = []
        for _ in range(5000):
            city = np.random.choice(cities)
            state = 'CA'
            title = np.random.choice(titles)
            company = np.random.choice(companies)
            exp_level = np.random.choice(['Entry-level', 'Mid-level', 'Senior'], p=[0.3, 0.5, 0.2])
            industry = np.random.choice(['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing'])
            
            # Base salaries
            base_salaries = {
                'Software Engineer': {'Entry-level': 90000, 'Mid-level': 130000, 'Senior': 180000},
                'Data Scientist': {'Entry-level': 95000, 'Mid-level': 135000, 'Senior': 190000},
                'Product Manager': {'Entry-level': 85000, 'Mid-level': 125000, 'Senior': 170000},
                'UX Designer': {'Entry-level': 80000, 'Mid-level': 110000, 'Senior': 150000},
                'DevOps Engineer': {'Entry-level': 100000, 'Mid-level': 140000, 'Senior': 190000},
                'Marketing Specialist': {'Entry-level': 60000, 'Mid-level': 85000, 'Senior': 120000}
            }
            
            salary = base_salaries[title][exp_level] * np.random.uniform(0.9, 1.15)
            
            data.append({
                'job_title': f"{exp_level} {title}",
                'company': company,
                'location': f"{city}, {state}",
                'Salary': salary,
                'Experience': exp_level,
                'Industry': industry,
                'City': city,
                'State': state
            })
        
        return pd.DataFrame(data)

# ======================
# PREDICTIVE MODELING
# ======================

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

# ======================
# AFFORDABILITY ANALYSIS
# ======================

class AffordabilityAnalyzer:
    @staticmethod
    def calculate_affordability(salary, home_price, down_payment=0.2, debt_ratio=0.36):
        """
        Calculate housing affordability based on salary and home price
        Returns affordability score (0-100) and mortgage details
        """
        # Mortgage parameters
        interest_rate = 0.065  # Average mortgage rate
        loan_term = 30  # years
        
        # Calculate loan amount
        loan_amount = home_price * (1 - down_payment)
        
        # Monthly mortgage payment
        monthly_interest = interest_rate / 12
        num_payments = loan_term * 12
        mortgage_payment = loan_amount * (monthly_interest * (1 + monthly_interest)**num_payments) / \
                          ((1 + monthly_interest)**num_payments - 1)
        
        # Annual housing cost
        annual_housing_cost = mortgage_payment * 12
        
        # Property taxes and insurance (estimated)
        annual_taxes_insurance = home_price * 0.015
        
        # Total annual cost
        total_annual_cost = annual_housing_cost + annual_taxes_insurance
        
        # Affordability (should be <= debt_ratio of salary)
        housing_ratio = total_annual_cost / salary
        
        # Affordability score (0-100)
        score = max(0, min(100, (1 - housing_ratio / debt_ratio) * 100))
        
        return {
            'affordability_score': score,
            'max_affordable_price': salary * 4,  # Simplified rule of thumb
            'mortgage_payment': mortgage_payment,
            'annual_housing_cost': total_annual_cost,
            'housing_ratio': housing_ratio,
            'debt_ratio_limit': debt_ratio
        }

# ======================
# RECOMMENDATION ENGINE
# ======================

class LocationRecommender:
    def __init__(self, housing_df, job_df, housing_models, job_models):
        self.housing_df = housing_df
        self.job_df = job_df
        self.housing_models = housing_models
        self.job_models = job_models
        self.county_coords = self._create_county_coordinates()
        
    def _create_county_coordinates(self):
        """Generate county coordinates for distance calculation"""
        counties = self.housing_df['CountyName'].unique()
        return {county: (np.random.uniform(-122, -121), np.random.uniform(37, 38)) 
                for county in counties}
    
    def calculate_commute(self, county1, county2):
        """Calculate approximate commute time between counties"""
        coord1 = self.county_coords.get(county1, (0, 0))
        coord2 = self.county_coords.get(county2, (0, 0))
        
        # Simple distance calculation (1° ≈ 69 miles)
        distance = np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2) * 69
        commute_minutes = distance * 1.5  # Approx 1.5 min per mile
        
        return min(120, max(5, commute_minutes))  # Cap commute at 2 hours
    
    def recommend_locations(self, job_field, experience_level, future_year=2025, max_commute=60):
        """Generate personalized location recommendations"""
        # Step 1: Predict future job market
        job_cities = self.job_df['City'].unique()
        job_predictions = []
        
        for city in job_cities:
            salary_pred, job_count_pred = self.job_models.predict_future_jobs(
                city, experience_level, job_field, future_year
            )
            if salary_pred and job_count_pred:
                job_predictions.append({
                    'City': city,
                    'PredictedSalary': salary_pred,
                    'PredictedJobCount': job_count_pred
                })
        
        job_pred_df = pd.DataFrame(job_predictions)
        
        # Step 2: Predict future housing prices
        housing_predictions = []
        for county_id, group in self.housing_df.groupby('CountyRegionID'):
            county_name = group['CountyName'].iloc[0]
            state = group['StateName'].iloc[0]
            
            # Predict to future year
            future_date = pd.Timestamp(f'{future_year}-01-01')
            price_pred, appreciation_rate = self.housing_models.predict_future_housing(
                county_id, future_date
            )
            
            if price_pred:
                housing_predictions.append({
                    'CountyName': county_name,
                    'StateName': state,
                    'PredictedPrice': price_pred,
                    'AppreciationRate': appreciation_rate
                })
        
        housing_pred_df = pd.DataFrame(housing_predictions)
        
        # Step 3: Create recommendations
        recommendations = []
        for _, job_row in job_pred_df.iterrows():
            job_city = job_row['City']
            for _, house_row in housing_pred_df.iterrows():
                county = house_row['CountyName']
                
                # Calculate commute
                commute = self.calculate_commute(job_city, county)
                if commute > max_commute:
                    continue
                
                # Calculate affordability
                affordability = AffordabilityAnalyzer.calculate_affordability(
                    job_row['PredictedSalary'],
                    house_row['PredictedPrice']
                )
                
                # Calculate investment potential
                investment_score = house_row['AppreciationRate'] * 100 + np.random.uniform(0, 20)
                
                recommendations.append({
                    'JobCity': job_city,
                    'HomeCounty': county,
                    'State': house_row['StateName'],
                    'CommuteMinutes': commute,
                    'PredictedSalary': job_row['PredictedSalary'],
                    'PredictedHomePrice': house_row['PredictedPrice'],
                    'AffordabilityScore': affordability['affordability_score'],
                    'InvestmentPotential': investment_score,
                    'JobOpportunities': job_row['PredictedJobCount'],
                    'AppreciationRate': house_row['AppreciationRate']
                })
        
        # Sort and rank recommendations
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            rec_df['OverallScore'] = (
                0.4 * rec_df['AffordabilityScore'] + 
                0.3 * rec_df['InvestmentPotential'] + 
                0.2 * (100 - rec_df['CommuteMinutes']/max_commute*100) +
                0.1 * (rec_df['JobOpportunities'] / rec_df['JobOpportunities'].max() * 100)
            )
            
            return rec_df.sort_values('OverallScore', ascending=False).head(10)
        
        return pd.DataFrame()

# ======================
# VISUALIZATION
# ======================

class Visualizer:
    @staticmethod
    def plot_historical_trends(housing_df, job_df):
        """Plot historical housing and job trends"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Housing price trends
        if housing_df is not None and not housing_df.empty:
            try:
                if 'Date' in housing_df and 'CountyName' in housing_df and 'MedianListingPrice' in housing_df:
                    housing_agg = housing_df.groupby(['Date', 'CountyName'])['MedianListingPrice'].mean().unstack()
                    housing_agg.plot(ax=ax1, lw=2)
                    ax1.set_title('Historical Housing Price Trends by County', fontsize=16)
                    ax1.set_ylabel('Median Listing Price ($)')
                    ax1.legend(title='County')
                    ax1.grid(True, linestyle='--', alpha=0.7)
                else:
                    ax1.text(0.5, 0.5, 'Missing housing data columns', 
                             ha='center', va='center', transform=ax1.transAxes)
            except Exception as e:
                print(f"Error plotting housing trends: {e}")
                ax1.text(0.5, 0.5, 'Error plotting housing trends', 
                         ha='center', va='center', transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, 'No housing data available', 
                     ha='center', va='center', transform=ax1.transAxes)
        
        # Job market trends
        if job_df is not None and not job_df.empty:
            try:
                if 'Experience' in job_df and 'Industry' in job_df and 'Salary' in job_df:
                    # Ensure Salary is numeric
                    job_df['Salary'] = pd.to_numeric(job_df['Salary'], errors='coerce')
                    if job_df['Salary'].isna().all():
                        raise ValueError("Salary data is not numeric")
                    
                    job_agg = job_df.groupby(['Experience', 'Industry'])['Salary'].mean().unstack()
                    
                    if not job_agg.empty:
                        job_agg.T.plot(kind='bar', ax=ax2, rot=0)
                        ax2.set_title('Average Salaries by Experience Level and Industry', fontsize=16)
                        ax2.set_ylabel('Salary ($)')
                        ax2.grid(axis='y', linestyle='--', alpha=0.7)
                    else:
                        ax2.text(0.5, 0.5, 'No job data to visualize', 
                                 ha='center', va='center', transform=ax2.transAxes)
                else:
                    ax2.text(0.5, 0.5, 'Missing job data columns', 
                             ha='center', va='center', transform=ax2.transAxes)
            except Exception as e:
                print(f"Error plotting job trends: {e}")
                ax2.text(0.5, 0.5, 'Error plotting job trends', 
                         ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'No job data available', 
                     ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig('historical_trends.png', dpi=300)
        plt.show()
    
    @staticmethod
    def plot_recommendations(recommendations):
        """Visualize top recommendations"""
        if recommendations.empty:
            print("No recommendations to visualize")
            return
        
        # Prepare data
        rec = recommendations.head(5).copy()
        rec['Label'] = rec.apply(lambda x: f"{x['HomeCounty']} → {x['JobCity']}", axis=1)
        
        # Create radar chart
        categories = ['Affordability', 'Investment', 'Commute', 'Jobs']
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        for idx, row in rec.iterrows():
            values = [
                row['AffordabilityScore'],
                row['InvestmentPotential'],
                100 - (row['CommuteMinutes'] / 60 * 100),  # Invert commute (lower is better)
                row['JobOpportunities'] / rec['JobOpportunities'].max() * 100
            ]
            
            # Close the radar plot
            values += values[:1]
            
            angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
            angles += angles[:1]
            
            ax.plot(angles, values, linewidth=2, label=row['Label'])
            ax.fill(angles, values, alpha=0.1)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Recommendation Comparison', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig('recommendations_radar.png', dpi=300)
        plt.show()
        
        # Create bar chart comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(x='OverallScore', y='Label', data=rec, palette='viridis')
        plt.title('Top Recommendation Scores', fontsize=16)
        plt.xlabel('Overall Score')
        plt.ylabel('Location Pair')
        plt.xlim(0, 100)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('recommendation_scores.png', dpi=300)
        plt.show()

# ======================
# MAIN EXECUTION
# ======================

def main():
    print("===== HOUSING & JOB MARKET ANALYSIS =====")
    print("This system analyzes historical housing and job data to predict")
    print("future affordability and recommend optimal locations\n")
    
    print("Step 1: Loading data...")
    housing_df = DataLoader.load_housing_data()
    job_df = DataLoader.load_job_data()
    
    print("\nStep 2: Visualizing historical trends...")
    Visualizer.plot_historical_trends(housing_df, job_df)
    
    print("\nStep 3: Training predictive models...")
    predictor = PredictiveModels(housing_df, job_df)
    county_models, appreciation_models = predictor.train_housing_models()
    salary_models, job_count_models, job_trend_df = predictor.train_job_market_models()
    
    # Store models
    predictor.county_models = county_models
    predictor.appreciation_models = appreciation_models
    predictor.salary_models = salary_models
    predictor.job_count_models = job_count_models
    
    print("\nStep 4: Generating recommendations...")
    recommender = LocationRecommender(
        housing_df, job_df, 
        housing_models=predictor,
        job_models=predictor
    )
    
    # User preferences
    user_prefs = {
        'job_field': 'Technology',
        'experience_level': 'Mid-level',
        'future_year': 2025,
        'max_commute': 45
    }
    
    print("\nUser Profile:")
    print(f"- Job Field: {user_prefs['job_field']}")
    print(f"- Experience Level: {user_prefs['experience_level']}")
    print(f"- Future Year: {user_prefs['future_year']}")
    print(f"- Max Commute: {user_prefs['max_commute']} minutes\n")
    
    recommendations = recommender.recommend_locations(
        job_field=user_prefs['job_field'],
        experience_level=user_prefs['experience_level'],
        future_year=user_prefs['future_year'],
        max_commute=user_prefs['max_commute']
    )
    
    if not recommendations.empty:
        print("\n===== TOP RECOMMENDATIONS =====")
        print(recommendations[['HomeCounty', 'JobCity', 'CommuteMinutes', 
                              'PredictedSalary', 'PredictedHomePrice', 
                              'AffordabilityScore', 'OverallScore']].head(5))
        
        print("\n===== BEST OVERALL LOCATION =====")
        best = recommendations.iloc[0]
        print(f"Live in: {best['HomeCounty']}, Work in: {best['JobCity']}")
        print(f"Commute: {best['CommuteMinutes']:.0f} minutes")
        print(f"Predicted Salary: ${best['PredictedSalary']:,.0f}")
        print(f"Predicted Home Price: ${best['PredictedHomePrice']:,.0f}")
        print(f"Affordability Score: {best['AffordabilityScore']:.1f}/100")
        print(f"Appreciation Rate: {best['AppreciationRate']:.1%} annually")
        print(f"Job Opportunities: {best['JobOpportunities']:.0f} positions")
        
        Visualizer.plot_recommendations(recommendations)
    else:
        print("No suitable locations found")

if __name__ == "__main__":
    # Install kagglehub if not available
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        import subprocess
        subprocess.check_call(["pip", "install", "kagglehub"])
        import kagglehub
    
    main()