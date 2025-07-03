import os
import kagglehub
import pandas as pd
import numpy as np
from tqdm import tqdm

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