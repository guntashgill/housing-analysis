import numpy as np
import pandas as pd


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
