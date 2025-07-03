
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader, KaggleDataLoader
from predictive_models import PredictiveModels
from recommender import LocationRecommender
from visualizer import Visualizer

sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)


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