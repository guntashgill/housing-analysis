import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
