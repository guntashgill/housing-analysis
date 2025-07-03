import numpy as np

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