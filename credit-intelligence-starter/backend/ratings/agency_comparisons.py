import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company, Score
from typing import List, Optional, Dict, Any, Tuple
import logging
import numpy as np
from dataclasses import dataclass
import json
import os
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class AgencyRating:
    agency: str
    rating: str
    outlook: str
    date: datetime
    scale_position: int  # Numerical position on rating scale
    investment_grade: bool

@dataclass
class RatingComparison:
    company_id: int
    company_name: str
    ticker: str
    internal_score: float
    internal_rating: str
    agency_ratings: List[AgencyRating]
    consensus_rating: str
    rating_spread: int  # Difference between highest and lowest ratings
    upgrade_probability: float
    downgrade_probability: float
    comparison_date: datetime

class AgencyRatingComparator:
    """Compare internal credit ratings with major rating agencies"""
    
    def __init__(self):
        self.session = next(get_db())
        
        # Rating agency scales (higher number = lower credit quality)
        self.rating_scales = {
            'moodys': {
                'Aaa': 1, 'Aa1': 2, 'Aa2': 3, 'Aa3': 4,
                'A1': 5, 'A2': 6, 'A3': 7,
                'Baa1': 8, 'Baa2': 9, 'Baa3': 10,
                'Ba1': 11, 'Ba2': 12, 'Ba3': 13,
                'B1': 14, 'B2': 15, 'B3': 16,
                'Caa1': 17, 'Caa2': 18, 'Caa3': 19,
                'Ca': 20, 'C': 21
            },
            'sp': {
                'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
                'A+': 5, 'A': 6, 'A-': 7,
                'BBB+': 8, 'BBB': 9, 'BBB-': 10,
                'BB+': 11, 'BB': 12, 'BB-': 13,
                'B+': 14, 'B': 15, 'B-': 16,
                'CCC+': 17, 'CCC': 18, 'CCC-': 19,
                'CC': 20, 'C': 21, 'D': 22
            },
            'fitch': {
                'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
                'A+': 5, 'A': 6, 'A-': 7,
                'BBB+': 8, 'BBB': 9, 'BBB-': 10,
                'BB+': 11, 'BB': 12, 'BB-': 13,
                'B+': 14, 'B': 15, 'B-': 16,
                'CCC': 17, 'CC': 18, 'C': 19, 'D': 20
            },
            'internal': {
                'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4,
                'BB': 5, 'B': 6, 'CCC': 7, 'CC': 8, 'C': 9, 'D': 10
            }
        }
        
        # Investment grade thresholds
        self.investment_grade_threshold = {
            'moodys': 10,  # Baa3 and above
            'sp': 10,      # BBB- and above
            'fitch': 10,   # BBB- and above
            'internal': 4  # BBB and above
        }
        
        # Mock agency data (in production, this would come from APIs or data feeds)
        self.mock_agency_ratings = self._generate_mock_agency_data()
    
    def _generate_mock_agency_data(self) -> Dict[str, Dict]:
        """Generate mock agency rating data for demonstration"""
        # In production, this would fetch from actual rating agency APIs/feeds
        mock_data = {
            'AAPL': {
                'moodys': {'rating': 'Aa1', 'outlook': 'Stable', 'date': '2024-01-15'},
                'sp': {'rating': 'AA+', 'outlook': 'Stable', 'date': '2024-01-20'},
                'fitch': {'rating': 'AA+', 'outlook': 'Stable', 'date': '2024-01-18'}
            },
            'MSFT': {
                'moodys': {'rating': 'Aaa', 'outlook': 'Stable', 'date': '2024-02-01'},
                'sp': {'rating': 'AAA', 'outlook': 'Stable', 'date': '2024-02-05'},
                'fitch': {'rating': 'AAA', 'outlook': 'Stable', 'date': '2024-02-03'}
            },
            'TSLA': {
                'moodys': {'rating': 'Ba1', 'outlook': 'Positive', 'date': '2024-01-10'},
                'sp': {'rating': 'BB+', 'outlook': 'Positive', 'date': '2024-01-12'},
                'fitch': {'rating': 'BB+', 'outlook': 'Stable', 'date': '2024-01-08'}
            },
            'F': {
                'moodys': {'rating': 'Ba2', 'outlook': 'Stable', 'date': '2024-01-25'},
                'sp': {'rating': 'BB', 'outlook': 'Stable', 'date': '2024-01-28'},
                'fitch': {'rating': 'BB', 'outlook': 'Negative', 'date': '2024-01-22'}
            },
            'GE': {
                'moodys': {'rating': 'Baa1', 'outlook': 'Stable', 'date': '2024-02-10'},
                'sp': {'rating': 'BBB+', 'outlook': 'Stable', 'date': '2024-02-12'},
                'fitch': {'rating': 'BBB+', 'outlook': 'Stable', 'date': '2024-02-08'}
            }
        }
        return mock_data
    
    def get_agency_ratings(self, ticker: str) -> List[AgencyRating]:
        """Get agency ratings for a company"""
        ratings = []
        
        # In production, this would fetch from actual APIs
        if ticker in self.mock_agency_ratings:
            agency_data = self.mock_agency_ratings[ticker]
            
            for agency, data in agency_data.items():
                rating_str = data['rating']
                scale_position = self.rating_scales[agency].get(rating_str, 15)
                investment_grade = scale_position <= self.investment_grade_threshold[agency]
                
                rating = AgencyRating(
                    agency=agency,
                    rating=rating_str,
                    outlook=data['outlook'],
                    date=datetime.strptime(data['date'], '%Y-%m-%d'),
                    scale_position=scale_position,
                    investment_grade=investment_grade
                )
                ratings.append(rating)
        
        return ratings
    
    def calculate_consensus_rating(self, agency_ratings: List[AgencyRating]) -> Tuple[str, int]:
        """Calculate consensus rating from agency ratings"""
        if not agency_ratings:
            return 'NR', 0  # Not Rated
        
        # Calculate average scale position
        positions = [rating.scale_position for rating in agency_ratings]
        avg_position = np.mean(positions)
        
        # Map back to rating (using S&P scale as reference)
        sp_scale_reverse = {v: k for k, v in self.rating_scales['sp'].items()}
        
        # Find closest rating
        closest_position = min(sp_scale_reverse.keys(), key=lambda x: abs(x - avg_position))
        consensus_rating = sp_scale_reverse[closest_position]
        
        # Calculate spread (difference between best and worst ratings)
        rating_spread = max(positions) - min(positions)
        
        return consensus_rating, rating_spread
    
    def compare_with_internal_rating(self, company_id: int) -> Optional[RatingComparison]:
        """Compare internal rating with agency ratings"""
        try:
            # Get company info
            company = self.session.query(Company).filter(Company.id == company_id).first()
            if not company:
                return None
            
            # Get latest internal score
            latest_score = self.session.query(Score).filter(
                Score.company_id == company_id
            ).order_by(Score.score_date.desc()).first()
            
            if not latest_score:
                return None
            
            # Get agency ratings
            agency_ratings = self.get_agency_ratings(company.ticker)
            
            # Calculate consensus
            consensus_rating, rating_spread = self.calculate_consensus_rating(agency_ratings)
            
            # Calculate upgrade/downgrade probabilities
            upgrade_prob, downgrade_prob = self._calculate_rating_probabilities(
                latest_score, agency_ratings
            )
            
            comparison = RatingComparison(
                company_id=company_id,
                company_name=company.name,
                ticker=company.ticker,
                internal_score=latest_score.credit_score,
                internal_rating=latest_score.risk_category,
                agency_ratings=agency_ratings,
                consensus_rating=consensus_rating,
                rating_spread=rating_spread,
                upgrade_probability=upgrade_prob,
                downgrade_probability=downgrade_prob,
                comparison_date=datetime.now()
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing ratings for company {company_id}: {str(e)}")
            return None
    
    def _calculate_rating_probabilities(self, internal_score: Score, 
                                      agency_ratings: List[AgencyRating]) -> Tuple[float, float]:
        """Calculate upgrade/downgrade probabilities based on rating differences"""
        if not agency_ratings:
            return 0.0, 0.0
        
        # Get internal rating position
        internal_position = self.rating_scales['internal'].get(internal_score.risk_category, 5)
        
        # Get average agency position (normalized to internal scale)
        agency_positions = []
        for rating in agency_ratings:
            # Normalize agency position to internal scale (rough approximation)
            normalized_pos = rating.scale_position * (10 / 22)  # Scale to internal range
            agency_positions.append(normalized_pos)
        
        avg_agency_position = np.mean(agency_positions)
        
        # Calculate difference (positive = internal is worse than agencies)
        position_diff = internal_position - avg_agency_position
        
        # Calculate probabilities based on difference and outlook
        base_upgrade_prob = max(0, position_diff * 0.1)  # 10% per position difference
        base_downgrade_prob = max(0, -position_diff * 0.1)
        
        # Adjust based on agency outlooks
        positive_outlooks = sum(1 for r in agency_ratings if r.outlook == 'Positive')
        negative_outlooks = sum(1 for r in agency_ratings if r.outlook == 'Negative')
        
        outlook_adjustment = (positive_outlooks - negative_outlooks) * 0.05
        
        upgrade_prob = min(max(base_upgrade_prob + outlook_adjustment, 0.0), 1.0)
        downgrade_prob = min(max(base_downgrade_prob - outlook_adjustment, 0.0), 1.0)
        
        return upgrade_prob, downgrade_prob
    
    def analyze_rating_convergence(self, company_id: int, months_back: int = 12) -> Dict[str, Any]:
        """Analyze how internal ratings have converged with agency ratings over time"""
        try:
            company = self.session.query(Company).filter(Company.id == company_id).first()
            if not company:
                return {'error': 'Company not found'}
            
            # Get historical internal scores
            cutoff_date = datetime.now() - timedelta(days=months_back * 30)
            historical_scores = self.session.query(Score).filter(
                Score.company_id == company_id,
                Score.score_date >= cutoff_date
            ).order_by(Score.score_date).all()
            
            if not historical_scores:
                return {'error': 'No historical scores found'}
            
            # Get current agency ratings (in production, would get historical)
            agency_ratings = self.get_agency_ratings(company.ticker)
            consensus_rating, _ = self.calculate_consensus_rating(agency_ratings)
            
            # Analyze convergence
            convergence_data = []
            for score in historical_scores:
                internal_pos = self.rating_scales['internal'].get(score.risk_category, 5)
                consensus_pos = self.rating_scales['sp'].get(consensus_rating, 10)
                
                # Normalize to same scale
                normalized_consensus = consensus_pos * (10 / 22)
                
                difference = abs(internal_pos - normalized_consensus)
                
                convergence_data.append({
                    'date': score.score_date.isoformat(),
                    'internal_rating': score.risk_category,
                    'internal_score': score.credit_score,
                    'difference_from_consensus': difference,
                    'converging': len(convergence_data) == 0 or difference < convergence_data[-1]['difference_from_consensus']
                })
            
            # Calculate overall convergence trend
            if len(convergence_data) >= 2:
                recent_diff = np.mean([d['difference_from_consensus'] for d in convergence_data[-3:]])
                early_diff = np.mean([d['difference_from_consensus'] for d in convergence_data[:3]])
                convergence_trend = 'improving' if recent_diff < early_diff else 'diverging'
            else:
                convergence_trend = 'insufficient_data'
            
            return {
                'company_name': company.name,
                'ticker': company.ticker,
                'consensus_rating': consensus_rating,
                'current_internal_rating': historical_scores[-1].risk_category,
                'convergence_trend': convergence_trend,
                'convergence_history': convergence_data,
                'analysis_period_months': months_back
            }
            
        except Exception as e:
            logger.error(f"Error analyzing rating convergence: {str(e)}")
            return {'error': str(e)}
    
    def bulk_compare_ratings(self, company_ids: List[int] = None) -> Dict[str, Any]:
        """Compare ratings for multiple companies"""
        try:
            if company_ids is None:
                # Get all companies with recent scores
                companies = self.session.query(Company).join(Score).distinct().all()
                company_ids = [c.id for c in companies]
            
            comparisons = []
            summary_stats = {
                'total_companies': 0,
                'with_agency_ratings': 0,
                'investment_grade_internal': 0,
                'investment_grade_consensus': 0,
                'rating_disagreements': 0,
                'avg_rating_spread': 0.0
            }
            
            rating_spreads = []
            
            for company_id in company_ids:
                comparison = self.compare_with_internal_rating(company_id)
                if comparison:
                    comparisons.append({
                        'company_id': comparison.company_id,
                        'company_name': comparison.company_name,
                        'ticker': comparison.ticker,
                        'internal_rating': comparison.internal_rating,
                        'internal_score': comparison.internal_score,
                        'consensus_rating': comparison.consensus_rating,
                        'agency_count': len(comparison.agency_ratings),
                        'rating_spread': comparison.rating_spread,
                        'upgrade_probability': comparison.upgrade_probability,
                        'downgrade_probability': comparison.downgrade_probability,
                        'agencies': [
                            {
                                'agency': r.agency,
                                'rating': r.rating,
                                'outlook': r.outlook,
                                'investment_grade': r.investment_grade
                            }
                            for r in comparison.agency_ratings
                        ]
                    })
                    
                    # Update summary stats
                    summary_stats['total_companies'] += 1
                    
                    if comparison.agency_ratings:
                        summary_stats['with_agency_ratings'] += 1
                        rating_spreads.append(comparison.rating_spread)
                        
                        # Check investment grade status
                        internal_pos = self.rating_scales['internal'].get(comparison.internal_rating, 5)
                        if internal_pos <= self.investment_grade_threshold['internal']:
                            summary_stats['investment_grade_internal'] += 1
                        
                        consensus_pos = self.rating_scales['sp'].get(comparison.consensus_rating, 15)
                        if consensus_pos <= self.investment_grade_threshold['sp']:
                            summary_stats['investment_grade_consensus'] += 1
                        
                        # Check for significant disagreements (>2 notches difference)
                        if comparison.rating_spread > 2:
                            summary_stats['rating_disagreements'] += 1
            
            if rating_spreads:
                summary_stats['avg_rating_spread'] = np.mean(rating_spreads)
            
            return {
                'success': True,
                'summary': summary_stats,
                'comparisons': comparisons,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in bulk rating comparison: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_rating_distribution_analysis(self) -> Dict[str, Any]:
        """Analyze rating distribution across internal vs agency ratings"""
        try:
            # Get all companies with recent scores
            companies = self.session.query(Company).join(Score).distinct().all()
            
            internal_distribution = defaultdict(int)
            agency_distribution = defaultdict(int)
            
            for company in companies:
                # Get latest internal score
                latest_score = self.session.query(Score).filter(
                    Score.company_id == company.id
                ).order_by(Score.score_date.desc()).first()
                
                if latest_score:
                    internal_distribution[latest_score.risk_category] += 1
                    
                    # Get agency ratings
                    agency_ratings = self.get_agency_ratings(company.ticker)
                    if agency_ratings:
                        consensus_rating, _ = self.calculate_consensus_rating(agency_ratings)
                        agency_distribution[consensus_rating] += 1
            
            return {
                'internal_distribution': dict(internal_distribution),
                'agency_consensus_distribution': dict(agency_distribution),
                'total_companies_analyzed': len(companies),
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing rating distribution: {str(e)}")
            return {'error': str(e)}
    
    def close(self):
        """Close database session"""
        self.session.close()

# Example usage
if __name__ == "__main__":
    comparator = AgencyRatingComparator()
    
    try:
        print("Running rating comparison analysis...")
        
        # Bulk comparison
        result = comparator.bulk_compare_ratings()
        
        if result['success']:
            summary = result['summary']
            print(f"Analyzed {summary['total_companies']} companies")
            print(f"Companies with agency ratings: {summary['with_agency_ratings']}")
            print(f"Investment grade (internal): {summary['investment_grade_internal']}")
            print(f"Investment grade (consensus): {summary['investment_grade_consensus']}")
            print(f"Rating disagreements: {summary['rating_disagreements']}")
            print(f"Average rating spread: {summary['avg_rating_spread']:.1f}")
            
            print("\nTop 5 companies by rating spread:")
            sorted_comparisons = sorted(result['comparisons'], 
                                      key=lambda x: x['rating_spread'], reverse=True)
            
            for comp in sorted_comparisons[:5]:
                print(f"  {comp['ticker']}: Internal {comp['internal_rating']} vs "
                      f"Consensus {comp['consensus_rating']} (spread: {comp['rating_spread']})")
        
        else:
            print(f"Error: {result['error']}")
        
        # Distribution analysis
        print("\nRating distribution analysis:")
        distribution = comparator.get_rating_distribution_analysis()
        
        if 'error' not in distribution:
            print("Internal ratings:", distribution['internal_distribution'])
            print("Agency consensus:", distribution['agency_consensus_distribution'])
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        comparator.close()
