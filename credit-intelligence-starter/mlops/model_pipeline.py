import os
import json
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company, Score, Price, NewsRaw
from scoring.credit_scoring_engine import CreditScoringEngine
from scoring.explainability import ExplainabilityEngine
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import boto3
from pathlib import Path
import schedule
import time
import threading

logger = logging.getLogger(__name__)

class MLOpsModelPipeline:
    """Automated MLOps pipeline for credit scoring models"""
    
    def __init__(self, 
                 mlflow_tracking_uri: str = None,
                 s3_bucket: str = None,
                 model_registry_name: str = "credit-scoring-model"):
        
        self.session = next(get_db())
        self.scoring_engine = CreditScoringEngine()
        self.explainability_engine = ExplainabilityEngine()
        
        # MLflow configuration
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            mlflow.set_tracking_uri("file:./mlruns")
        
        self.model_registry_name = model_registry_name
        
        # AWS S3 configuration for model artifacts
        self.s3_bucket = s3_bucket
        if s3_bucket:
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = None
        
        # Model performance thresholds
        self.performance_thresholds = {
            'min_accuracy': 0.75,
            'min_precision': 0.70,
            'min_recall': 0.65,
            'min_f1': 0.70,
            'min_auc': 0.80,
            'max_drift_score': 0.3
        }
        
        # Model versioning
        self.model_versions = {}
        self.current_production_model = None
        
        # Monitoring metrics
        self.monitoring_metrics = {
            'prediction_count': 0,
            'avg_confidence': 0.0,
            'drift_alerts': 0,
            'performance_alerts': 0,
            'last_retrain_date': None,
            'model_version': None
        }
    
    def collect_training_data(self, months_back: int = 12) -> Tuple[pd.DataFrame, pd.Series]:
        """Collect and prepare training data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=months_back * 30)
            
            # Get companies with sufficient data
            companies = self.session.query(Company).join(Score).join(Price).join(NewsRaw).filter(
                Score.score_date >= cutoff_date
            ).distinct().all()
            
            training_data = []
            
            for company in companies:
                # Get historical data
                scores = self.session.query(Score).filter(
                    Score.company_id == company.id,
                    Score.score_date >= cutoff_date
                ).order_by(Score.score_date).all()
                
                prices = self.session.query(Price).filter(
                    Price.company_id == company.id,
                    Price.date >= cutoff_date
                ).order_by(Price.date).all()
                
                news = self.session.query(NewsRaw).filter(
                    NewsRaw.company_id == company.id,
                    NewsRaw.published_at >= cutoff_date
                ).all()
                
                # Create features for each time period
                for i, score in enumerate(scores):
                    if i == 0:
                        continue  # Skip first score as we need historical data
                    
                    # Get data up to this point
                    historical_prices = [p for p in prices if p.date <= score.score_date]
                    historical_news = [n for n in news if n.published_at <= score.score_date]
                    
                    if len(historical_prices) < 10 or len(historical_news) < 5:
                        continue
                    
                    # Generate features
                    features = self.scoring_engine.generate_features(
                        company.id, score.score_date
                    )
                    
                    if features:
                        # Add target variable (next period's rating change)
                        if i < len(scores) - 1:
                            next_score = scores[i + 1]
                            current_rating_pos = self._get_rating_position(score.risk_category)
                            next_rating_pos = self._get_rating_position(next_score.risk_category)
                            
                            # Target: 1 if rating improved, 0 if stayed same, -1 if worsened
                            if next_rating_pos < current_rating_pos:
                                target = 1  # Improvement
                            elif next_rating_pos > current_rating_pos:
                                target = -1  # Deterioration
                            else:
                                target = 0  # No change
                            
                            features['target'] = target
                            features['company_id'] = company.id
                            features['date'] = score.score_date
                            
                            training_data.append(features)
            
            if not training_data:
                raise ValueError("No training data collected")
            
            # Convert to DataFrame
            df = pd.DataFrame(training_data)
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col not in ['target', 'company_id', 'date']]
            X = df[feature_cols]
            y = df['target']
            
            # Handle missing values
            X = X.fillna(X.median())
            
            logger.info(f"Collected {len(X)} training samples with {len(feature_cols)} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error collecting training data: {str(e)}")
            raise
    
    def _get_rating_position(self, rating: str) -> int:
        """Get numerical position of rating (lower = better)"""
        rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
        return rating_order.index(rating) if rating in rating_order else 5
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   experiment_name: str = "credit-scoring") -> Dict[str, Any]:
        """Train and evaluate model with MLflow tracking"""
        
        try:
            # Set MLflow experiment
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_param("n_samples", len(X))
                mlflow.log_param("n_features", len(X.columns))
                mlflow.log_param("training_date", datetime.now().isoformat())
                
                # Train model using existing scoring engine
                model_results = self.scoring_engine.train_models(X, y)
                
                if not model_results['success']:
                    raise ValueError(f"Model training failed: {model_results['message']}")
                
                # Get best model
                best_model_name = model_results['best_model']
                best_model = model_results['models'][best_model_name]
                
                # Cross-validation
                cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
                
                # Calculate metrics
                y_pred = best_model.predict(X)
                y_pred_proba = best_model.predict_proba(X) if hasattr(best_model, 'predict_proba') else None
                
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, average='weighted')
                recall = recall_score(y, y_pred, average='weighted')
                f1 = f1_score(y, y_pred, average='weighted')
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("cv_mean", cv_scores.mean())
                mlflow.log_metric("cv_std", cv_scores.std())
                
                if y_pred_proba is not None:
                    # For multiclass, calculate AUC for each class
                    try:
                        auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
                        mlflow.log_metric("auc", auc)
                    except:
                        auc = 0.0
                else:
                    auc = 0.0
                
                # Log model
                mlflow.sklearn.log_model(
                    best_model, 
                    "model",
                    registered_model_name=self.model_registry_name
                )
                
                # Log feature importance if available
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Save as artifact
                    feature_importance.to_csv("feature_importance.csv", index=False)
                    mlflow.log_artifact("feature_importance.csv")
                
                # Model validation
                model_valid = self._validate_model_performance({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc
                })
                
                mlflow.log_param("model_valid", model_valid)
                
                # Save model locally
                model_path = f"models/credit_model_{run.info.run_id}.pkl"
                os.makedirs("models", exist_ok=True)
                joblib.dump(best_model, model_path)
                
                # Upload to S3 if configured
                if self.s3_client and model_valid:
                    s3_key = f"models/credit_model_{run.info.run_id}.pkl"
                    self.s3_client.upload_file(model_path, self.s3_bucket, s3_key)
                    mlflow.log_param("s3_model_path", f"s3://{self.s3_bucket}/{s3_key}")
                
                result = {
                    'success': True,
                    'run_id': run.info.run_id,
                    'model_name': best_model_name,
                    'model_path': model_path,
                    'metrics': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': auc,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    },
                    'model_valid': model_valid,
                    'training_date': datetime.now().isoformat()
                }
                
                # Update model registry
                if model_valid:
                    self.model_versions[run.info.run_id] = result
                    if self._should_promote_model(result):
                        self.current_production_model = run.info.run_id
                        mlflow.log_param("promoted_to_production", True)
                
                return result
                
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _validate_model_performance(self, metrics: Dict[str, float]) -> bool:
        """Validate if model meets performance thresholds"""
        for metric, threshold in self.performance_thresholds.items():
            if metric.startswith('min_') and metrics.get(metric[4:], 0) < threshold:
                logger.warning(f"Model failed {metric} threshold: {metrics.get(metric[4:], 0)} < {threshold}")
                return False
            elif metric.startswith('max_') and metrics.get(metric[4:], 1) > threshold:
                logger.warning(f"Model failed {metric} threshold: {metrics.get(metric[4:], 1)} > {threshold}")
                return False
        return True
    
    def _should_promote_model(self, model_result: Dict) -> bool:
        """Determine if model should be promoted to production"""
        if not model_result['model_valid']:
            return False
        
        # If no current production model, promote
        if not self.current_production_model:
            return True
        
        # Compare with current production model
        current_model = self.model_versions.get(self.current_production_model)
        if not current_model:
            return True
        
        # Promote if significantly better
        new_f1 = model_result['metrics']['f1_score']
        current_f1 = current_model['metrics']['f1_score']
        
        improvement_threshold = 0.02  # 2% improvement required
        return new_f1 > current_f1 + improvement_threshold
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and current data"""
        try:
            from scipy import stats
            
            drift_results = {}
            drift_detected = False
            
            for column in reference_data.columns:
                if column in current_data.columns:
                    # Kolmogorov-Smirnov test for distribution drift
                    ks_stat, ks_pvalue = stats.ks_2samp(
                        reference_data[column].dropna(),
                        current_data[column].dropna()
                    )
                    
                    # Population Stability Index (PSI)
                    psi = self._calculate_psi(
                        reference_data[column].dropna(),
                        current_data[column].dropna()
                    )
                    
                    drift_results[column] = {
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'psi': psi,
                        'drift_detected': psi > 0.2 or ks_pvalue < 0.05
                    }
                    
                    if drift_results[column]['drift_detected']:
                        drift_detected = True
            
            overall_drift_score = np.mean([r['psi'] for r in drift_results.values()])
            
            return {
                'drift_detected': drift_detected,
                'overall_drift_score': overall_drift_score,
                'feature_drift': drift_results,
                'drift_threshold_exceeded': overall_drift_score > self.performance_thresholds['max_drift_score']
            }
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, 
                      buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=buckets)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / len(reference)
            cur_props = cur_counts / len(current)
            
            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            cur_props = np.where(cur_props == 0, 0.0001, cur_props)
            
            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            
            return psi
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {str(e)}")
            return 0.0
    
    def monitor_model_performance(self) -> Dict[str, Any]:
        """Monitor current model performance"""
        try:
            if not self.current_production_model:
                return {'error': 'No production model available'}
            
            # Get recent predictions and actual outcomes
            cutoff_date = datetime.now() - timedelta(days=30)
            recent_scores = self.session.query(Score).filter(
                Score.score_date >= cutoff_date
            ).all()
            
            if len(recent_scores) < 10:
                return {'warning': 'Insufficient recent data for monitoring'}
            
            # Calculate performance metrics on recent data
            # This is simplified - in practice, you'd need actual outcomes
            monitoring_results = {
                'monitoring_period': '30_days',
                'sample_count': len(recent_scores),
                'avg_confidence': np.mean([s.confidence or 0.5 for s in recent_scores]),
                'rating_distribution': self._get_rating_distribution(recent_scores),
                'alerts': []
            }
            
            # Check for performance degradation indicators
            if monitoring_results['avg_confidence'] < 0.6:
                monitoring_results['alerts'].append({
                    'type': 'low_confidence',
                    'message': 'Average model confidence below threshold',
                    'severity': 'medium'
                })
            
            # Update monitoring metrics
            self.monitoring_metrics.update({
                'prediction_count': len(recent_scores),
                'avg_confidence': monitoring_results['avg_confidence'],
                'last_monitoring_date': datetime.now().isoformat()
            })
            
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Error monitoring model performance: {str(e)}")
            return {'error': str(e)}
    
    def _get_rating_distribution(self, scores: List[Score]) -> Dict[str, int]:
        """Get distribution of ratings"""
        distribution = {}
        for score in scores:
            rating = score.risk_category
            distribution[rating] = distribution.get(rating, 0) + 1
        return distribution
    
    def automated_retraining_pipeline(self) -> Dict[str, Any]:
        """Run automated retraining pipeline"""
        try:
            logger.info("Starting automated retraining pipeline...")
            
            # Step 1: Collect fresh training data
            X, y = self.collect_training_data(months_back=18)
            
            # Step 2: Check for data drift
            if hasattr(self, 'reference_data'):
                drift_results = self.detect_data_drift(self.reference_data, X)
                if drift_results.get('drift_threshold_exceeded'):
                    logger.warning("Significant data drift detected - retraining recommended")
            
            # Step 3: Train new model
            training_result = self.train_model(X, y, f"automated-retrain-{datetime.now().strftime('%Y%m%d')}")
            
            if not training_result['success']:
                return {
                    'success': False,
                    'error': 'Model training failed',
                    'details': training_result
                }
            
            # Step 4: Model validation and A/B testing setup
            if training_result['model_valid']:
                # Store reference data for future drift detection
                self.reference_data = X
                
                # Update monitoring
                self.monitoring_metrics['last_retrain_date'] = datetime.now().isoformat()
                self.monitoring_metrics['model_version'] = training_result['run_id']
                
                return {
                    'success': True,
                    'message': 'Automated retraining completed successfully',
                    'model_promoted': self.current_production_model == training_result['run_id'],
                    'training_results': training_result,
                    'retrain_date': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'message': 'New model did not meet performance thresholds',
                    'training_results': training_result
                }
                
        except Exception as e:
            logger.error(f"Error in automated retraining: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def deploy_model_to_production(self, run_id: str) -> Dict[str, Any]:
        """Deploy model to production"""
        try:
            if run_id not in self.model_versions:
                return {'success': False, 'error': 'Model version not found'}
            
            model_info = self.model_versions[run_id]
            
            # Load model
            model_path = model_info['model_path']
            if not os.path.exists(model_path):
                return {'success': False, 'error': 'Model file not found'}
            
            # Update production model
            self.current_production_model = run_id
            
            # Update scoring engine with new model
            new_model = joblib.load(model_path)
            self.scoring_engine.models[model_info['model_name']] = new_model
            self.scoring_engine.best_model_name = model_info['model_name']
            
            # Log deployment
            with mlflow.start_run(run_id=run_id):
                mlflow.log_param("deployed_to_production", True)
                mlflow.log_param("deployment_date", datetime.now().isoformat())
            
            logger.info(f"Model {run_id} deployed to production")
            
            return {
                'success': True,
                'message': f'Model {run_id} deployed to production',
                'deployment_date': datetime.now().isoformat(),
                'model_metrics': model_info['metrics']
            }
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_model_registry_status(self) -> Dict[str, Any]:
        """Get status of model registry"""
        try:
            return {
                'total_models': len(self.model_versions),
                'production_model': self.current_production_model,
                'model_versions': {
                    run_id: {
                        'training_date': info['training_date'],
                        'model_name': info['model_name'],
                        'f1_score': info['metrics']['f1_score'],
                        'model_valid': info['model_valid']
                    }
                    for run_id, info in self.model_versions.items()
                },
                'monitoring_metrics': self.monitoring_metrics,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting model registry status: {str(e)}")
            return {'error': str(e)}
    
    def schedule_automated_tasks(self):
        """Schedule automated MLOps tasks"""
        # Schedule retraining every week
        schedule.every().sunday.at("02:00").do(self.automated_retraining_pipeline)
        
        # Schedule monitoring every day
        schedule.every().day.at("06:00").do(self.monitor_model_performance)
        
        # Schedule data drift detection every 3 days
        schedule.every(3).days.at("12:00").do(self._check_data_drift_scheduled)
        
        logger.info("Automated MLOps tasks scheduled")
    
    def _check_data_drift_scheduled(self):
        """Scheduled data drift check"""
        try:
            if hasattr(self, 'reference_data'):
                # Get recent data
                X, _ = self.collect_training_data(months_back=1)
                drift_results = self.detect_data_drift(self.reference_data, X)
                
                if drift_results.get('drift_threshold_exceeded'):
                    logger.warning("Scheduled drift check: Significant drift detected")
                    self.monitoring_metrics['drift_alerts'] += 1
                    
                    # Trigger retraining if drift is severe
                    if drift_results['overall_drift_score'] > 0.5:
                        logger.info("Severe drift detected - triggering automated retraining")
                        self.automated_retraining_pipeline()
        except Exception as e:
            logger.error(f"Error in scheduled drift check: {str(e)}")
    
    def run_scheduler(self):
        """Run the task scheduler"""
        logger.info("Starting MLOps scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start_background_scheduler(self):
        """Start scheduler in background thread"""
        scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        scheduler_thread.start()
        logger.info("MLOps scheduler started in background")
    
    def close(self):
        """Close database connections"""
        self.session.close()

# Example usage
if __name__ == "__main__":
    # Initialize MLOps pipeline
    pipeline = MLOpsModelPipeline()
    
    try:
        print("Running automated retraining pipeline...")
        result = pipeline.automated_retraining_pipeline()
        
        if result['success']:
            print("Retraining completed successfully!")
            print(f"Model promoted to production: {result['model_promoted']}")
            
            # Get model registry status
            registry_status = pipeline.get_model_registry_status()
            print(f"Total models in registry: {registry_status['total_models']}")
            print(f"Production model: {registry_status['production_model']}")
        else:
            print(f"Retraining failed: {result['error']}")
        
        # Monitor performance
        monitoring_result = pipeline.monitor_model_performance()
        if 'error' not in monitoring_result:
            print(f"Recent predictions: {monitoring_result['sample_count']}")
            print(f"Average confidence: {monitoring_result['avg_confidence']:.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        pipeline.close()
