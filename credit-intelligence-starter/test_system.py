#!/usr/bin/env python3
"""
System integration test script for Credit Risk Assessment Platform
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class SystemTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self) -> bool:
        """Test basic health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✓ Health check passed")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check failed: {str(e)}")
            return False
    
    def test_companies_endpoint(self) -> bool:
        """Test companies listing endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/companies")
            if response.status_code == 200:
                companies = response.json()
                print(f"✓ Companies endpoint working - Found {len(companies)} companies")
                return True
            else:
                print(f"✗ Companies endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Companies endpoint failed: {str(e)}")
            return False
    
    def test_data_initialization(self) -> bool:
        """Test data system initialization"""
        try:
            response = self.session.post(f"{self.base_url}/data/initialize")
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Data initialization successful: {result}")
                return True
            else:
                print(f"✗ Data initialization failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Data initialization failed: {str(e)}")
            return False
    
    def test_model_training(self) -> bool:
        """Test model training endpoint"""
        try:
            payload = {"n_samples": 500}  # Small sample for testing
            response = self.session.post(f"{self.base_url}/scoring/train", json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Model training started: {result}")
                return True
            else:
                print(f"✗ Model training failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Model training failed: {str(e)}")
            return False
    
    def test_data_collection(self) -> bool:
        """Test data collection for a few companies"""
        try:
            payload = {"tickers": ["AAPL", "MSFT"], "days_back": 3}
            response = self.session.post(f"{self.base_url}/data/collect", json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Data collection started: {result}")
                return True
            else:
                print(f"✗ Data collection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Data collection failed: {str(e)}")
            return False
    
    def test_credit_scoring(self, company_id: int = 1) -> bool:
        """Test credit score prediction"""
        try:
            payload = {"company_id": company_id}
            response = self.session.post(f"{self.base_url}/scoring/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                if 'error' not in result:
                    print(f"✓ Credit scoring successful: Score {result.get('credit_score', 'N/A')}")
                    return True
                else:
                    print(f"✗ Credit scoring failed: {result['error']}")
                    return False
            else:
                print(f"✗ Credit scoring failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Credit scoring failed: {str(e)}")
            return False
    
    def test_data_status(self) -> bool:
        """Test data status endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/data/status")
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Data status retrieved: {result.get('total_companies', 0)} companies tracked")
                return True
            else:
                print(f"✗ Data status failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Data status failed: {str(e)}")
            return False
    
    def run_comprehensive_test(self) -> Dict[str, bool]:
        """Run all system tests"""
        print("=" * 60)
        print("Credit Risk Assessment System - Integration Test")
        print("=" * 60)
        
        results = {}
        
        # Basic connectivity tests
        print("\n1. Testing Basic Connectivity...")
        results['health_check'] = self.test_health_check()
        
        if not results['health_check']:
            print("❌ System not accessible. Stopping tests.")
            return results
        
        # API endpoint tests
        print("\n2. Testing API Endpoints...")
        results['companies'] = self.test_companies_endpoint()
        results['data_status'] = self.test_data_status()
        
        # Data system tests
        print("\n3. Testing Data System...")
        results['data_init'] = self.test_data_initialization()
        
        # Wait a moment for initialization
        time.sleep(2)
        
        results['data_collection'] = self.test_data_collection()
        
        # ML system tests
        print("\n4. Testing ML System...")
        results['model_training'] = self.test_model_training()
        
        # Wait for some processing
        time.sleep(3)
        
        results['credit_scoring'] = self.test_credit_scoring()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✓ PASS" if passed_test else "✗ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! System is working correctly.")
        elif passed >= total * 0.7:
            print("⚠️  Most tests passed. Some features may need attention.")
        else:
            print("❌ Multiple test failures. System needs debugging.")
        
        return results

def main():
    """Main test execution"""
    tester = SystemTester()
    
    # Wait for system to be ready
    print("Waiting for system to be ready...")
    max_retries = 30
    for i in range(max_retries):
        if tester.test_health_check():
            break
        print(f"Attempt {i+1}/{max_retries} - System not ready, waiting...")
        time.sleep(5)
    else:
        print("❌ System failed to start after 150 seconds")
        sys.exit(1)
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
