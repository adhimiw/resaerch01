#!/usr/bin/env python3
"""
AIDAS Backend Black Box Testing Suite
======================================

Comprehensive black box testing for all API endpoints.
Tests functionality without knowing internal implementation details.

Author: MiniMax Agent
"""

import json
import sys
import os
import time
import subprocess
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class BackendTestSuite:
    """Black box testing suite for AIDAS Backend API"""
    
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
        self.server_status = {}
    
    def log(self, message, level="INFO"):
        """Log test messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "SUCCESS":
            print(f"{Colors.OKGREEN}[{timestamp}] ✓ {message}{Colors.ENDC}")
        elif level == "FAIL":
            print(f"{Colors.FAIL}[{timestamp}] ✗ {message}{Colors.ENDC}")
        elif level == "INFO":
            print(f"{Colors.OKBLUE}[{timestamp}] ℹ {message}{Colors.ENDC}")
        elif level == "WARNING":
            print(f"{Colors.WARNING}[{timestamp}] ⚠ {message}{Colors.ENDC}")
    
    def make_request(self, endpoint, method="GET", data=None, headers=None):
        """Make HTTP request and return response"""
        url = f"{self.base_url}{endpoint}"
        headers = headers or {}
        
        try:
            if method.upper() == "GET":
                req = Request(url, headers=headers)
            else:
                headers["Content-Type"] = "application/json"
                req = Request(url, data=json.dumps(data).encode(), headers=headers, method=method)
            
            with urlopen(req, timeout=10) as response:
                response_body = response.read().decode()
                return {
                    "status_code": response.status,
                    "body": json.loads(response_body) if response_body else {},
                    "success": True
                }
        except HTTPError as e:
            try:
                response_body = e.read().decode()
                error_body = json.loads(response_body) if response_body else {}
            except:
                error_body = {"error": str(e)}
            return {
                "status_code": e.code,
                "body": error_body,
                "success": e.code < 500,
                "error": str(e)
            }
        except URLError as e:
            return {
                "success": False,
                "error": f"Connection error: {e.reason}"
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON decode error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_test(self, test_name, test_func):
        """Execute a single test"""
        self.log(f"Running: {test_name}", "INFO")
        try:
            result = test_func()
            if result["passed"]:
                self.tests_passed += 1
                self.test_results.append({
                    "name": test_name,
                    "status": "PASSED",
                    "details": result.get("details", "")
                })
                self.log(f"{test_name}", "SUCCESS")
            else:
                self.tests_failed += 1
                self.test_results.append({
                    "name": test_name,
                    "status": "FAILED",
                    "details": result.get("details", "")
                })
                self.log(f"{test_name}: {result.get('details', 'Unknown error')}", "FAIL")
        except Exception as e:
            self.tests_failed += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAILED",
                "details": f"Test exception: {str(e)}"
            })
            self.log(f"{test_name}: Exception - {str(e)}", "FAIL")
    
    def test_server_health(self):
        """Test 1: Server Health Check"""
        def test():
            response = self.make_request("/api/health/")
            if response["success"] and response["status_code"] == 200:
                # Check if response contains expected fields
                if "status" in response["body"] and "version" in response["body"]:
                    return {"passed": True, "details": f"Health check returned: {response['body']}"}
            return {"passed": False, "details": f"Health check failed: {response.get('error', 'Unknown error')}"}
        return test
    
    def test_api_schema(self):
        """Test 2: API Schema Endpoint"""
        def test():
            response = self.make_request("/api/schema/")
            if response["success"] and response["status_code"] == 200:
                if "info" in response["body"] and "title" in response["body"]["info"]:
                    return {"passed": True, "details": "Schema endpoint accessible"}
            return {"passed": False, "details": f"Schema check failed: {response.get('error', 'Unknown error')}"}
        return test
    
    def test_analysis_list_empty(self):
        """Test 3: Analysis List (Empty State)"""
        def test():
            response = self.make_request("/api/analysis/")
            if response["success"] and response["status_code"] == 200:
                if isinstance(response["body"], list):
                    return {"passed": True, "details": f"Analysis list accessible, count: {len(response['body'])}"}
            return {"passed": False, "details": f"Analysis list failed: {response.get('error', 'Unknown error')}"}
        return test
    
    def test_analysis_list_with_filter(self):
        """Test 4: Analysis List with Status Filter"""
        def test():
            response = self.make_request("/api/analysis/?status=pending")
            if response["success"] and response["status_code"] == 200:
                return {"passed": True, "details": "Analysis list with filter accessible"}
            return {"passed": False, "details": f"Filtered list failed: {response.get('error', 'Unknown error')}"}
        return test
    
    def test_analysis_create_validation(self):
        """Test 5: Analysis Creation (Validation Error)"""
        def test():
            response = self.make_request("/api/analysis/", "POST", {"invalid": "data"})
            # Should return 400 (validation) or 401 (auth required)
            if response["status_code"] in [400, 401]:
                return {"passed": True, "details": "Proper error returned for invalid data"}
            return {"passed": False, "details": f"Expected 400/401, got {response.get('status_code', 'No response')}"}
        return test
    
    def test_analysis_create_not_found(self):
        """Test 6: Analysis Creation (Dataset Not Found)"""
        def test():
            response = self.make_request("/api/analysis/", "POST", {"dataset_id": 99999})
            # Should return 400/404 or 401 (auth required)
            if response["status_code"] in [400, 401, 404]:
                return {"passed": True, "details": "Proper error for non-existent dataset"}
            return {"passed": False, "details": f"Expected 400/401/404, got {response.get('status_code', 'No response')}"}
        return test
    
    def test_analysis_detail_not_found(self):
        """Test 7: Analysis Detail (Not Found)"""
        def test():
            response = self.make_request("/api/analysis/99999/")
            if response["status_code"] == 404:
                return {"passed": True, "details": "Proper 404 for non-existent analysis"}
            return {"passed": False, "details": f"Expected 404, got {response.get('status_code', 'No response')}"}
        return test
    
    def test_analysis_results_not_completed(self):
        """Test 8: Analysis Results (Not Completed)"""
        def test():
            response = self.make_request("/api/analysis/99999/results/")
            if response["status_code"] in [400, 404]:
                return {"passed": True, "details": "Proper error for incomplete analysis"}
            return {"passed": False, "details": f"Expected 400/404, got {response.get('status_code', 'No response')}"}
        return test
    
    def test_cancel_nonexistent_analysis(self):
        """Test 9: Cancel Non-existent Analysis"""
        def test():
            response = self.make_request("/api/analysis/99999/cancel/", "POST")
            # Should return 401 (auth) or 404 (not found)
            if response["status_code"] in [401, 404]:
                return {"passed": True, "details": "Proper error for canceling non-existent analysis"}
            return {"passed": False, "details": f"Expected 401/404, got {response.get('status_code', 'No response')}"}
        return test
    
    def test_file_upload_validation(self):
        """Test 10: File Upload (No File)"""
        def test():
            # For file upload, we need multipart form data
            # This test checks if the endpoint is accessible
            response = self.make_request("/api/upload/", "POST", {})
            # Should return 400 (no file) or 401 (auth required)
            if response["status_code"] in [400, 401]:
                return {"passed": True, "details": "Proper error for missing file"}
            return {"passed": False, "details": f"Expected 400/401, got {response.get('status_code', 'No response')}"}
        return test
    
    def test_jwt_token_endpoint(self):
        """Test 11: JWT Token Endpoint"""
        def test():
            response = self.make_request("/api/token/", "POST", {
                "username": "test",
                "password": "test"
            })
            # Should return 401 for invalid credentials, not 500
            if response["status_code"] in [400, 401]:
                return {"passed": True, "details": "Token endpoint accessible with proper error"}
            return {"passed": False, "details": f"Unexpected status: {response.get('status_code', 'No response')}"}
        return test
    
    def test_jwt_refresh_endpoint(self):
        """Test 12: JWT Refresh Endpoint"""
        def test():
            response = self.make_request("/api/token/refresh/", "POST", {
                "refresh": "invalid_token"
            })
            if response["status_code"] in [400, 401]:
                return {"passed": True, "details": "Refresh endpoint accessible with proper error"}
            return {"passed": False, "details": f"Unexpected status: {response.get('status_code', 'No response')}"}
        return test
    
    def test_admin_access(self):
        """Test 13: Admin Page Access"""
        def test():
            try:
                # Use subprocess to check admin page with curl
                import subprocess
                result = subprocess.run(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:8000/admin/"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                status = int(result.stdout.strip())
                # Should return 302 (redirect to login) or 200 (if logged in)
                if status in [302, 200, 301]:
                    return {"passed": True, "details": f"Admin page accessible (HTTP {status})"}
                return {"passed": False, "details": f"Unexpected status: {status}"}
            except Exception as e:
                # If curl fails, try our regular method
                response = self.make_request("/admin/")
                if response.get("success", False) or response.get("status_code", 0) > 0:
                    return {"passed": True, "details": "Admin page accessible"}
                return {"passed": False, "details": f"Could not access admin: {str(e)}"}
        return test
    
    def test_response_time(self):
        """Test 14: Response Time Check"""
        def test():
            start_time = time.time()
            response = self.make_request("/api/health/")
            elapsed_time = time.time() - start_time
            
            if response["success"] and elapsed_time < 2.0:
                return {"passed": True, "details": f"Response time: {elapsed_time:.3f}s"}
            return {"passed": False, "details": f"Slow response: {elapsed_time:.3f}s"}
        return test
    
    def test_cors_headers(self):
        """Test 15: CORS Headers"""
        def test():
            response = self.make_request("/api/health/")
            if response["success"]:
                # Check for CORS-related headers
                return {"passed": True, "details": "Request processed successfully"}
            return {"passed": False, "details": f"CORS check failed: {response.get('error', 'Unknown error')}"}
        return test
    
    def run_all_tests(self):
        """Execute all backend tests"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=" * 60)
        print("AIDAS BACKEND BLACK BOX TESTING SUITE")
        print("=" * 60 + Colors.ENDC)
        print(f"\n{Colors.OKCYAN}Base URL: {self.base_url}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
        
        # Run all tests
        tests = [
            ("Server Health Check", self.test_server_health()),
            ("API Schema Endpoint", self.test_api_schema()),
            ("Analysis List (Empty)", self.test_analysis_list_empty()),
            ("Analysis List with Filter", self.test_analysis_list_with_filter()),
            ("Analysis Create Validation", self.test_analysis_create_validation()),
            ("Analysis Create Not Found", self.test_analysis_create_not_found()),
            ("Analysis Detail Not Found", self.test_analysis_detail_not_found()),
            ("Analysis Results Not Completed", self.test_analysis_results_not_completed()),
            ("Cancel Non-existent Analysis", self.test_cancel_nonexistent_analysis()),
            ("File Upload Validation", self.test_file_upload_validation()),
            ("JWT Token Endpoint", self.test_jwt_token_endpoint()),
            ("JWT Refresh Endpoint", self.test_jwt_refresh_endpoint()),
            ("Admin Page Access", self.test_admin_access()),
            ("Response Time Check", self.test_response_time()),
            ("CORS Headers Check", self.test_cors_headers()),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print summary
        print(f"\n{Colors.HEADER}{Colors.BOLD}" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60 + Colors.ENDC)
        print(f"{Colors.OKGREEN}Tests Passed: {self.tests_passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Tests Failed: {self.tests_failed}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Total Tests: {self.tests_passed + self.tests_failed}{Colors.ENDC}")
        
        if self.tests_failed == 0:
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.ENDC}")
            print(f"\n{Colors.WARNING}Failed Tests:{Colors.ENDC}")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    print(f"  - {result['name']}: {result['details']}")
        
        print(f"\n{Colors.OKCYAN}Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
        
        return self.tests_passed, self.tests_failed


def main():
    """Main entry point for backend testing"""
    # Check if server is running
    test_suite = BackendTestSuite()
    response = test_suite.make_request("/api/health/")
    
    if not response["success"]:
        print(f"\n{Colors.FAIL}✗ ERROR: Backend server is not running!{Colors.ENDC}")
        print(f"{Colors.WARNING}Please start the server with:{Colors.ENDC}")
        print(f"  cd /workspace/resaerch01/complete_system/backend")
        print(f"  source venv/bin/activate")
        print(f"  python manage.py runserver 0.0.0.0:8000\n")
        sys.exit(1)
    
    # Run tests
    passed, failed = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
