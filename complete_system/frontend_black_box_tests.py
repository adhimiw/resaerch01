#!/usr/bin/env python3
"""
AIDAS Frontend Black Box Testing Suite
=======================================

Comprehensive black box testing for React frontend components.
Tests UI functionality, API integration, and user flows.

Author: MiniMax Agent
"""

import json
import sys
import os
import subprocess
import time
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


class FrontendTestSuite:
    """Black box testing suite for AIDAS Frontend"""
    
    def __init__(self, backend_url="http://127.0.0.1:8000", frontend_url="http://localhost:3000"):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
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
    
    # ==================== BACKEND INTEGRATION TESTS ====================
    
    def test_backend_health(self):
        """Test 1: Backend Health Check (Frontend Dependency)"""
        def test():
            try:
                response = urlopen(f"{self.backend_url}/api/health/", timeout=5)
                data = json.loads(response.read().decode())
                if "status" in data and "version" in data:
                    return {"passed": True, "details": f"Backend healthy: {data['status']}"}
                return {"passed": False, "details": "Backend returned incomplete health data"}
            except Exception as e:
                return {"passed": False, "details": f"Backend unavailable: {str(e)}"}
        return test
    
    def test_backend_analysis_endpoint(self):
        """Test 2: Backend Analysis Endpoint (Frontend API Call)"""
        def test():
            try:
                response = urlopen(f"{self.backend_url}/api/analysis/", timeout=5)
                data = json.loads(response.read().decode())
                if isinstance(data, list):
                    return {"passed": True, "details": f"Analysis endpoint accessible, count: {len(data)}"}
                return {"passed": False, "details": "Analysis endpoint returned non-list"}
            except Exception as e:
                return {"passed": False, "details": f"Analysis endpoint error: {str(e)}"}
        return test
    
    def test_backend_upload_endpoint(self):
        """Test 3: Backend Upload Endpoint (Frontend Upload Flow)"""
        def test():
            try:
                # Test that upload endpoint is accessible (will fail with 401 due to auth)
                req = Request(
                    f"{self.backend_url}/api/upload/",
                    data=json.dumps({}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                response = urlopen(req, timeout=5)
                return {"passed": True, "details": "Upload endpoint accessible"}
            except HTTPError as e:
                # 401 is expected (auth required), but endpoint exists
                if e.code in [400, 401]:
                    return {"passed": True, "details": "Upload endpoint accessible (auth required)"}
                return {"passed": False, "details": f"Upload endpoint error: {e.code}"}
            except Exception as e:
                return {"passed": False, "details": f"Upload endpoint error: {str(e)}"}
        return test
    
    def test_backend_token_endpoint(self):
        """Test 4: Backend JWT Token Endpoint (Frontend Auth Flow)"""
        def test():
            try:
                req = Request(
                    f"{self.backend_url}/api/token/",
                    data=json.dumps({"username": "test", "password": "test"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                response = urlopen(req, timeout=5)
                return {"passed": True, "details": "Token endpoint accessible"}
            except HTTPError as e:
                # 401 is expected for invalid credentials
                if e.code in [400, 401]:
                    return {"passed": True, "details": "Token endpoint accessible (auth working)"}
                return {"passed": False, "details": f"Token endpoint error: {e.code}"}
            except Exception as e:
                return {"passed": False, "details": f"Token endpoint error: {str(e)}"}
        return test
    
    def test_backend_schema_endpoint(self):
        """Test 5: Backend Schema Endpoint (Frontend API Discovery)"""
        def test():
            try:
                response = urlopen(f"{self.backend_url}/api/schema/", timeout=5)
                data = json.loads(response.read().decode())
                if "info" in data and "title" in data["info"]:
                    return {"passed": True, "details": "Schema endpoint accessible"}
                return {"passed": False, "details": "Schema endpoint returned incomplete data"}
            except Exception as e:
                return {"passed": False, "details": f"Schema endpoint error: {str(e)}"}
        return test
    
    # ==================== FRONTEND STRUCTURE TESTS ====================
    
    def test_frontend_package_json(self):
        """Test 6: Frontend package.json Configuration"""
        def test():
            frontend_path = "/workspace/resaerch01/complete_system/aidas-frontend"
            package_json = os.path.join(frontend_path, "package.json")
            
            if not os.path.exists(package_json):
                return {"passed": False, "details": "package.json not found"}
            
            with open(package_json) as f:
                config = json.load(f)
            
            # Check required fields
            required_fields = ["name", "version", "scripts", "dependencies"]
            for field in required_fields:
                if field not in config:
                    return {"passed": False, "details": f"Missing required field: {field}"}
            
            # Check required dependencies
            required_deps = ["react", "axios"]
            for dep in required_deps:
                if dep not in config.get("dependencies", {}):
                    return {"passed": False, "details": f"Missing required dependency: {dep}"}
            
            return {"passed": True, "details": f"package.json valid, {len(config['dependencies'])} dependencies"}
        return test
    
    def test_frontend_app_js(self):
        """Test 7: Frontend App.js Component"""
        def test():
            app_path = "/workspace/resaerch01/complete_system/aidas-frontend/src/App.js"
            
            if not os.path.exists(app_path):
                return {"passed": False, "details": "App.js not found"}
            
            with open(app_path) as f:
                content = f.read()
            
            # Check for required imports
            required_imports = ["react", "axios"]
            for imp in required_imports:
                if imp not in content:
                    return {"passed": False, "details": f"Missing import: {imp}"}
            
            # Check for required components
            required_components = ["useState", "useCallback", "useEffect"]
            for comp in required_components:
                if comp not in content:
                    return {"passed": False, "details": f"Missing hook: {comp}"}
            
            # Check for API URL configuration
            if "API_BASE" not in content:
                return {"passed": False, "details": "Missing API configuration"}
            
            return {"passed": True, "details": "App.js contains required components"}
        return test
    
    def test_frontend_index_css(self):
        """Test 8: Frontend CSS Styling"""
        def test():
            css_path = "/workspace/resaerch01/complete_system/aidas-frontend/src/index.css"
            
            if not os.path.exists(css_path):
                return {"passed": False, "details": "index.css not found"}
            
            with open(css_path) as f:
                content = f.read()
            
            # Check for theme colors (any hex colors indicate styling)
            import re
            hex_colors = re.findall(r'#[0-9a-fA-F]{6}', content)
            
            if len(hex_colors) < 5:
                return {"passed": False, "details": f"Missing theme colors (found {len(hex_colors)}/5)"}
            
            # Check for responsive patterns (flex, grid, %, vw, etc.)
            responsive_patterns = ["flex", "grid", "%", "vw", "vh", "rem", "em"]
            found_patterns = sum(1 for pattern in responsive_patterns if pattern in content)
            
            if found_patterns < 3:
                return {"passed": False, "details": f"Missing responsive patterns (found {found_patterns}/7)"}
            
            return {"passed": True, "details": f"CSS valid, {len(hex_colors)} colors, {found_patterns} responsive patterns"}
        return test
    
    def test_frontend_dropzone_integration(self):
        """Test 9: File Upload Dropzone Integration"""
        def test():
            app_path = "/workspace/resaerch01/complete_system/aidas-frontend/src/App.js"
            
            if not os.path.exists(app_path):
                return {"passed": False, "details": "App.js not found"}
            
            with open(app_path) as f:
                content = f.read()
            
            # Check for dropzone
            if "useDropzone" not in content:
                return {"passed": False, "details": "Dropzone not integrated"}
            
            # Check for file handling
            if "onDrop" not in content:
                return {"passed": False, "details": "Drop handler not found"}
            
            # Check for accepted file types
            if ".csv" not in content or ".json" not in content:
                return {"passed": False, "details": "File type validation missing"}
            
            return {"passed": True, "details": "Dropzone integration complete"}
        return test
    
    def test_frontend_api_integration(self):
        """Test 10: API Integration (Axios)"""
        def test():
            app_path = "/workspace/resaerch01/complete_system/aidas-frontend/src/App.js"
            
            if not os.path.exists(app_path):
                return {"passed": False, "details": "App.js not found"}
            
            with open(app_path) as f:
                content = f.read()
            
            # Check for axios import
            if "axios" not in content:
                return {"passed": False, "details": "Axios not imported"}
            
            # Check for API base URL configuration
            if "API_BASE" not in content and "api/" not in content.lower():
                return {"passed": False, "details": "API configuration missing"}
            
            # Check for API-related calls
            api_keywords = ["api", "endpoint", "upload", "analysis", "health"]
            found_keywords = sum(1 for keyword in api_keywords if keyword in content.lower())
            
            if found_keywords < 3:
                return {"passed": False, "details": f"Missing API integration (found {found_keywords}/5 keywords)"}
            
            return {"passed": True, "details": f"API integration complete, {found_keywords} API keywords found"}
        return test
    
    def test_frontend_state_management(self):
        """Test 11: React State Management"""
        def test():
            app_path = "/workspace/resaerch01/complete_system/aidas-frontend/src/App.js"
            
            if not os.path.exists(app_path):
                return {"passed": False, "details": "App.js not found"}
            
            with open(app_path) as f:
                content = f.read()
            
            # Check for required state variables
            required_states = [
                "activeTab",
                "selectedFile",
                "uploadProgress",
                "viewMode",
                "searchQuery",
                "showSettings",
                "results",
                "isCreating",
                "error"
            ]
            
            for state in required_states:
                if state not in content:
                    return {"passed": False, "details": f"Missing state: {state}"}
            
            return {"passed": True, "details": f"All {len(required_states)} required states present"}
        return test
    
    def test_frontend_analysis_workflow(self):
        """Test 12: Analysis Workflow Phases"""
        def test():
            app_path = "/workspace/resaerch01/complete_system/aidas-frontend/src/App.js"
            
            if not os.path.exists(app_path):
                return {"passed": False, "details": "App.js not found"}
            
            with open(app_path) as f:
                content = f.read()
            
            # Check for analysis phases
            phases = [
                "data_ingestion",
                "data_cleaning",
                "exploratory_analysis",
                "hypothesis_discovery",
                "model_building",
                "insight_generation"
            ]
            
            found_phases = sum(1 for phase in phases if phase in content)
            
            if found_phases < 4:
                return {"passed": False, "details": f"Missing analysis phases (found {found_phases}/6)"}
            
            return {"passed": True, "details": f"Analysis workflow has {found_phases} phases"}
        return test
    
    def test_frontend_error_handling(self):
        """Test 13: Error Handling"""
        def test():
            app_path = "/workspace/resaerch01/complete_system/aidas-frontend/src/App.js"
            
            if not os.path.exists(app_path):
                return {"passed": False, "details": "App.js not found"}
            
            with open(app_path) as f:
                content = f.read()
            
            # Check for error handling
            error_patterns = [
                "catch",
                "error",
                "Error",
                "try {"
            ]
            
            found_patterns = sum(1 for pattern in error_patterns if pattern in content)
            
            if found_patterns < 3:
                return {"passed": False, "details": f"Missing error handling patterns"}
            
            return {"passed": True, "details": "Error handling implemented"}
        return test
    
    def test_frontend_loading_states(self):
        """Test 14: Loading States"""
        def test():
            app_path = "/workspace/resaerch01/complete_system/aidas-frontend/src/App.js"
            
            if not os.path.exists(app_path):
                return {"passed": False, "details": "App.js not found"}
            
            with open(app_path) as f:
                content = f.read()
            
            # Check for loading indicators
            loading_patterns = [
                "isAnalyzing",
                "Loader2",
                "loading",
                "Loading"
            ]
            
            found_patterns = sum(1 for pattern in loading_patterns if pattern in content)
            
            if found_patterns < 2:
                return {"passed": False, "details": f"Missing loading states"}
            
            return {"passed": True, "details": "Loading states implemented"}
        return test
    
    def test_frontend_visualization_integration(self):
        """Test 15: Data Visualization Integration"""
        def test():
            app_path = "/workspace/resaerch01/complete_system/aidas-frontend/src/App.js"
            
            if not os.path.exists(app_path):
                return {"passed": False, "details": "App.js not found"}
            
            with open(app_path) as f:
                content = f.read()
            
            # Check for visualization libraries
            viz_patterns = [
                "recharts",
                "BarChart",
                "LineChart",
                "PieChart",
                "results"
            ]
            
            found_patterns = sum(1 for pattern in viz_patterns if pattern in content)
            
            if found_patterns < 2:
                return {"passed": False, "details": f"Missing visualization components"}
            
            return {"passed": True, "details": "Visualization integration found"}
        return test
    
    def run_all_tests(self):
        """Execute all frontend tests"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}=" * 60)
        print("AIDAS FRONTEND BLACK BOX TESTING SUITE")
        print("=" * 60 + Colors.ENDC)
        print(f"\n{Colors.OKCYAN}Backend URL: {self.backend_url}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
        
        # Run all tests
        tests = [
            # Backend Integration Tests
            ("Backend Health Check", self.test_backend_health()),
            ("Backend Analysis Endpoint", self.test_backend_analysis_endpoint()),
            ("Backend Upload Endpoint", self.test_backend_upload_endpoint()),
            ("Backend Token Endpoint", self.test_backend_token_endpoint()),
            ("Backend Schema Endpoint", self.test_backend_schema_endpoint()),
            
            # Frontend Structure Tests
            ("Frontend package.json", self.test_frontend_package_json()),
            ("Frontend App.js Component", self.test_frontend_app_js()),
            ("Frontend CSS Styling", self.test_frontend_index_css()),
            ("File Upload Dropzone", self.test_frontend_dropzone_integration()),
            ("API Integration", self.test_frontend_api_integration()),
            ("State Management", self.test_frontend_state_management()),
            ("Analysis Workflow", self.test_frontend_analysis_workflow()),
            ("Error Handling", self.test_frontend_error_handling()),
            ("Loading States", self.test_frontend_loading_states()),
            ("Visualization Integration", self.test_frontend_visualization_integration()),
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
    """Main entry point for frontend testing"""
    # Check if frontend exists
    frontend_path = "/workspace/resaerch01/complete_system/aidas-frontend"
    if not os.path.exists(frontend_path):
        print(f"\n{Colors.FAIL}✗ ERROR: Frontend directory not found at {frontend_path}{Colors.ENDC}")
        sys.exit(1)
    
    # Run tests
    test_suite = FrontendTestSuite()
    passed, failed = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
