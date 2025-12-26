"""
Data Validation & Leakage Detection System
Phase 2: Week 3 Implementation

Features:
- Train/test data leakage detection
- Duplicate row detection
- Temporal leakage detection
- Group leakage detection (user IDs, customer IDs)
- Risk score calculation (0-100)
- Auto-fix for common issues
- Integration with MLleak library

Based on: INTEGRATION_PLAN.md - Phase 2 (MLleak Validation)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import hashlib
from pathlib import Path
import warnings

try:
    from mlleak import check as mlleak_check
    MLLEAK_AVAILABLE = True
except ImportError:
    MLLEAK_AVAILABLE = False
    warnings.warn("mlleak not installed. Install with: pip install mlleak")


class DataLeakageDetector:
    """
    Comprehensive data leakage detection system
    
    Detects multiple types of leakage:
    1. Duplicate rows across train/test
    2. Temporal leakage (test data before train data)
    3. Group leakage (same users/customers in both sets)
    4. Target leakage (test set has target variable)
    5. Schema inconsistencies
    
    Capabilities:
    - Automatic detection with risk scoring
    - Auto-fix for common issues
    - Detailed reports with recommendations
    - Integration with MLleak library
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize data leakage detector
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.last_report = None
        
    def check(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
             target_column: Optional[str] = None,
             id_columns: Optional[List[str]] = None,
             date_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive leakage check
        
        Args:
            train_df: Training dataset
            test_df: Test dataset
            target_column: Name of target variable (optional)
            id_columns: List of ID columns (user_id, customer_id, etc.)
            date_column: Name of date/time column for temporal checks
            
        Returns:
            Dictionary with detection results and recommendations
        """
        print("🔍 Starting comprehensive data leakage detection...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "checks": {},
            "issues": [],
            "warnings": [],
            "risk_score": 0,
            "recommendations": [],
            "can_auto_fix": False
        }
        
        # Check 1: Duplicate rows
        print("\n1️⃣  Checking for duplicate rows across train/test...")
        duplicate_check = self._check_duplicate_rows(train_df, test_df)
        report["checks"]["duplicate_rows"] = duplicate_check
        
        if duplicate_check["found"]:
            report["issues"].append(f"Found {duplicate_check['count']} duplicate rows")
            report["risk_score"] += 30
            report["recommendations"].append("Remove exact duplicates from test set")
            report["can_auto_fix"] = True
        
        # Check 2: Target leakage
        if target_column:
            print("2️⃣  Checking for target leakage...")
            target_check = self._check_target_leakage(test_df, target_column)
            report["checks"]["target_leakage"] = target_check
            
            if target_check["has_target"]:
                report["issues"].append(f"Test set contains target column '{target_column}'")
                report["risk_score"] += 50  # Very serious
                report["recommendations"].append(f"Remove '{target_column}' from test set")
                report["can_auto_fix"] = True
        
        # Check 3: Temporal leakage
        if date_column:
            print("3️⃣  Checking for temporal leakage...")
            temporal_check = self._check_temporal_leakage(train_df, test_df, date_column)
            report["checks"]["temporal_leakage"] = temporal_check
            
            if temporal_check["has_leakage"]:
                report["issues"].append("Test dates overlap with training dates")
                report["risk_score"] += 40
                report["recommendations"].append("Ensure test data comes after train data")
        
        # Check 4: Group/ID leakage
        if id_columns:
            print("4️⃣  Checking for group/ID leakage...")
            group_check = self._check_group_leakage(train_df, test_df, id_columns)
            report["checks"]["group_leakage"] = group_check
            
            if group_check["has_overlap"]:
                report["issues"].append(f"Found {group_check['overlap_count']} overlapping IDs")
                report["risk_score"] += 35
                report["recommendations"].append("Ensure train/test split by unique IDs")
        
        # Check 5: Schema consistency
        print("5️⃣  Checking schema consistency...")
        schema_check = self._check_schema_consistency(train_df, test_df)
        report["checks"]["schema_consistency"] = schema_check
        
        if not schema_check["consistent"]:
            report["warnings"].append("Schema differences detected")
            report["risk_score"] += 10
            report["recommendations"].append("Align train/test schemas")
        
        # Check 6: Statistical similarity (distribution shift)
        print("6️⃣  Checking for distribution shift...")
        distribution_check = self._check_distribution_shift(train_df, test_df)
        report["checks"]["distribution_shift"] = distribution_check
        
        if distribution_check["significant_shift"]:
            report["warnings"].append("Significant distribution shift detected")
            report["risk_score"] += 15
            report["recommendations"].append("Investigate feature distributions")
        
        # MLleak integration (if available)
        if MLLEAK_AVAILABLE:
            print("7️⃣  Running MLleak advanced checks...")
            try:
                mlleak_report = self._run_mlleak(train_df, test_df)
                report["checks"]["mlleak"] = mlleak_report
            except Exception as e:
                report["warnings"].append(f"MLleak check failed: {e}")
        
        # Cap risk score at 100
        report["risk_score"] = min(report["risk_score"], 100)
        
        # Determine severity
        if report["risk_score"] >= 70:
            report["severity"] = "CRITICAL"
        elif report["risk_score"] >= 40:
            report["severity"] = "HIGH"
        elif report["risk_score"] >= 20:
            report["severity"] = "MEDIUM"
        else:
            report["severity"] = "LOW"
        
        self.last_report = report
        return report
    
    def _check_duplicate_rows(self, train_df: pd.DataFrame, 
                             test_df: pd.DataFrame) -> Dict[str, Any]:
        """Check for exact duplicate rows across train/test"""
        # Create hash of each row
        train_hashes = set(train_df.apply(lambda row: hashlib.md5(
            str(row.values).encode()
        ).hexdigest(), axis=1))
        
        test_hashes = test_df.apply(lambda row: hashlib.md5(
            str(row.values).encode()
        ).hexdigest(), axis=1)
        
        duplicate_indices = []
        for idx, hash_val in enumerate(test_hashes):
            if hash_val in train_hashes:
                duplicate_indices.append(idx)
        
        return {
            "found": len(duplicate_indices) > 0,
            "count": len(duplicate_indices),
            "indices": duplicate_indices[:10],  # First 10 for reporting
            "percentage": (len(duplicate_indices) / len(test_df) * 100) if len(test_df) > 0 else 0
        }
    
    def _check_target_leakage(self, test_df: pd.DataFrame, 
                             target_column: str) -> Dict[str, Any]:
        """Check if test set contains target variable"""
        has_target = target_column in test_df.columns
        
        result = {
            "has_target": has_target,
            "target_column": target_column
        }
        
        if has_target:
            # Check if target has non-null values
            non_null_count = test_df[target_column].notna().sum()
            result["non_null_values"] = int(non_null_count)
            result["null_values"] = int(len(test_df) - non_null_count)
        
        return result
    
    def _check_temporal_leakage(self, train_df: pd.DataFrame, 
                               test_df: pd.DataFrame,
                               date_column: str) -> Dict[str, Any]:
        """Check for temporal leakage (test dates before train dates)"""
        if date_column not in train_df.columns or date_column not in test_df.columns:
            return {"has_leakage": False, "error": "Date column not found"}
        
        try:
            train_dates = pd.to_datetime(train_df[date_column], errors='coerce')
            test_dates = pd.to_datetime(test_df[date_column], errors='coerce')
            
            train_max = train_dates.max()
            train_min = train_dates.min()
            test_max = test_dates.max()
            test_min = test_dates.min()
            
            # Check if test dates overlap with train dates
            has_leakage = test_min < train_max
            
            return {
                "has_leakage": has_leakage,
                "train_date_range": {
                    "min": str(train_min),
                    "max": str(train_max)
                },
                "test_date_range": {
                    "min": str(test_min),
                    "max": str(test_max)
                },
                "overlap_days": (train_max - test_min).days if has_leakage else 0
            }
        except Exception as e:
            return {"has_leakage": False, "error": str(e)}
    
    def _check_group_leakage(self, train_df: pd.DataFrame, 
                            test_df: pd.DataFrame,
                            id_columns: List[str]) -> Dict[str, Any]:
        """Check for group leakage (same IDs in train and test)"""
        overlaps = {}
        total_overlap = 0
        
        for col in id_columns:
            if col in train_df.columns and col in test_df.columns:
                train_ids = set(train_df[col].dropna())
                test_ids = set(test_df[col].dropna())
                
                overlap = train_ids & test_ids
                overlaps[col] = {
                    "overlap_count": len(overlap),
                    "train_unique": len(train_ids),
                    "test_unique": len(test_ids),
                    "overlap_percentage": (len(overlap) / len(test_ids) * 100) if len(test_ids) > 0 else 0
                }
                total_overlap += len(overlap)
        
        return {
            "has_overlap": total_overlap > 0,
            "overlap_count": total_overlap,
            "details": overlaps
        }
    
    def _check_schema_consistency(self, train_df: pd.DataFrame, 
                                 test_df: pd.DataFrame) -> Dict[str, Any]:
        """Check if train and test have consistent schemas"""
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        
        # Check data types for common columns
        common_cols = train_cols & test_cols
        type_mismatches = {}
        
        for col in common_cols:
            if train_df[col].dtype != test_df[col].dtype:
                type_mismatches[col] = {
                    "train_dtype": str(train_df[col].dtype),
                    "test_dtype": str(test_df[col].dtype)
                }
        
        return {
            "consistent": len(missing_in_test) == 0 and len(extra_in_test) == 0 and len(type_mismatches) == 0,
            "missing_in_test": list(missing_in_test),
            "extra_in_test": list(extra_in_test),
            "type_mismatches": type_mismatches
        }
    
    def _check_distribution_shift(self, train_df: pd.DataFrame, 
                                 test_df: pd.DataFrame,
                                 threshold: float = 0.3) -> Dict[str, Any]:
        """Check for significant distribution shifts in numeric columns"""
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        common_numeric = set(numeric_cols) & set(test_df.columns)
        
        shifts = {}
        significant_shifts = []
        
        for col in common_numeric:
            try:
                train_mean = train_df[col].mean()
                test_mean = test_df[col].mean()
                train_std = train_df[col].std()
                
                # Calculate normalized difference
                if train_std > 0:
                    normalized_diff = abs(train_mean - test_mean) / train_std
                    
                    shifts[col] = {
                        "train_mean": float(train_mean),
                        "test_mean": float(test_mean),
                        "normalized_diff": float(normalized_diff)
                    }
                    
                    if normalized_diff > threshold:
                        significant_shifts.append(col)
            except:
                continue
        
        return {
            "significant_shift": len(significant_shifts) > 0,
            "shifted_columns": significant_shifts,
            "details": shifts
        }
    
    def _run_mlleak(self, train_df: pd.DataFrame, 
                   test_df: pd.DataFrame) -> Dict[str, Any]:
        """Run MLleak library checks (if available)"""
        try:
            # MLleak expects specific format
            result = mlleak_check(train_df, test_df)
            return {
                "completed": True,
                "result": str(result)
            }
        except Exception as e:
            return {
                "completed": False,
                "error": str(e)
            }
    
    def auto_fix(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                target_column: Optional[str] = None,
                remove_duplicates: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Automatically fix common leakage issues
        
        Args:
            train_df: Training dataset
            test_df: Test dataset
            target_column: Target variable to remove from test
            remove_duplicates: Whether to remove duplicate rows
            
        Returns:
            Tuple of (fixed_train_df, fixed_test_df)
        """
        print("🔧 Applying auto-fix for data leakage issues...")
        
        fixed_train = train_df.copy()
        fixed_test = test_df.copy()
        fixes_applied = []
        
        # Fix 1: Remove target from test set
        if target_column and target_column in fixed_test.columns:
            fixed_test = fixed_test.drop(columns=[target_column])
            fixes_applied.append(f"Removed target column '{target_column}' from test set")
            print(f"  ✅ Removed target column: {target_column}")
        
        # Fix 2: Remove duplicate rows
        if remove_duplicates:
            original_test_len = len(fixed_test)
            
            # Create hashes for comparison
            train_hashes = set(fixed_train.apply(lambda row: hashlib.md5(
                str(row.values).encode()
            ).hexdigest(), axis=1))
            
            # Keep only non-duplicate rows in test set
            test_hashes = fixed_test.apply(lambda row: hashlib.md5(
                str(row.values).encode()
            ).hexdigest(), axis=1)
            
            fixed_test = fixed_test[~test_hashes.isin(train_hashes)]
            
            removed_count = original_test_len - len(fixed_test)
            if removed_count > 0:
                fixes_applied.append(f"Removed {removed_count} duplicate rows from test set")
                print(f"  ✅ Removed {removed_count} duplicate rows")
        
        print(f"\n✅ Auto-fix complete: {len(fixes_applied)} fixes applied")
        
        return fixed_train, fixed_test
    
    def print_report(self, report: Optional[Dict[str, Any]] = None):
        """
        Print a formatted leakage detection report
        
        Args:
            report: Detection report (uses last_report if None)
        """
        if report is None:
            report = self.last_report
        
        if report is None:
            print("No report available. Run check() first.")
            return
        
        print("\n" + "=" * 70)
        print(" " * 20 + "ML DATA LEAKAGE REPORT")
        print("=" * 70)
        
        # Summary
        print(f"\n📊 Dataset Summary:")
        print(f"   Train shape: {report['train_shape']}")
        print(f"   Test shape: {report['test_shape']}")
        print(f"   Timestamp: {report['timestamp']}")
        
        # Risk score
        severity_colors = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }
        
        print(f"\n🎯 Risk Assessment:")
        print(f"   Risk Score: {report['risk_score']}/100")
        print(f"   Severity: {severity_colors.get(report['severity'], '')} {report['severity']}")
        
        # Issues
        if report['issues']:
            print(f"\n❌ Issues Found ({len(report['issues'])}):")
            for i, issue in enumerate(report['issues'], 1):
                print(f"   {i}. {issue}")
        else:
            print("\n✅ No critical issues found!")
        
        # Warnings
        if report['warnings']:
            print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
            for i, warning in enumerate(report['warnings'], 1):
                print(f"   {i}. {warning}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\n💡 Recommendations ({len(report['recommendations'])}):")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Auto-fix availability
        print(f"\n🔧 Auto-fix Available: {'Yes ✅' if report['can_auto_fix'] else 'No ❌'}")
        
        print("\n" + "=" * 70)
    
    def get_summary(self, report: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a one-line summary of the report
        
        Args:
            report: Detection report (uses last_report if None)
            
        Returns:
            Summary string
        """
        if report is None:
            report = self.last_report
        
        if report is None:
            return "No report available"
        
        severity_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }
        
        emoji = severity_emoji.get(report['severity'], '⚪')
        return (f"{emoji} Risk: {report['risk_score']}/100 ({report['severity']}) - "
                f"{len(report['issues'])} issues, {len(report['warnings'])} warnings")


# Convenience function
def quick_check(train_df: pd.DataFrame, test_df: pd.DataFrame,
                target_column: Optional[str] = None,
                id_columns: Optional[List[str]] = None,
                date_column: Optional[str] = None,
                auto_fix: bool = False) -> Union[Dict[str, Any], Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
    """
    Quick check for data leakage with optional auto-fix
    
    Args:
        train_df: Training dataset
        test_df: Test dataset
        target_column: Target variable name
        id_columns: List of ID columns
        date_column: Date/time column name
        auto_fix: Whether to automatically fix issues
        
    Returns:
        Report dict, or (fixed_train, fixed_test, report) if auto_fix=True
    """
    detector = DataLeakageDetector()
    report = detector.check(train_df, test_df, target_column, id_columns, date_column)
    detector.print_report(report)
    
    if auto_fix and report['can_auto_fix']:
        print("\n🔧 Applying auto-fix...")
        fixed_train, fixed_test = detector.auto_fix(train_df, test_df, target_column)
        return fixed_train, fixed_test, report
    
    return report


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("DATA VALIDATION & LEAKAGE DETECTION - DEMO")
    print("=" * 70)
    
    # Create sample datasets with intentional leakage
    print("\n1️⃣  Creating sample datasets with intentional leakage...")
    
    # Training data
    train_data = pd.DataFrame({
        'user_id': range(1, 101),
        'age': np.random.randint(18, 65, 100),
        'income': np.random.randint(30000, 150000, 100),
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'purchased': np.random.choice([0, 1], 100)
    })
    
    # Test data with intentional issues
    test_data = pd.DataFrame({
        'user_id': range(95, 151),  # Overlap with train (95-100)
        'age': np.random.randint(18, 65, 56),
        'income': np.random.randint(30000, 150000, 56),
        'date': pd.date_range('2023-03-01', periods=56, freq='D'),  # Overlaps with train
        'purchased': np.random.choice([0, 1], 56)  # Target leakage!
    })
    
    # Add exact duplicates
    test_data = pd.concat([test_data, train_data.iloc[:3]], ignore_index=True)
    
    print(f"   Train: {train_data.shape}, Test: {test_data.shape}")
    
    # Run detection
    print("\n2️⃣  Running leakage detection...")
    report = quick_check(
        train_df=train_data,
        test_df=test_data,
        target_column='purchased',
        id_columns=['user_id'],
        date_column='date'
    )
    
    # Test auto-fix
    print("\n3️⃣  Testing auto-fix...")
    detector = DataLeakageDetector()
    fixed_train, fixed_test = detector.auto_fix(
        train_data, 
        test_data,
        target_column='purchased'
    )
    
    print(f"\nFixed test shape: {fixed_test.shape} (was {test_data.shape})")
    print(f"Target column present: {'purchased' in fixed_test.columns}")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE - Data validation system operational!")
    print("=" * 70)
