"""
Multi-File Dataset Handler
Detects relationships between uploaded files (Kaggle-style)

Handles:
- train.csv + test.csv + sample_submission.csv (Kaggle competitions)
- customers.csv + transactions.csv + products.csv (relational data)
- images/ + annotations.json + metadata.csv (multi-modal datasets)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import json
import os
from datetime import datetime


class MultiFileDatasetHandler:
    """Handle multi-file dataset uploads with relationship detection"""
    
    def __init__(self, temp_dir: str = '/tmp'):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.file_relationships = []
        self.detected_context = {}
    
    async def parse_files(self, files: Dict) -> Dict:
        """
        Parse uploaded files and detect relationships
        
        Args:
            files: {
                'train': UploadFile,
                'test': UploadFile,
                'metadata': UploadFile,
                'additional': List[UploadFile]
            }
        
        Returns:
            dataset_context: {
                'type': 'kaggle_competition' | 'relational' | 'multi_modal',
                'files': {...},
                'relationships': [...],
                'recommended_strategy': str
            }
        """
        
        context = {
            'type': 'unknown',
            'files': {},
            'relationships': [],
            'recommended_strategy': '',
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Detect dataset type
        if files.get('train') and files.get('test'):
            context['type'] = 'kaggle_competition'
            context['recommended_strategy'] = 'train_test_split_analysis'
        elif len([k for k in files.keys() if files.get(k)]) > 2:
            context['type'] = 'relational'
            context['recommended_strategy'] = 'relational_join_analysis'
        
        # 2. Parse each file
        for key, file in files.items():
            if file is None or (isinstance(file, list) and len(file) == 0):
                continue
            
            if isinstance(file, list):
                # Handle additional files
                for idx, f in enumerate(file):
                    file_info = await self._parse_single_file(f)
                    context['files'][f'{key}_{idx}'] = file_info
            else:
                file_info = await self._parse_single_file(file)
                context['files'][key] = file_info
        
        # 3. Detect relationships
        context['relationships'] = self._detect_relationships(context['files'])
        
        # 4. Validate schema consistency
        if context['type'] == 'kaggle_competition':
            context['schema_consistency'] = self._check_schema_consistency(
                context['files'].get('train'),
                context['files'].get('test')
            )
        
        # 5. Generate dataset summary
        context['summary'] = self._generate_summary(context)
        
        return context
    
    async def _parse_single_file(self, file) -> Dict:
        """Parse single file and extract metadata"""
        
        # Read file content
        content = await file.read()
        file_path = self.temp_dir / file.filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        info = {
            'filename': file.filename,
            'size_bytes': len(content),
            'size_mb': round(len(content) / (1024 * 1024), 2),
            'path': str(file_path),
            'extension': Path(file.filename).suffix
        }
        
        # Parse CSV
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            info.update({
                'type': 'csv',
                'rows': len(df),
                'columns': list(df.columns),
                'column_count': len(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'sample': df.head(3).to_dict(),
                'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
            })
        
        # Parse JSON
        elif file.filename.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            info.update({
                'type': 'json',
                'keys': list(data.keys()) if isinstance(data, dict) else None,
                'is_list': isinstance(data, list),
                'item_count': len(data) if isinstance(data, (list, dict)) else None
            })
        
        # Parse text/other
        else:
            info.update({
                'type': 'unknown',
                'content_preview': content[:500].decode('utf-8', errors='ignore')
            })
        
        return info
    
    def _detect_relationships(self, files: Dict) -> List[Dict]:
        """Detect relationships between files"""
        
        relationships = []
        
        # Check for common columns (foreign keys)
        csv_files = {k: v for k, v in files.items() if v.get('type') == 'csv'}
        
        if len(csv_files) >= 2:
            file_pairs = [(k1, k2) for k1 in csv_files.keys() 
                          for k2 in csv_files.keys() if k1 < k2]
            
            for file1, file2 in file_pairs:
                common_cols = set(csv_files[file1]['columns']) & set(csv_files[file2]['columns'])
                
                if common_cols:
                    # Determine if likely foreign key relationship
                    potential_fks = self._identify_foreign_keys(
                        csv_files[file1], 
                        csv_files[file2], 
                        common_cols
                    )
                    
                    relationships.append({
                        'type': 'common_columns',
                        'files': [file1, file2],
                        'columns': list(common_cols),
                        'potential_foreign_keys': potential_fks,
                        'join_strategy': f'merge on {potential_fks[0]}' if potential_fks else 'merge on common keys',
                        'cardinality': self._estimate_cardinality(csv_files[file1], csv_files[file2], common_cols)
                    })
        
        # Check for Kaggle-style relationships
        if 'train' in files and 'test' in files:
            relationships.append({
                'type': 'kaggle_train_test',
                'files': ['train', 'test'],
                'description': 'Kaggle competition train/test split',
                'strategy': 'Train model on train.csv, predict on test.csv'
            })
        
        return relationships
    
    def _identify_foreign_keys(self, file1_info: Dict, file2_info: Dict, 
                                common_cols: set) -> List[str]:
        """Identify likely foreign key columns"""
        
        potential_fks = []
        
        for col in common_cols:
            # Heuristics for FK detection:
            # 1. Column name ends with 'id' or '_id'
            # 2. Column is not a common name (not 'name', 'date', etc.)
            col_lower = col.lower()
            
            if ('id' in col_lower or 
                col_lower.endswith('_key') or
                col_lower.endswith('_code')):
                potential_fks.append(col)
        
        return potential_fks
    
    def _estimate_cardinality(self, file1_info: Dict, file2_info: Dict, 
                              common_cols: set) -> str:
        """Estimate relationship cardinality (1:1, 1:N, N:M)"""
        
        # Simple heuristic based on row counts
        rows1 = file1_info.get('rows', 0)
        rows2 = file2_info.get('rows', 0)
        
        ratio = rows1 / rows2 if rows2 > 0 else 0
        
        if 0.8 < ratio < 1.2:
            return '1:1 (likely)'
        elif ratio > 1.5:
            return '1:N (file1 has many rows)'
        elif ratio < 0.67:
            return 'N:1 (file2 has many rows)'
        else:
            return 'N:M (many-to-many possible)'
    
    def _check_schema_consistency(self, train_info: Dict, test_info: Dict) -> Dict:
        """Check if train/test have consistent schemas"""
        
        if not train_info or not test_info:
            return {'consistent': False, 'reason': 'Missing file'}
        
        train_cols = set(train_info['columns'])
        test_cols = set(test_info['columns'])
        
        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        
        # Check data types consistency
        common_cols = train_cols & test_cols
        dtype_mismatches = []
        
        for col in common_cols:
            train_dtype = train_info['dtypes'].get(col)
            test_dtype = test_info['dtypes'].get(col)
            
            if train_dtype != test_dtype:
                dtype_mismatches.append({
                    'column': col,
                    'train_dtype': train_dtype,
                    'test_dtype': test_dtype
                })
        
        return {
            'consistent': len(missing_in_test) <= 1 and len(dtype_mismatches) == 0,
            'missing_in_test': list(missing_in_test),
            'extra_in_test': list(extra_in_test),
            'dtype_mismatches': dtype_mismatches,
            'warning': 'Target column missing from test is expected' if len(missing_in_test) == 1 else None,
            'target_column_candidate': list(missing_in_test)[0] if len(missing_in_test) == 1 else None
        }
    
    def _generate_summary(self, context: Dict) -> Dict:
        """Generate human-readable summary"""
        
        total_rows = sum(f.get('rows', 0) for f in context['files'].values() 
                        if f.get('type') == 'csv')
        total_columns = sum(f.get('column_count', 0) for f in context['files'].values() 
                           if f.get('type') == 'csv')
        total_size_mb = sum(f.get('size_mb', 0) for f in context['files'].values())
        
        return {
            'file_count': len(context['files']),
            'total_rows': total_rows,
            'total_columns': total_columns,
            'total_size_mb': round(total_size_mb, 2),
            'dataset_type': context['type'],
            'has_relationships': len(context['relationships']) > 0,
            'relationship_count': len(context['relationships'])
        }
    
    def get_kaggle_analysis_plan(self, context: Dict) -> Dict:
        """Generate Kaggle-specific analysis plan"""
        
        if context['type'] != 'kaggle_competition':
            return {}
        
        train_info = context['files'].get('train', {})
        test_info = context['files'].get('test', {})
        
        # Detect target column
        target_col = None
        if context.get('schema_consistency'):
            target_col = context['schema_consistency'].get('target_column_candidate')
        
        features = [c for c in train_info.get('columns', []) if c != target_col]
        
        return {
            'competition_type': 'kaggle',
            'target_column': target_col,
            'train_rows': train_info.get('rows'),
            'test_rows': test_info.get('rows'),
            'feature_count': len(features),
            'features': features,
            'numeric_features': [c for c in features if c in train_info.get('numeric_columns', [])],
            'categorical_features': [c for c in features if c in train_info.get('categorical_columns', [])],
            'recommended_models': [
                'XGBoost (tabular gold standard)',
                'LightGBM (faster alternative)',
                'CatBoost (handles categoricals)',
                'Neural Network (if enough data)'
            ],
            'validation_strategy': 'StratifiedKFold (5-fold)',
            'preprocessing_steps': [
                'Handle missing values',
                'Encode categorical features',
                'Feature engineering',
                'Scale/normalize if needed',
                'Create train/validation split'
            ],
            'expected_output': 'submission.csv with predictions for test set'
        }


def parse_files_sync(files: Dict[str, str]) -> Dict:
    """
    Synchronous version for local file paths
    
    Args:
        files: {
            'train': 'path/to/train.csv',
            'test': 'path/to/test.csv',
            ...
        }
    """
    
    handler = MultiFileDatasetHandler()
    context = {
        'type': 'unknown',
        'files': {},
        'relationships': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Detect dataset type
    if files.get('train') and files.get('test'):
        context['type'] = 'kaggle_competition'
    
    # Parse each file
    for key, filepath in files.items():
        if not filepath or not os.path.exists(filepath):
            continue
        
        file_info = _parse_file_sync(filepath)
        context['files'][key] = file_info
    
    # Detect relationships
    context['relationships'] = handler._detect_relationships(context['files'])
    
    # Schema consistency
    if context['type'] == 'kaggle_competition':
        context['schema_consistency'] = handler._check_schema_consistency(
            context['files'].get('train'),
            context['files'].get('test')
        )
    
    context['summary'] = handler._generate_summary(context)
    
    return context


def _parse_file_sync(filepath: str) -> Dict:
    """Parse single file synchronously"""
    
    path = Path(filepath)
    
    info = {
        'filename': path.name,
        'size_bytes': path.stat().st_size,
        'size_mb': round(path.stat().st_size / (1024 * 1024), 2),
        'path': str(path),
        'extension': path.suffix
    }
    
    # Parse CSV
    if path.suffix == '.csv':
        df = pd.read_csv(filepath)
        info.update({
            'type': 'csv',
            'rows': len(df),
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample': df.head(3).to_dict(),
            'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        })
    
    # Parse JSON
    elif path.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        info.update({
            'type': 'json',
            'keys': list(data.keys()) if isinstance(data, dict) else None
        })
    
    return info
