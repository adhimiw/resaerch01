# 📓 Jupyter MCP Server Integration Guide

## Overview

Jupyter MCP enables your agent to execute Python code in real Jupyter notebooks with **persistent state** - variables stay in memory across agent calls, just like a human data scientist working interactively.

## Why Jupyter MCP?

### Traditional Approach (Limited)
```python
# Agent runs code in isolated process
result = exec("df = pd.read_csv('data.csv'); df.head()")
# Next call: df is GONE - no persistence!
result2 = exec("df.describe()")  # ERROR: df not defined
```

### Jupyter MCP Approach (Persistent)
```python
# Agent executes in Jupyter kernel
jupyter.execute("df = pd.read_csv('data.csv'); df.head()")
# Kernel remembers df!
jupyter.execute("df.describe()")  # WORKS - df still exists
jupyter.execute("df['new_col'] = df['old_col'] * 2")  # Build on previous work
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Your Agent (DSPy Universal System)                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Jupyter MCP Server (Python)                            │
│  ┌────────────────────────────────────────────────┐     │
│  │  Operations:                                   │     │
│  │  • create_notebook: New .ipynb file            │     │
│  │  • add_cell: Insert code/markdown cell         │     │
│  │  • execute_cell: Run cell, get output          │     │
│  │  • get_result: Retrieve execution results      │     │
│  │  • list_notebooks: Show all notebooks          │     │
│  └────────────────────────────────────────────────┘     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Jupyter Kernel (IPython)                               │
│  ┌────────────────────────────────────────────────┐     │
│  │  Persistent State:                             │     │
│  │  • Variables (df, model, results)              │     │
│  │  • Imports (pandas, sklearn, etc.)             │     │
│  │  • Functions & classes                         │     │
│  │  • Plot outputs (matplotlib figures)           │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Installation

### From GitHub (Recommended)
```bash
# Clone reference
git clone https://github.com/datalayer/jupyter-mcp-server.git reference/jupyter-mcp-server

# Install
cd reference/jupyter-mcp-server
pip install -e .
```

### Via PyPI
```bash
pip install jupyter-mcp-server
```

## Usage

### Starting Jupyter MCP Server

```python
from jupyter_mcp_server import JupyterMCPServer

# Start server
server = JupyterMCPServer(port=8888)
server.start()
```

### Basic Operations

```python
import requests

class JupyterMCPClient:
    """Client for Jupyter MCP Server"""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url
    
    def create_notebook(self, name: str) -> str:
        """Create new notebook, returns notebook_id"""
        response = requests.post(
            f"{self.base_url}/notebooks",
            json={"name": name}
        )
        return response.json()["notebook_id"]
    
    def add_code_cell(self, notebook_id: str, code: str) -> str:
        """Add code cell, returns cell_id"""
        response = requests.post(
            f"{self.base_url}/notebooks/{notebook_id}/cells",
            json={"code": code, "cell_type": "code"}
        )
        return response.json()["cell_id"]
    
    def execute_cell(self, notebook_id: str, cell_id: str) -> dict:
        """Execute cell and get output"""
        response = requests.post(
            f"{self.base_url}/notebooks/{notebook_id}/cells/{cell_id}/execute"
        )
        return response.json()
    
    def get_output(self, notebook_id: str, cell_id: str) -> dict:
        """Get cell execution output"""
        response = requests.get(
            f"{self.base_url}/notebooks/{notebook_id}/cells/{cell_id}/output"
        )
        return response.json()
```

## Integration with DSPy Agent

### Enhanced Agent with Jupyter Support

```python
class UniversalAgenticDataScience:
    """Agent with Jupyter MCP for iterative analysis"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jupyter_client = JupyterMCPClient()
        self.current_notebook = None
    
    def _execute_analysis_with_jupyter(self, dataset_path: str, plan):
        """Execute analysis using Jupyter for persistence"""
        
        # Create notebook for this analysis
        notebook_id = self.jupyter_client.create_notebook(
            name=f"analysis_{os.path.basename(dataset_path)}"
        )
        self.current_notebook = notebook_id
        
        # Cell 1: Load data
        cell1 = self.jupyter_client.add_code_cell(
            notebook_id,
            f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('{dataset_path}')
print(f"Loaded {{len(df)}} rows, {{len(df.columns)}} columns")
df.head()
"""
        )
        result1 = self.jupyter_client.execute_cell(notebook_id, cell1)
        
        # Cell 2: Exploratory analysis (builds on Cell 1!)
        cell2 = self.jupyter_client.add_code_cell(
            notebook_id,
            """
# Data is still in memory from Cell 1!
print("Dataset Info:")
print(df.info())

print("\\nMissing Values:")
print(df.isnull().sum())

print("\\nStatistics:")
df.describe()
"""
        )
        result2 = self.jupyter_client.execute_cell(notebook_id, cell2)
        
        # Cell 3: Correlations
        cell3 = self.jupyter_client.add_code_cell(
            notebook_id,
            """
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("Correlation matrix saved")
"""
        )
        result3 = self.jupyter_client.execute_cell(notebook_id, cell3)
        
        # Agent can add more cells dynamically based on plan!
        
        return {
            'notebook_id': notebook_id,
            'cells_executed': 3,
            'outputs': [result1, result2, result3]
        }
```

## Real Example: Iterative Model Development

```python
def train_ml_model_iteratively(self, df_path: str, target: str):
    """Shows the power of persistent state"""
    
    nb_id = self.jupyter_client.create_notebook("model_training")
    
    # Step 1: Data prep
    self.jupyter_client.add_code_cell(nb_id, f"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('{df_path}')
X = df.drop('{target}', axis=1)
y = df['{target}']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {{len(X_train)}}, Test: {{len(X_test)}}")
""")
    
    # Step 2: Try model 1
    self.jupyter_client.add_code_cell(nb_id, """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model1 = RandomForestClassifier(n_estimators=100)
model1.fit(X_train_scaled, y_train)
pred1 = model1.predict(X_test_scaled)
acc1 = accuracy_score(y_test, pred1)
print(f"RandomForest Accuracy: {acc1:.3f}")
""")
    
    # Step 3: Try model 2 (data still in memory!)
    self.jupyter_client.add_code_cell(nb_id, """
from sklearn.ensemble import GradientBoostingClassifier

model2 = GradientBoostingClassifier(n_estimators=100)
model2.fit(X_train_scaled, y_train)
pred2 = model2.predict(X_test_scaled)
acc2 = accuracy_score(y_test, pred2)
print(f"GradientBoosting Accuracy: {acc2:.3f}")

# Compare
best = 'RandomForest' if acc1 > acc2 else 'GradientBoosting'
print(f"\\nBest Model: {best}")
""")
    
    # Execute all cells
    for cell_id in self.jupyter_client.list_cells(nb_id):
        self.jupyter_client.execute_cell(nb_id, cell_id)
```

## Key Benefits

### 1. **No Code Duplication**
- Load data once, use in multiple cells
- Train model once, evaluate in different ways
- No need to re-run expensive operations

### 2. **Visual Outputs**
- Matplotlib plots saved automatically
- Tables rendered in notebook
- Agent can "see" visualizations

### 3. **Debugging**
- Agent can inspect variables at any point
- Add print statements between steps
- Notebook shows full execution history

### 4. **Reproducibility**
- Generated .ipynb file is shareable
- Others can re-run exact analysis
- Clear documentation of what agent did

## Configuration

### MCP Config Entry
```json
{
  "mcpServers": {
    "jupyter": {
      "command": "python",
      "args": ["-m", "jupyter_mcp_server"],
      "env": {
        "JUPYTER_PORT": "8888",
        "JUPYTER_TOKEN": "your-token-here"
      }
    }
  }
}
```

## Advanced: Multi-Notebook Workflows

```python
def comparative_analysis(self, datasets: List[str]):
    """Run analysis on multiple datasets with separate notebooks"""
    
    results = {}
    
    for dataset in datasets:
        # Each dataset gets its own notebook & kernel
        nb_id = self.jupyter_client.create_notebook(
            f"analysis_{Path(dataset).stem}"
        )
        
        # Run analysis
        result = self.analyze_in_notebook(nb_id, dataset)
        results[dataset] = result
    
    # Create summary notebook comparing all results
    summary_nb = self.jupyter_client.create_notebook("summary")
    self.jupyter_client.add_code_cell(summary_nb, f"""
import pandas as pd

results_data = {results}
summary_df = pd.DataFrame(results_data).T
summary_df.plot(kind='bar', figsize=(12, 6))
plt.title('Cross-Dataset Comparison')
plt.savefig('comparison.png')
""")
    
    return results
```

## Integration Checklist

- [ ] Install jupyter-mcp-server
- [ ] Start Jupyter kernel
- [ ] Add to MCP config
- [ ] Update agent to use Jupyter client
- [ ] Test with sample dataset
- [ ] Verify state persistence
- [ ] Check visual outputs
- [ ] Document in paper

## Resources

- GitHub: https://github.com/datalayer/jupyter-mcp-server
- Jupyter Protocol: https://jupyter-client.readthedocs.io/
- MCP Spec: https://modelcontextprotocol.io/
