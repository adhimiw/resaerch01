"""
Conversational RAG Interface with ChromaDB
Phase 1: Week 1-2 Implementation

Features:
- Vector database memory for past analyses
- Semantic search across historical insights
- Code snippet retrieval
- Similar dataset suggestions
- Interactive chat interface

Based on: INTEGRATION_PLAN.md - Phase 1 (ChromaDB RAG)
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib
from pathlib import Path


class ConversationalDataScienceAgent:
    """
    Interactive chatbot interface with RAG memory for data science
    
    Capabilities:
    1. Index past analyses for semantic search
    2. Chat interface with context from similar analyses
    3. Code snippet retrieval from successful past work
    4. Similar dataset suggestions
    5. Pattern recognition across domains
    """
    
    def __init__(self, dspy_agent=None, mcp_integration=None, persist_dir: str = "./chroma_db"):
        """
        Initialize conversational agent with ChromaDB backend
        
        Args:
            dspy_agent: Enhanced DSPy agent instance
            mcp_integration: MCP integration layer
            persist_dir: Directory for persisting ChromaDB data
        """
        print("🚀 Initializing Conversational Data Science Agent with ChromaDB...")
        
        # Create persist directory if not exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Use default embedding function (all-MiniLM-L6-v2)
        # This is a lightweight sentence transformer for semantic similarity
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create collections for different types of data
        self.analyses = self.chroma_client.get_or_create_collection(
            name="past_analyses",
            embedding_function=self.embedding_function,
            metadata={"description": "Historical data science analyses and results"}
        )
        
        self.insights = self.chroma_client.get_or_create_collection(
            name="insights",
            embedding_function=self.embedding_function,
            metadata={"description": "Key insights and discovered patterns"}
        )
        
        self.code_snippets = self.chroma_client.get_or_create_collection(
            name="code_snippets",
            embedding_function=self.embedding_function,
            metadata={"description": "Reusable code patterns and solutions"}
        )
        
        # Store references to other system components
        self.dspy_agent = dspy_agent
        self.mcp = mcp_integration
        
        print("✅ ChromaDB initialized with 3 collections:")
        print(f"   - past_analyses: {self.analyses.count()} documents")
        print(f"   - insights: {self.insights.count()} documents")
        print(f"   - code_snippets: {self.code_snippets.count()} documents")
    
    def index_analysis(self, dataset: str, analysis_result: Dict[str, Any]) -> str:
        """
        Store a completed analysis in ChromaDB for future retrieval
        
        Args:
            dataset: Name of the dataset analyzed
            analysis_result: Complete analysis output from DSPy agent
            
        Returns:
            Document ID of the stored analysis
        """
        print(f"📚 Indexing analysis for dataset: {dataset}")
        
        # Extract key information from analysis
        understanding = analysis_result.get('understanding', {})
        insights_data = analysis_result.get('insights', {})
        recommendations = analysis_result.get('recommendations', [])
        
        # Create comprehensive document text for embedding
        doc_text = f"""
Dataset: {dataset}
Domain: {understanding.get('domain', 'Unknown')}
ML Task: {understanding.get('ml_task', 'Unknown')}
Dataset Type: {understanding.get('dataset_type', 'Unknown')}

Key Insights:
{insights_data.get('key_insights', 'No insights available')}

Recommendations:
{chr(10).join(recommendations) if recommendations else 'No recommendations'}

Summary:
{insights_data.get('summary', 'No summary available')}
""".strip()
        
        # Generate unique ID
        doc_id = f"analysis_{dataset}_{hashlib.md5(doc_text.encode()).hexdigest()[:8]}"
        
        # Store in analyses collection
        self.analyses.add(
            documents=[doc_text],
            metadatas=[{
                "dataset": dataset,
                "domain": understanding.get('domain', 'Unknown'),
                "ml_task": understanding.get('ml_task', 'Unknown'),
                "dataset_type": understanding.get('dataset_type', 'Unknown'),
                "timestamp": datetime.now().isoformat(),
                "has_recommendations": len(recommendations) > 0
            }],
            ids=[doc_id]
        )
        
        # Also index individual insights
        if 'key_insights' in insights_data:
            insights_list = insights_data['key_insights'].split('\n')
            for i, insight in enumerate(insights_list):
                if insight.strip():
                    insight_id = f"{doc_id}_insight_{i}"
                    self.insights.add(
                        documents=[insight.strip()],
                        metadatas=[{
                            "source_dataset": dataset,
                            "analysis_id": doc_id,
                            "timestamp": datetime.now().isoformat()
                        }],
                        ids=[insight_id]
                    )
        
        print(f"✅ Analysis indexed with ID: {doc_id}")
        return doc_id
    
    def index_code_snippet(self, code: str, description: str, task: str, 
                          dataset: str = None, success: bool = True) -> str:
        """
        Store a reusable code snippet from a successful analysis
        
        Args:
            code: The actual code snippet
            description: What this code does
            task: Type of task (e.g., "feature_engineering", "outlier_detection")
            dataset: Optional dataset this was used on
            success: Whether this code successfully solved the problem
            
        Returns:
            Document ID of the stored snippet
        """
        doc_text = f"""
Task: {task}
Description: {description}

Code:
{code}
"""
        
        snippet_id = f"code_{task}_{hashlib.md5(code.encode()).hexdigest()[:8]}"
        
        self.code_snippets.add(
            documents=[doc_text],
            metadatas=[{
                "task": task,
                "description": description,
                "dataset": dataset or "general",
                "success": success,
                "timestamp": datetime.now().isoformat()
            }],
            ids=[snippet_id]
        )
        
        return snippet_id
    
    def chat(self, user_query: str, dataset_context: str = None, n_results: int = 5) -> Dict[str, Any]:
        """
        Conversational interface with RAG - retrieves relevant context and generates response
        
        Args:
            user_query: User's question or request
            dataset_context: Optional current dataset being worked on
            n_results: Number of similar analyses to retrieve
            
        Returns:
            Dictionary with response and sources
        """
        print(f"\n💬 User Query: {user_query}")
        
        # Retrieve relevant past analyses
        similar_analyses = self.analyses.query(
            query_texts=[user_query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Build context from retrieved analyses
        context_parts = []
        sources = []
        
        if similar_analyses['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                similar_analyses['documents'][0],
                similar_analyses['metadatas'][0],
                similar_analyses['distances'][0]
            )):
                context_parts.append(f"--- Past Analysis {i+1} (Similarity: {1-distance:.2f}) ---\n{doc}")
                sources.append({
                    "dataset": metadata.get('dataset', 'Unknown'),
                    "domain": metadata.get('domain', 'Unknown'),
                    "similarity": round(1 - distance, 3)
                })
        
        context = "\n\n".join(context_parts) if context_parts else "No similar past analyses found."
        
        # Generate response (if DSPy agent available)
        if self.dspy_agent:
            response = self._generate_response_with_context(user_query, context, dataset_context)
        else:
            response = f"Based on {len(sources)} similar past analyses:\n\n{context}"
        
        return {
            "response": response,
            "sources": sources,
            "context": context,
            "n_sources": len(sources)
        }
    
    def _generate_response_with_context(self, query: str, context: str, 
                                       dataset_context: str = None) -> str:
        """
        Generate response using DSPy agent with retrieved context
        
        This would integrate with the enhanced DSPy agent to generate
        contextually-aware responses based on past analyses
        """
        # TODO: Integrate with DSPy agent's chat signature
        # For now, return formatted context
        prompt = f"""
Based on past analyses, here's what I found relevant to your query:

{context}

Query: {query}
Current Dataset: {dataset_context or 'Not specified'}

Would you like me to analyze your current dataset using insights from these similar past analyses?
"""
        return prompt
    
    def suggest_similar_datasets(self, current_dataset: str, domain: str = None, 
                                n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar past analyses based on dataset characteristics
        
        Args:
            current_dataset: Description of current dataset
            domain: Optional domain filter (e.g., "finance", "healthcare")
            n_results: Number of suggestions to return
            
        Returns:
            List of similar dataset analyses
        """
        print(f"🔍 Finding datasets similar to: {current_dataset}")
        
        query_text = f"dataset similar to {current_dataset}"
        if domain:
            query_text += f" in {domain} domain"
        
        where_filter = {"domain": domain} if domain else None
        
        results = self.analyses.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        suggestions = []
        if results['documents'][0]:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                suggestions.append({
                    "dataset": metadata.get('dataset', 'Unknown'),
                    "domain": metadata.get('domain', 'Unknown'),
                    "ml_task": metadata.get('ml_task', 'Unknown'),
                    "similarity": round(1 - distance, 3),
                    "preview": doc[:200] + "..." if len(doc) > 200 else doc
                })
        
        return suggestions
    
    def get_code_suggestions(self, task_description: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant code snippets from past analyses
        
        Args:
            task_description: Description of the coding task
            n_results: Number of code snippets to return
            
        Returns:
            List of relevant code snippets with metadata
        """
        print(f"💻 Finding code snippets for: {task_description}")
        
        results = self.code_snippets.query(
            query_texts=[task_description],
            n_results=n_results,
            where={"success": True},  # Only return successful snippets
            include=['documents', 'metadatas', 'distances']
        )
        
        snippets = []
        if results['documents'][0]:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                snippets.append({
                    "code": doc,
                    "task": metadata.get('task', 'Unknown'),
                    "description": metadata.get('description', 'No description'),
                    "relevance": round(1 - distance, 3),
                    "dataset": metadata.get('dataset', 'general')
                })
        
        return snippets
    
    def get_insights_by_domain(self, domain: str, n_results: int = 10) -> List[str]:
        """
        Get all insights from a specific domain (e.g., "finance", "healthcare")
        
        Args:
            domain: Domain to filter by
            n_results: Maximum number of insights to return
            
        Returns:
            List of insight strings
        """
        # Search in analyses collection
        results = self.analyses.query(
            query_texts=[f"{domain} domain insights"],
            n_results=n_results,
            where={"domain": domain},
            include=['documents']
        )
        
        return results['documents'][0] if results['documents'] else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary with collection counts and metadata
        """
        # Get persist directory from client settings
        try:
            persist_dir = self.chroma_client._settings.persist_directory
        except AttributeError:
            # ChromaDB v1.3+ uses different internal structure
            persist_dir = getattr(self.chroma_client, '_settings', {}).get('persist_directory', 'Unknown')
            if persist_dir == 'Unknown':
                persist_dir = self.chroma_client._identifier if hasattr(self.chroma_client, '_identifier') else 'Unknown'
        
        return {
            "total_analyses": self.analyses.count(),
            "total_insights": self.insights.count(),
            "total_code_snippets": self.code_snippets.count(),
            "persist_directory": str(persist_dir)
        }
    
    def export_knowledge_base(self, output_file: str = "knowledge_base.json"):
        """
        Export entire knowledge base to JSON for backup/sharing
        
        Args:
            output_file: Path to save JSON file
        """
        print(f"📦 Exporting knowledge base to {output_file}...")
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "analyses": [],
            "insights": [],
            "code_snippets": []
        }
        
        # Export all analyses
        all_analyses = self.analyses.get(include=['documents', 'metadatas'])
        for doc, metadata in zip(all_analyses['documents'], all_analyses['metadatas']):
            export_data["analyses"].append({
                "document": doc,
                "metadata": metadata
            })
        
        # Export all insights
        all_insights = self.insights.get(include=['documents', 'metadatas'])
        for doc, metadata in zip(all_insights['documents'], all_insights['metadatas']):
            export_data["insights"].append({
                "document": doc,
                "metadata": metadata
            })
        
        # Export all code snippets
        all_snippets = self.code_snippets.get(include=['documents', 'metadatas'])
        for doc, metadata in zip(all_snippets['documents'], all_snippets['metadatas']):
            export_data["code_snippets"].append({
                "document": doc,
                "metadata": metadata
            })
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Knowledge base exported successfully")
        print(f"   Total size: {len(json.dumps(export_data))} bytes")


# Example usage demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("CONVERSATIONAL DATA SCIENCE AGENT - DEMO")
    print("=" * 60)
    
    # Initialize agent
    agent = ConversationalDataScienceAgent()
    
    # Example 1: Index a sample analysis
    print("\n1️⃣ Indexing sample Spotify analysis...")
    sample_analysis = {
        "understanding": {
            "domain": "music streaming",
            "ml_task": "time-series analysis",
            "dataset_type": "temporal"
        },
        "insights": {
            "key_insights": "COVID-19 caused 42.5% drop in streaming\nRecovery trend started in Q3 2020\nSeasonality detected with yearly pattern",
            "summary": "Music streaming dataset shows significant anomaly during pandemic"
        },
        "recommendations": [
            "Use Prophet for time-series forecasting",
            "Include COVID-19 as external regressor",
            "Consider segmenting by genre for deeper insights"
        ]
    }
    
    analysis_id = agent.index_analysis("spotify_monthly_streams.csv", sample_analysis)
    
    # Example 2: Index a code snippet
    print("\n2️⃣ Indexing reusable code snippet...")
    code = """
import pandas as pd
from fbprophet import Prophet

# Handle time-series with external regressors
df_prophet = df[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})
df_prophet['covid_period'] = (df_prophet['ds'] >= '2020-03-01') & (df_prophet['ds'] <= '2020-12-31')

model = Prophet()
model.add_regressor('covid_period')
model.fit(df_prophet)
"""
    
    agent.index_code_snippet(
        code=code,
        description="Time-series forecasting with external regressor for COVID impact",
        task="time_series_forecasting",
        dataset="spotify_monthly_streams.csv"
    )
    
    # Example 3: Chat with the agent
    print("\n3️⃣ Testing conversational interface...")
    response = agent.chat("How should I analyze a music streaming dataset?")
    print(f"\n🤖 Agent Response:\n{response['response']}")
    print(f"\n📚 Sources: {response['n_sources']} similar analyses found")
    
    # Example 4: Find similar datasets
    print("\n4️⃣ Finding similar datasets...")
    similar = agent.suggest_similar_datasets("podcast listening data", domain="music streaming")
    for i, suggestion in enumerate(similar, 1):
        print(f"   {i}. {suggestion['dataset']} (similarity: {suggestion['similarity']})")
    
    # Example 5: Get code suggestions
    print("\n5️⃣ Getting code suggestions...")
    snippets = agent.get_code_suggestions("detect anomalies in time-series data")
    for i, snippet in enumerate(snippets, 1):
        print(f"   {i}. {snippet['description']} (relevance: {snippet['relevance']})")
    
    # Example 6: Get statistics
    print("\n6️⃣ Knowledge base statistics:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE - ChromaDB RAG system operational!")
    print("=" * 60)
