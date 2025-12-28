"""
AIDAS API Views
===============

API endpoints for the AIDAS (Autonomous Intelligence for Data Analysis & Science) system.

Features:
- File upload and management
- Analysis job submission and monitoring
- Results retrieval
- Visualization data

Author: MiniMax Agent
"""

import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated, AllowAny
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from .models import AnalysisJob, Dataset, User
from .serializers import (
    AnalysisJobSerializer,
    DatasetSerializer,
    AnalysisResultSerializer,
    UploadSerializer
)

logger = logging.getLogger(__name__)


class FileUploadView(APIView):
    """Handle file uploads for analysis"""
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [AllowAny]  # Change to IsAuthenticated in production
    
    @extend_schema(
        request=UploadSerializer,
        responses={201: DatasetSerializer},
        description="Upload a dataset file for analysis"
    )
    def post(self, request):
        """Upload a dataset file"""
        file_serializer = UploadSerializer(data=request.data)
        
        if not file_serializer.is_valid():
            return Response(
                {'error': 'Invalid file', 'details': file_serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response(
                {'error': 'No file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate file type
        valid_extensions = ['.csv', '.json', '.xlsx', '.parquet']
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        if file_ext not in valid_extensions:
            return Response(
                {'error': f'Invalid file type. Supported types: {", ".join(valid_extensions)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Save file
        upload_dir = Path(settings.MEDIA_UPLOAD_DIR) / 'datasets'
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{uploaded_file.name}"
        file_path = upload_dir / safe_filename
        
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Create dataset record
        dataset = Dataset.objects.create(
            name=uploaded_file.name,
            file_path=str(file_path),
            file_size=uploaded_file.size,
            file_type=file_ext.replace('.', ''),
            uploaded_by=request.user if request.user.is_authenticated else None
        )
        
        logger.info(f"File uploaded: {file_path}")
        
        return Response(
            DatasetSerializer(dataset).data,
            status=status.HTTP_201_CREATED
        )


class AnalysisViewSet(viewsets.ViewSet):
    """ViewSet for analysis operations"""
    permission_classes = [AllowAny]  # Change to IsAuthenticated in production
    authentication_classes = [JWTAuthentication]
    
    @extend_schema(
        request={
            'type': 'object',
            'properties': {
                'dataset_id': {'type': 'integer'},
                'config': {
                    'type': 'object',
                    'properties': {
                        'max_iterations': {'type': 'integer'},
                        'confidence_threshold': {'type': 'number'},
                        'sample_size': {'type': 'integer'}
                    }
                }
            }
        },
        responses={202: AnalysisJobSerializer},
        description="Start a new analysis job"
    )
    @action(detail=False, methods=['post'])
    def start(self, request):
        """Start a new analysis job"""
        dataset_id = request.data.get('dataset_id')
        config = request.data.get('config', {})
        
        if not dataset_id:
            return Response(
                {'error': 'dataset_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            dataset = Dataset.objects.get(id=dataset_id)
        except Dataset.DoesNotExist:
            return Response(
                {'error': 'Dataset not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Create analysis job
        job = AnalysisJob.objects.create(
            dataset=dataset,
            config=config,
            status='pending'
        )
        
        # Start analysis (in production, use Celery)
        self._run_analysis(job)
        
        return Response(
            AnalysisJobSerializer(job).data,
            status=status.HTTP_202_ACCEPTED
        )
    
    def _run_analysis(self, job):
        """Run the analysis (synchronous for demo, use Celery in production)"""
        try:
            job.status = 'running'
            job.save()
            
            # Import and run AIDAS
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            
            from aidas_system import AutonomousIntelligence, AnalysisConfig
            
            # Create config
            aidas_config = AnalysisConfig(
                max_iterations=job.config.get('max_iterations', 3),
                confidence_threshold=job.config.get('confidence_threshold', 0.95),
                sample_size=job.config.get('sample_size', 10000)
            )
            
            # Run analysis
            aidas = AutonomousIntelligence(config=aidas_config)
            results = aidas.analyze(job.dataset.file_path, str(job.output_dir))
            
            # Save results
            job.results = results
            job.status = 'completed' if results.get('status') == 'success' else 'failed'
            job.completed_at = datetime.now()
            job.save()
            
            logger.info(f"Analysis completed: {job.id}")
            
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.save()
            logger.error(f"Analysis failed: {job.id} - {e}")
    
    @extend_schema(
        responses={200: AnalysisJobSerializer},
        description="Get all analysis jobs"
    )
    @action(detail=False, methods=['get'])
    def list(self, request):
        """List all analysis jobs"""
        jobs = AnalysisJob.objects.all().order_by('-created_at')
        serializer = AnalysisJobSerializer(jobs, many=True)
        return Response(serializer.data)
    
    @extend_schema(
        responses={200: AnalysisJobSerializer},
        description="Get a specific analysis job"
    )
    def retrieve(self, request, pk=None):
        """Get a specific analysis job"""
        try:
            job = AnalysisJob.objects.get(id=pk)
            return Response(AnalysisJobSerializer(job).data)
        except AnalysisJob.DoesNotExist:
            return Response(
                {'error': 'Job not found'},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @extend_schema(
        responses={200: AnalysisResultSerializer},
        description="Get analysis results"
    )
    @action(detail=True, methods=['get'])
    def results(self, request, pk=None):
        """Get analysis results"""
        try:
            job = AnalysisJob.objects.get(id=pk)
            
            if job.status != 'completed':
                return Response(
                    {'error': 'Analysis not completed', 'status': job.status},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            return Response(job.results)
        except AnalysisJob.DoesNotExist:
            return Response(
                {'error': 'Job not found'},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @extend_schema(
        responses={200: dict},
        description="Cancel a running analysis job"
    )
    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel a running analysis job"""
        try:
            job = AnalysisJob.objects.get(id=pk)
            
            if job.status not in ['pending', 'running']:
                return Response(
                    {'error': 'Job cannot be cancelled', 'status': job.status},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            job.status = 'cancelled'
            job.save()
            
            return Response({'status': 'cancelled', 'job_id': job.id})
        except AnalysisJob.DoesNotExist:
            return Response(
                {'error': 'Job not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class HealthCheckView(APIView):
    """Health check endpoint"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Return health status"""
        return Response({
            'status': 'healthy',
            'service': 'AIDAS Backend',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })


class SchemaView(APIView):
    """Return OpenAPI schema"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Return OpenAPI schema as JSON"""
        from drf_spectacular.generators import SchemaGenerator
        from drf_spectacular.renderers import JSONRenderer
        
        generator = SchemaGenerator()
        schema = generator.get_schema(request=request)
        renderer = JSONRenderer()
        
        return Response(
            renderer.render(schema),
            content_type='application/json'
        )
