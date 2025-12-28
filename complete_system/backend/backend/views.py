"""
AIDAS Views
===========

API views for the AIDAS (Autonomous Intelligence for Data Analysis & Science) system.

Author: MiniMax Agent
"""

import logging
import os
import uuid
from django.conf import settings
from django.http import JsonResponse, FileResponse
from rest_framework import status, viewsets, views
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from api.models import Dataset, AnalysisJob, AnalysisTemplate, SavedInsight
from api.serializers import (
    DatasetSerializer,
    AnalysisJobSerializer,
    AnalysisJobCreateSerializer,
    AnalysisJobResultSerializer,
    AnalysisTemplateSerializer,
    SavedInsightSerializer,
)


logger = logging.getLogger(__name__)


class FileUploadView(views.APIView):
    """View for uploading datasets"""
    parser_classes = [MultiPartParser, FormParser]
    
    @extend_schema(
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'file': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'The file to upload (CSV, JSON, XLSX, or Parquet)'
                    }
                },
                'required': ['file']
            }
        },
        responses={201: DatasetSerializer}
    )
    def post(self, request):
        """Upload a new dataset for analysis"""
        uploaded_file = request.FILES.get('file')
        
        if not uploaded_file:
            return Response(
                {'error': 'No file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate file type
        valid_extensions = ['.csv', '.json', '.xlsx', '.parquet']
        file_name = uploaded_file.name.lower()
        file_ext = os.path.splitext(file_name)[1]
        
        if file_ext not in valid_extensions:
            return Response(
                {'error': f'Invalid file type. Allowed types: {", ".join(valid_extensions)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Generate unique filename
        unique_name = f"{uuid.uuid4()}_{uploaded_file.name}"
        file_path = os.path.join(settings.MEDIA_ROOT, unique_name)
        
        # Save file
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Create dataset record
        dataset = Dataset.objects.create(
            name=uploaded_file.name,
            file_path=file_path,
            file_size=uploaded_file.size,
            file_type=file_ext[1:].lower(),
        )
        
        serializer = DatasetSerializer(dataset)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class AnalysisViewSet(viewsets.ViewSet):
    """ViewSet for managing analysis jobs"""
    
    def list(self, request):
        """List all analysis jobs"""
        queryset = AnalysisJob.objects.all()
        status_filter = request.query_params.get('status')
        
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        serializer = AnalysisJobSerializer(queryset, many=True)
        return Response(serializer.data)
    
    @extend_schema(
        request=AnalysisJobCreateSerializer,
        responses={201: AnalysisJobSerializer}
    )
    def create(self, request):
        """Start a new analysis job"""
        serializer = AnalysisJobCreateSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        dataset_id = serializer.validated_data['dataset_id']
        config = serializer.validated_data.get('config', {})
        
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
            status='pending',
            output_dir=str(settings.ANALYSIS_OUTPUT_DIR / str(uuid.uuid4())),
        )
        
        # TODO: Start async task (Celery)
        # For now, we'll mark it as running immediately
        job.status = 'running'
        job.save()
        
        # TODO: Integrate with AIDAS system
        # from aidas_system import AutonomousIntelligence
        # aidas = AutonomousIntelligence()
        # results = aidas.run_analysis(dataset.file_path)
        
        serializer = AnalysisJobSerializer(job)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    def retrieve(self, request, pk=None):
        """Get a specific analysis job"""
        try:
            job = AnalysisJob.objects.get(id=pk)
        except AnalysisJob.DoesNotExist:
            return Response(
                {'error': 'Analysis job not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer = AnalysisJobSerializer(job)
        return Response(serializer.data)
    
    @extend_schema(
        responses={200: AnalysisJobResultSerializer}
    )
    @action(detail=True, methods=['get'])
    def results(self, request, pk=None):
        """Get results for a specific analysis job"""
        try:
            job = AnalysisJob.objects.get(id=pk)
        except AnalysisJob.DoesNotExist:
            return Response(
                {'error': 'Analysis job not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        if job.status != 'completed':
            return Response(
                {'error': 'Analysis not yet completed', 'status': job.status},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        serializer = AnalysisJobResultSerializer(job)
        return Response(serializer.data)
    
    @extend_schema(
        responses={200: {'description': 'Analysis cancelled'}}
    )
    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel a running analysis job"""
        try:
            job = AnalysisJob.objects.get(id=pk)
        except AnalysisJob.DoesNotExist:
            return Response(
                {'error': 'Analysis job not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        if job.status not in ['pending', 'running']:
            return Response(
                {'error': f'Cannot cancel job in {job.status} status'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        job.status = 'cancelled'
        job.save()
        
        return Response({'message': 'Analysis job cancelled'})


class HealthCheckView(views.APIView):
    """View for system health checks"""
    
    def get(self, request):
        """Perform health check"""
        health_status = {
            'status': 'healthy',
            'version': '1.0.0',
            'services': {}
        }
        
        # Check database connection
        try:
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute('SELECT 1')
            health_status['services']['database'] = 'healthy'
        except Exception as e:
            health_status['services']['database'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check Redis connection
        try:
            import redis
            r = redis.Redis(
                host=settings.CACHES['default']['LOCATION'].split('://')[1].split(':')[0],
                port=int(settings.CACHES['default']['LOCATION'].split(':')[1].split('/')[0]),
                socket_timeout=5
            )
            r.ping()
            health_status['services']['redis'] = 'healthy'
        except Exception as e:
            health_status['services']['redis'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        return Response(health_status)


class SchemaView(views.APIView):
    """View for serving OpenAPI schema"""
    
    def get(self, request):
        """Return OpenAPI schema"""
        return Response({
            'info': {
                'title': 'AIDAS API',
                'version': '1.0.0',
                'description': 'Autonomous Intelligence for Data Analysis & Science - API Documentation'
            },
            'servers': [
                {'url': 'http://localhost:8000', 'description': 'Development server'}
            ]
        })
