"""
AIDAS Serializers
=================

DRF serializers for the AIDAS API.

Author: MiniMax Agent
"""

from rest_framework import serializers
from django.contrib.auth.models import User

from .models import Dataset, AnalysisJob, AnalysisTemplate, SavedInsight


class UserSerializer(serializers.ModelSerializer):
    """User serializer"""
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id']


class DatasetSerializer(serializers.ModelSerializer):
    """Dataset serializer"""
    size_mb = serializers.ReadOnlyField()
    
    class Meta:
        model = Dataset
        fields = [
            'id', 'name', 'file_path', 'file_size', 'size_mb',
            'file_type', 'uploaded_by', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'file_path', 'file_size', 'file_type', 'created_at']


class UploadSerializer(serializers.Serializer):
    """File upload serializer"""
    file = serializers.FileField(required=True)
    
    def validate_file(self, value):
        """Validate uploaded file"""
        valid_extensions = ['.csv', '.json', '.xlsx', '.parquet']
        file_ext = value.name.split('.')[-1].lower()
        
        if f'.{file_ext}' not in valid_extensions:
            raise serializers.ValidationError(
                f'Invalid file type. Supported types: {", ".join(valid_extensions)}'
            )
        
        # Max file size: 100MB
        max_size = 100 * 1024 * 1024
        if value.size > max_size:
            raise serializers.ValidationError(
                f'File too large. Maximum size: 100MB'
            )
        
        return value


class AnalysisJobSerializer(serializers.ModelSerializer):
    """Analysis job serializer"""
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    duration_seconds = serializers.ReadOnlyField()
    
    class Meta:
        model = AnalysisJob
        fields = [
            'id', 'session_id', 'dataset', 'dataset_name', 'config',
            'status', 'results', 'output_dir', 'error_message',
            'started_at', 'completed_at', 'duration_seconds',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'session_id', 'status', 'results', 'output_dir',
            'error_message', 'started_at', 'completed_at',
            'created_at', 'updated_at'
        ]


class AnalysisJobCreateSerializer(serializers.Serializer):
    """Serializer for creating analysis jobs"""
    dataset_id = serializers.IntegerField(required=True)
    config = serializers.JSONField(required=False, default=dict)
    
    def validate_dataset_id(self, value):
        """Validate dataset exists"""
        try:
            Dataset.objects.get(id=value)
        except Dataset.DoesNotExist:
            raise serializers.ValidationError('Dataset not found')
        return value


class AnalysisJobResultSerializer(serializers.ModelSerializer):
    """Serializer for analysis job results"""
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    duration_seconds = serializers.ReadOnlyField()
    
    class Meta:
        model = AnalysisJob
        fields = [
            'id', 'session_id', 'dataset', 'dataset_name',
            'status', 'results', 'error_message',
            'started_at', 'completed_at', 'duration_seconds',
            'created_at'
        ]
        read_only_fields = fields


class AnalysisResultSerializer(serializers.Serializer):
    """Analysis result serializer"""
    status = serializers.CharField()
    session_id = serializers.CharField()
    total_duration = serializers.FloatField()
    data_summary = serializers.DictField()
    quality_report = serializers.DictField()
    hypotheses = serializers.DictField()
    models = serializers.DictField()
    insights = serializers.ListField()
    explanation = serializers.CharField()


class AnalysisTemplateSerializer(serializers.ModelSerializer):
    """Analysis template serializer"""
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = AnalysisTemplate
        fields = [
            'id', 'name', 'description', 'config', 'is_public',
            'created_by', 'created_by_username', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_by', 'created_at', 'updated_at']


class SavedInsightSerializer(serializers.ModelSerializer):
    """Saved insight serializer"""
    class Meta:
        model = SavedInsight
        fields = [
            'id', 'job', 'title', 'description', 'category',
            'impact', 'recommendation', 'metadata', 'created_at'
        ]
        read_only_fields = ['id', 'job', 'created_at']
