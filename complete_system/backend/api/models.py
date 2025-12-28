"""
AIDAS Models
============

Database models for the AIDAS (Autonomous Intelligence for Data Analysis & Science) system.

Author: MiniMax Agent
"""

import uuid
from django.db import models
from django.conf import settings
from django.contrib.auth.models import User


class Dataset(models.Model):
    """Model to store uploaded datasets"""
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    file_size = models.BigIntegerField()
    file_type = models.CharField(max_length=50)  # csv, json, xlsx, parquet
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.file_type})"
    
    @property
    def size_mb(self):
        return f"{self.file_size / (1024 * 1024):.2f} MB"


class AnalysisJob(models.Model):
    """Model to track analysis jobs"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    id = models.AutoField(primary_key=True)
    session_id = models.CharField(max_length=100, unique=True, default=uuid.uuid4)
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='analysis_jobs'
    )
    config = models.JSONField(default=dict)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    results = models.JSONField(null=True, blank=True)
    output_dir = models.CharField(max_length=500)
    error_message = models.TextField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Analysis {self.session_id} - {self.status}"
    
    def save(self, *args, **kwargs):
        if not self.output_dir:
            self.output_dir = str(settings.ANALYSIS_OUTPUT_DIR / self.session_id)
        super().save(*args, **kwargs)
    
    @property
    def duration_seconds(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class AnalysisTemplate(models.Model):
    """Model to store reusable analysis templates"""
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    description = models.TextField()
    config = models.JSONField()
    is_public = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.name


class SavedInsight(models.Model):
    """Model to store saved insights"""
    id = models.AutoField(primary_key=True)
    job = models.ForeignKey(
        AnalysisJob,
        on_delete=models.CASCADE,
        related_name='saved_insights'
    )
    title = models.CharField(max_length=255)
    description = models.TextField()
    category = models.CharField(max_length=100)
    impact = models.CharField(max_length=50)
    recommendation = models.TextField()
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title
