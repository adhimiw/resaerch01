"""
AIDAS URL Configuration
========================

URL routing for the AIDAS backend API.

Author: MiniMax Agent
"""

from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from .views import (
    FileUploadView,
    AnalysisViewSet,
    HealthCheckView,
    SchemaView
)

urlpatterns = [
    # Django Admin
    path('admin/', admin.site.urls),
    
    # API Documentation
    path('api/schema/', SchemaView.as_view(), name='api-schema'),
    path('api/docs/', SpectacularAPIView.as_view(), name='api-schema-yaml'),
    path('api/swagger/', SpectacularSwaggerView.as_view(url_name='api-schema'), name='api-docs'),
    
    # JWT Authentication
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    
    # File Upload
    path('api/upload/', FileUploadView.as_view(), name='file-upload'),
    
    # Analysis endpoints
    path('api/analysis/', AnalysisViewSet.as_view({
        'get': 'list',
        'post': 'create'
    }), name='analysis-list'),
    
    path('api/analysis/<int:pk>/', AnalysisViewSet.as_view({
        'get': 'retrieve'
    }), name='analysis-detail'),
    
    path('api/analysis/<int:pk>/results/', AnalysisViewSet.as_view({
        'get': 'results'
    }), name='analysis-results'),
    
    path('api/analysis/<int:pk>/cancel/', AnalysisViewSet.as_view({
        'post': 'cancel'
    }), name='analysis-cancel'),
    
    # Health Check
    path('api/health/', HealthCheckView.as_view(), name='health-check'),
]

# Media files in development
from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
