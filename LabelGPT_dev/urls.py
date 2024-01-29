"""
URL configuration for LabelGPT_dev project.
"""
from django.contrib import admin
from django.urls import path
from qa import views as qa

urlpatterns = [
    # path('admin/', admin.site.urls), # No User control in current version
    path('qa/',qa.home),
    path('',qa.home), # Automatic redirect
    path('response_demo/', qa.response)
]
