from django.contrib import admin
from django.urls import path, include  # Make sure 'include' is imported

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Add this line:
    # It tells Django to check 'predictor.urls' for any URL
    # that starts with 'experiment/'
    path('experiment/', include('predictor.urls')),
]
