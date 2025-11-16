from django.urls import path
from . import views

urlpatterns = [
    # This URL pattern will point to your 'run_experiment' view.
    # The name 'run_experiment' lets us refer to it in templates.
    path('', views.run_experiment, name='run_experiment'),
]
