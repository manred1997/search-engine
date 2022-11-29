from django.urls import re_path
from . import views


urlpatterns = [
    re_path(r'^$', views.index, name='index'),
    re_path(r'^auto-complete$', views.auto_complete, name='auto_complete'),
    re_path(r'^instant-search$', views.instant_search, name='instant_search')
]

app_name = 'notebook'