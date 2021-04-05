from django.conf.urls import url
from apps.tumovies import views

urlpatterns = [
	url(r'^$', view=views.tumovies_index_view, name='bookadmin_index_view'),
    url(r'^query', views.query, name='query'),
    url(r'^search_query', views.search_query, name='search_query'),
    url(r'^search', views.search, name='search'),
    url(r'^movie', views.movie, name='movie'),
	url(r'^trailer', views.trailer, name='trailer'),
]
