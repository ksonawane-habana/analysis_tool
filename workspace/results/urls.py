from django.urls import path
from . import views

app_name = 'results'
urlpatterns = [
    path('', views.home, name='home'),
    path('models/<str:category>/', views.model_list, name='model_list'),
    path('model/<str:model_name>/', views.model_detail, name='model_detail'),
    path('subgraph/<str:model_name>/<str:subgraph_name>/', views.subgraph_index, name='subgraph_index'),
    path('mlir/<str:model_name>/<str:fused_kernel_name>/<str:source_type>/', views.mlir_content, name='mlir_content'),
    path('api/category/<str:category>/', views.get_category_data, name='category_data'),
]
