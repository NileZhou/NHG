"""
每写好一个view函数，都要在这里配置一下路由地址
"""

from django.urls import path
from . import views

app_name = 'article'

urlpatterns = [
    path('article-list/', views.article_list, name='article_list'), # name用于反查url地址，相当于给url起名字
    # 用尖括号定义需要传递的参数
    path('article-detail/<int:id>/', views.article_detail, name='article_detail'),

]