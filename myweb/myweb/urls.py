"""myweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url

urlpatterns = [

    # # namespace可以保证反查到唯一的url，即使不同的app使用了相同的url（后面会用到）
    # path('article/', include('article.urls', namespace='article')), # 配置article这个app的urls
    #
    # path('summarizer/', include('summarizer.urls'), namespace='summarizer'),
    url(r'^admin/', admin.site.urls),
    url(r'', include('gentitle.urls')),
    # path(r'', include('gentitle.urls'), namespace='gentitle'),

]
