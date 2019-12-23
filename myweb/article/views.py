"""
每建一个视图，都要配置URLConfs, 将用户请求的URL链接关联起来(将URL映射到视图上)

views.py 相当于逻辑业务控制层
"""

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render
from .models import ArticlePost


# request里面包含get或post的内容、用户浏览器、系统等信息
def article_list(request):
    articles = ArticlePost.objects.all() # 取出sqlite中所有的这个对象
    context = {'articles': articles} # 包装在这个上下文中,其实就是一个类似json字符串的东西
    # 但是list.html我们并没有，因此创建templates/article/list.html对象
    return render(request, 'article/list.html', context) # render的作用: 结合模板和上下文,并返回渲染后的HttpResponse对象


# 文章详情
def article_detail(request, id):
    article = ArticlePost.objects.get(id=id) # id是Django自动生成的用于索引的主键(Primary Key)
    context = {'article': article}
    return render(request, 'article/detail.html', context)
