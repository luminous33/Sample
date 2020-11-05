"""Investar_final URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.urls import path
from Stock.views import homepage, introduce, introbol, introdeep, introdual, introtriple, bol2, triple, deep, search

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', homepage),
    path('homepage/', homepage),
    path('introduce', introduce),
    path('introbol', introbol),
    path('introtriple', introtriple),
    path('introdual', introdual),
    path('introdeep', introdeep),
    path('bol2', bol2),
    path('triple', triple),
    path('deep', deep),
    path('search', search),
]
