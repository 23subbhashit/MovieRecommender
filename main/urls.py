from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('CF',views.main,name="main"),
    path('',views.home,name="home"),
    path('SVDk',views.SvdK,name="SvdK"),
]