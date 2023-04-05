from django.contrib import admin

# Register your models here.
from .models import *

class New(admin.ModelAdmin):
    list_display = ('name','created_date')

class Frame(admin.ModelAdmin):
    list_display = ('frame','id')

class Products(admin.ModelAdmin):
    list_display = ('product','price','active','view','created_date')
    ordering = ['view','price','created_date']
    search_fields = ['product','price']

class Order(admin.ModelAdmin):
    list_display = ('name_customer','total_order')

admin.site.register(Orders, Order)
admin.site.register(FrameGlasses, Frame)
admin.site.register(Product,Products)
admin.site.register(News, New)
