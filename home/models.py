from django.contrib.auth.models import AbstractUser
from django.db import models

# Create your models here.
class User(AbstractUser):
    avatar = models.ImageField(upload_to='uploads/%y/%m')

class Category(models.Model):
    name = models.CharField(max_length=100, null=False, unique=True)

class FrameGlasses(models.Model):
    frame = models.CharField(max_length=50,null=False)

class Product(models.Model):
    product = models.CharField(max_length=100, null=False)
    description = models.TextField(null=True, blank=True)
    price = models.DecimalField(null=False, max_digits=5, decimal_places=2, default='0')
    discount = models.IntegerField(null=True, default='0')
    image = models.CharField(max_length=150, null=False, default="null")
    view = models.IntegerField(null=False, default='0')
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    active = models.BooleanField(default=True)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True)
    type = models.ForeignKey(FrameGlasses, on_delete=models.SET_NULL,null=True)
class Orders(models.Model):
    name_customer = models.CharField(max_length=100, null=False)
    phone_customer= models.CharField(max_length=10, null=False)
    add_customer  = models.CharField(max_length=200,null=False)
    addDetail_customer = models.CharField(max_length=200,null=False)
    total_order = models.IntegerField(null=False)
    message = models.CharField(max_length=300, null=True)
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    payment = models.CharField(max_length=50, null=False)

class OrderDetails(models.Model):
    id_order = models.ForeignKey(Orders,on_delete=models.SET_NULL,null=True)
    id_product = models.ForeignKey(Product,on_delete=models.SET_NULL,null=True)
    num_product = models.IntegerField(null=False)
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)

class ShapeFace(models.Model):
    shape = models.CharField(max_length=50, null=False)

class Recommendation(models.Model):
    id_shape = models.ForeignKey(ShapeFace,on_delete=models.SET_NULL,null=True)
    id_frame = models.ForeignKey(FrameGlasses,on_delete=models.SET_NULL,null=True)

class News(models.Model):
    name = models.CharField(max_length=300, null=False, default="")
    image = models.CharField(max_length=300, null=True)
    content = models.TextField()
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)



