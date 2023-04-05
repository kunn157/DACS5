import json
import urllib
import uuid
from itertools import product

# importing the libraries
import numpy as np  # for mathematical calculations
import cv2  # for face detection and other image operations
import dlib  # for detection of facial landmarks ex:nose,jawline,eyes
from django.core.files.storage import FileSystemStorage
from django.core.mail import send_mail, EmailMultiAlternatives
from joblib.numpy_pickle_utils import xrange
from sklearn.cluster import KMeans  # for clustering
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from decimal import Decimal
from PIL import Image
import os
from django.conf import settings
from django.conf.urls.static import static

from django.utils.functional import cached_property

from home.models import *
import math
from math import degrees


# Create your views here.
def index(request):
    manProduct = Product.objects.filter(category_id='2').values().order_by('-view')[:4]
    womanProduct = Product.objects.filter(category_id='1').values().order_by('-view')[:4]
    kidProduct = Product.objects.filter(category_id='3').values().order_by('-view')[:4]
    blog = News.objects.all()[:4]
    context = {
        'items': manProduct,
        'items_2': womanProduct,
        'items_3': kidProduct,
        'blog' : blog
    }

    return render(request, "home/index.html", context)

def LoginView(request):
    return render(request,'admin/login.html')

def productDetail(request, id):
    productItem = Product.objects.get(id=id)
    productItem.view +=1
    productItem.save()
    return render(request, 'home/single-product.html',{
        'item':productItem})

cart = {}
def shoppingCart(request):
    if request.is_ajax():
        id = request.POST.get('id')
        num = request.POST.get('num')

        proDetail = Product.objects.get(id =id)
        if id in cart.keys():
            itemCart = {
                "name": proDetail.product,
                "price":float(proDetail.price),
                "description": str(proDetail.description),
                "image":proDetail.image,
                "number":float(float(cart[str(id)]['number'])+float(num))
            }
        else:
            itemCart = {
                "name": proDetail.product,
                "price": float(proDetail.price),
                "description": str(proDetail.description),
                "image": proDetail.image,
                "number": num
            }


        cart[id] = itemCart
        request.session['cart'] = cart
        cartInfo = request.session['cart']

        html = render_to_string('home/cart.html',{'cart': cartInfo})
    return HttpResponse(html)

def deleteProduct(request):
    if request.is_ajax():
        id = request.POST.get('id')
        cart = {}
        cart = request.session['cart']
        del cart[str(id)]
        request.session['cart'] = cart
    return render(request, 'home/index.html')

def checkout(request):
    return render(request, 'home/checkout.html')

def news(request):
    news = News.objects.all()
    return render(request,'home/news.html',{'items':news})

def newsDetail(request,id):
    news = News.objects.filter(id =id).first()
    return render(request, 'home/single-news.html', {'item': news})

def typeGlass(request,name,id):
    frame = FrameGlasses.objects.all()
    category = Category.objects.filter(name=name).first().id
    if id !=0:

        product = Product.objects.filter(category = category,type_id=id)
        nam = FrameGlasses.objects.filter(id = id).first().frame
        count=0
        if len(product)>0:
            count = 1

    else:
        nam = "all in"
        count =1
        product = Product.objects.filter(category = category)

    return render(request, 'home/type-glasses.html',{'frame': frame,'product': product,'name': nam,'gender': name,'count': count})

def order(request):
    if request.method == 'POST':

        cart = {}
        cart = request.session['cart']
        orders = Orders()
        orders.name_customer = request.POST.get('name')
        orders.add_customer = request.POST.get('add1')
        orders.addDetail_customer = request.POST.get('add1')+','+request.POST.get('add2')+','+request.POST.get('add3')
        orders.total_order = 0
        orders.payment = request.POST.get('payment')
        orders.phone_customer = request.POST.get('phone')

        orders.save()
        total = 0
        for key in request.session['cart']:
            detail = OrderDetails()
            detail.id_order_id = orders.id
            detail.id_product_id = key
            detail.num_product = cart[key]['number']
            detail.save()
            product = Product.objects.filter(id = key).first().price
            total += float(detail.num_product)*float(product)

        charges = total*3/100
        total_price = total + charges
        ord = Orders.objects.filter(id = orders.id).first()
        ord.total_order = total_price
        ord.save()
        context = {
            'cart' : request.session['cart'],
            'bill' : orders.id,
            'date' : orders.created_date,
            'name' : orders.name_customer,
            'add'  : orders.add_customer,
            'phone': orders.phone_customer,
            'price': total,
            'charges': charges,
            'total': total_price
        }

        message = render_to_string('home/message.html',context,)
        msg = EmailMultiAlternatives("Details Bill", "abc", settings.EMAIL_HOST_PASSWORD, [request.POST.get('email')])
        msg.attach_alternative(message, "text/html")
        msg.send()
        for key in request.session['cart']:
            print(cart[key]['name'])
        # del cart
        request.session['cart'] = {}
        return redirect('/')

    else:
        return render(request,'home/checkout.html')

def classification(request):
    return render(request,'home/classification.html')

def face(request):
    # load the image
    imagepath = "home/static/images/new.jpeg"
    face_cascade_path = "/haarcascade_frontalface_default.xml"
    predictor_path = "/shape_predictor_68_face_landmarks.dat"

    # create the haar cascade for detecting face and smile
    faceCascade = cv2.CascadeClassifier(face_cascade_path)

    predictor = dlib.shape_predictor(predictor_path)
    scale_percent = 50
    # read the image
    image = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
    print(image.shape);F
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    image = cv2.resize(image, dsize)
    original = image.copy()

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply a Gaussian blur with a 3 x 3 kernel to help remove high frequency noise
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)

    faces = faceCascade.detectMultiScale(
        gauss,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print("found {0} faces!".format(len(faces)))
    try:
        for (x, y, w, h) in faces:
            # draw a rectangle around the faces
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            detected_landmarks = predictor(image.astype('uint8'), dlib_rect).parts()
            landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        results = original.copy()

        for (x, y, w, h) in faces:
            # draw a rectangle around the faces
            cv2.rectangle(results, (x, y), (x + w, y + h), (0, 255, 0), 2)
            temp = original.copy()
            forehead = temp[y:y + int(0.25 * h), x:x + w]
            rows, cols, bands = forehead.shape
            X = forehead.reshape(rows * cols, bands)

            # kmeans
            kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
            y_kmeans = kmeans.fit_predict(X)
            for i in range(0, rows):
                for j in range(0, cols):
                    if y_kmeans[i * cols + j] == True:
                        forehead[i][j] = [255, 255, 255]
                    if y_kmeans[i * cols + j] == False:
                        forehead[i][j] = [0, 0, 0]

            forehead_mid = [int(cols / 2), int(rows / 2)]  # midpoint of forehead
            lef = 0
            # gets the value of forehead point
            pixel_value = forehead[forehead_mid[1], forehead_mid[0]]
            for i in range(0, cols):
                # enters if when change in pixel color is detected
                if forehead[forehead_mid[1], forehead_mid[0] - i].all() != pixel_value.all():
                    lef = forehead_mid[0] - i
                    break;
            left = [lef, forehead_mid[1]]
            rig = 0
            for i in range(0, cols):
                # enters if when change in pixel color is detected
                if forehead[forehead_mid[1], forehead_mid[0] + i].all() != pixel_value.all():
                    rig = forehead_mid[0] + i
                    break;
            right = [rig, forehead_mid[1]]

        # drawing line1 on forehead with circles
        # specific landmarks are used.
        line1 = np.subtract(right + y, left + x)[0]
        cv2.line(results, tuple(x + left), tuple(y + right), color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 1', tuple(x + left), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),
                    thickness=2)
        cv2.circle(results, tuple(x + left), 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, tuple(y + right), 5, color=(255, 0, 0), thickness=-1)

        # drawing line 2 with circles
        linepointleft = (landmarks[1, 0], landmarks[1, 1])
        linepointright = (landmarks[15, 0], landmarks[15, 1])
        line2 = np.subtract(linepointright, linepointleft)[0]
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 2', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),
                    thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

        # drawing line 3 with circles
        linepointleft = (landmarks[3, 0], landmarks[3, 1])
        linepointright = (landmarks[13, 0], landmarks[13, 1])
        line3 = np.subtract(linepointright, linepointleft)[0]
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 3', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),
                    thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

        # drawing line 4 with circles
        linepointbottom = (landmarks[8, 0], landmarks[8, 1])
        linepointtop = (landmarks[8, 0], y)
        line4 = np.subtract(linepointbottom, linepointtop)[1]
        cv2.line(results, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 4', linepointbottom, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),
                    thickness=2)
        cv2.circle(results, linepointtop, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointbottom, 5, color=(255, 0, 0), thickness=-1)
        print(line1,line2,line3,line4)

        similarity = np.std([line1, line2, line3])
        print("similarity=",similarity)
        ovalsimilarity = np.std([line2, line4])
        print('diam=',ovalsimilarity)

        # we use arcustangens for angle calculation
        ax, ay = landmarks[3, 0], landmarks[3, 1]
        bx, by = landmarks[4, 0], landmarks[4, 1]
        cx, cy = landmarks[5, 0], landmarks[5, 1]
        dx, dy = landmarks[6, 0], landmarks[6, 1]

        alpha0 = math.atan2(cy - ay, cx - ax)
        alpha1 = math.atan2(dy - by, dx - bx)
        alpha = alpha1 - alpha0
        angle = abs(degrees(alpha))
        angle = 180 - angle
        a = ""
        for i in range(1):
            if similarity < 10:
                if angle < 160:
                    print('squared shape.Jawlines are more angular')
                    a = "squared shape"
                    break
                else:
                    print('round shape.Jawlines are not that angular')
                    a = "round shape"
                    break
            if line3 > line1:
                if angle < 160:
                    print('triangle shape.Forehead is more wider')
                    a = "trianfle shape"
                    break
            if ovalsimilarity < 10:
                print('diamond shape. line2 & line4 are similar and line2 is slightly larger')
                a = "diamond"
                break
            if line4 > line2:
                if angle < 160:
                    print('rectangular. face length is largest and jawline are angular ')
                    a = "rectangular"
                    break
                else:
                    print('oblong. face length is largest and jawlines are not angular')
                    a = "oblong"
                    break
            print("Damn! Contact the developer")

        shape = ShapeFace.objects.get(shape=a)
        glasses = Recommendation.objects.filter(id_shape= shape.id)
        eye = {}
        print(request.POST.get('gender'))
        gender = Category.objects.filter(name = 'Men').first()
        for i in glasses:
            eye = Product.objects.filter(type_id= i.id_frame_id, category_id= gender).values()

    except:
        a = "No face searched"
        eye =  {}
        gender = Category.objects.filter(name = request.POST.get('gender')).first()
    return render(request, 'home/classification.html', {'face': a, 'items_3': eye, 'gender': gender.name})


# !/usr/bin/python

import dlib
import cv2
import numpy as np
from scipy import ndimage

# Resize an image to a certain width
def resize(img, width):
    r = float(width) / img.shape[1]
    dim = (width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


# Combine an image that has a transparency alpha channel
def blend_transparent(face_img, sunglasses_img):
    overlay_img = sunglasses_img[:, :, :3]
    overlay_mask = sunglasses_img[:, :, 3:]

    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


# Find the angle between two points
def angle_between(point_1, point_2):
    angle_1 = np.arctan2(*point_1[::-1])
    angle_2 = np.arctan2(*point_2[::-1])
    return np.rad2deg((angle_1 - angle_2) % (2 * np.pi))

import base64
def saveImage(request):
    imageUrls = request.POST.get('url_image')
    new_data = imageUrls.replace('data:image/jpeg;base64,', '')
    imgdata = base64.b64decode(new_data)
    filename = 'home/static/images/new.jpeg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    return redirect("/")

def tryonGlass(request,id):
    video_capture = cv2.VideoCapture("male.mp4")
    glasses = cv2.imread("home/static/images/product"+str(id)+".png", -1)

    imga = cv2.imread("home/static/images/new.jpeg")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat")

    while True:
        # ret, img = video_capture.read()
        img = resize(imga, 700)
        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            # detect faces
            dets = detector(gray, 1)

            # find face box bounding points
            for d in dets:
                x = d.left()
                y = d.top()
                w = d.right()
                h = d.bottom()

            dlib_rect = dlib.rectangle(x, y, w, h)

            detected_landmarks = predictor(gray, dlib_rect).parts()

            landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                if idx == 0:
                    eye_left = pos
                elif idx == 16:
                    eye_right = pos

                try:
                    # cv2.line(img_copy, eye_left, eye_right, color=(0, 255, 255))
                    degree = np.rad2deg(np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]))

                except:
                    pass


            eye_center = (eye_left[1] + eye_right[1]) / 2

            # Sunglasses translation
            glass_trans = int(-0.75 * (eye_center - y))

            # resize glasses to width of face and blend images
            face_width = w - x

            # resize_glasses
            glasses_resize = resize(glasses, face_width)

            # Rotate glasses based on angle between eyes
            yG, xG, cG = glasses_resize.shape
            glasses_resize_rotated = ndimage.rotate(glasses_resize, (degree + 90))
            glass_rec_rotated = ndimage.rotate(img[y + glass_trans:y + yG + glass_trans, x:w], (degree + 90))

            # blending with rotation
            h5, w5, s5 = glass_rec_rotated.shape
            rec_resize = img_copy[y + glass_trans:y + h5 + glass_trans, x:x + w5]
            blend_glass3 = blend_transparent(rec_resize, glasses_resize_rotated)
            img_copy[y + glass_trans:y + h5 + glass_trans, x:x + w5] = blend_glass3
            # print(img_copy)
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            data = Image.fromarray(img_copy)
            url ="home/static/"
            path = "images/my.png"
            data.save(url + path)

        except:
            print('abc')

        productItem = Product.objects.get(id=id)
        return render(request, 'home/single-product.html', {'item': productItem,'image': path})

