from rest_framework.views import APIView
from django.core.files.storage import FileSystemStorage
from apps.multiapp.services import predict_flower
from rest_framework.response import Response
from rest_framework import status
from django.db import connection


class FlowerPrediction(APIView):
    
    def post(self, request, *args, **kwargs):
        fss = FileSystemStorage()
        image = fss.save(request.FILES['imagePath'].name, request.FILES['imagePath'])
        img_url = fss.url(image)
        predictions = predict_flower(img_url[1:])
        data = {
            'img_url': img_url,
            'predictions': predictions
        }
        print(data)
        if predictions[0]['rate'] > 40:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM flowers WHERE Name = %s LIMIT 1", predictions[0]['name'])
                row = cursor.fetchone()
                data = {
                    'img_url': img_url,
                    'predictions': predictions[0],
                    'infomation': row
                }
        return Response(data, status=status.HTTP_200_OK)
