from rest_framework.views import APIView
from django.core.files.storage import FileSystemStorage
from apps.multiapp.services import predict_flower
from rest_framework.response import Response
from rest_framework import status

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
        return Response(data, status=status.HTTP_200_OK)
