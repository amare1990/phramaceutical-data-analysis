from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services import PredictionService

class PredictView(APIView):
  """
  Handles POST requests for making predictions using the trained model.
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.predictor = PredictionService()

  def post(self, request):
    """
    Accepts input data in JSON format and returns model predictions.
    """
    try:
      input_data = request.data.get('input', None)
      if not input_data:
        return Response({"error": "No input data provided"}, status=status.HTTP_400_BAD_REQUEST)

      predictions = self.predictor(input_data)
      return Response({'predictions': predictions}, status=status.HTTP_200_OK)
    except Exception as e:
      return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

