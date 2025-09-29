from django.urls import path

from . import views

urlpatterns = [
    path("<int:index>/", views.index, name="index"),
    path("<int:index>/<int:n_recommendations>", views.index, name="index"),
    path("ping/", views.ping, name="ping"),
    path("hybrid/<str:userId>/", views.hybridNeuralNetworkRecomendations, name="hybrid_recommendations"),
    path("hybrid/<str:userId>/<int:n_recommendations>/", views.hybridNeuralNetworkRecomendations, name="hybrid_recommendations"),
    path("train-hybrid-model/", views.trainModel, name="train_hybrid_model"),
]