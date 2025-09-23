from django.db import models

from movinderAPI.settings import USE_POSTGRES

class Movie(models.Model):
    movieId = models.IntegerField()
    title = models.CharField(max_length=255)
    original_title = models.CharField(max_length=255)
    genre = models.CharField(max_length=255)
    genres = models.TextField()  # could be JSON or comma-separated
    release_date = models.CharField(max_length=20)
    overview = models.TextField()
    vote_average = models.DecimalField(max_digits=4, decimal_places=2)
    vote_count = models.IntegerField()
    popularity = models.DecimalField(max_digits=10, decimal_places=2)
    adult = models.BooleanField()
    backdrop_path = models.CharField(max_length=255)
    original_language = models.CharField(max_length=10)
    poster_path = models.CharField(max_length=255)
    video = models.BooleanField()
    cast = models.TextField()
    direct = models.TextField()
    combined_features = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.title

    class Meta:
        db_table = "Movie" if USE_POSTGRES else "recommendations_movie"
        managed = False if USE_POSTGRES else True


class MovieVote(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="movie_votes")
    movieId = models.IntegerField()  # Redundant but kept for compatibility
    vote = models.IntegerField()
    createdById = models.CharField(max_length=255)  # Now a plain string, not a FK
    createdAt = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "MovieVote" if USE_POSTGRES else "recommendations_movievote"
        managed = False if USE_POSTGRES else True
        unique_together = ('movie', 'createdById')

    def __str__(self):
        return f"{self.createdById} voted {self.vote} for {self.movie.title}"


    from django.db import models

class User(models.Model):
    id = models.CharField(max_length=25, primary_key=True)  # cuid in Prisma
    name = models.CharField(max_length=255, null=True, blank=True)
    email = models.EmailField(unique=True, null=True, blank=True)
    email_verified = models.DateTimeField(null=True, blank=True)
    image = models.CharField(max_length=255, null=True, blank=True)

    # Relations (optional for read-only; can define if needed)
    # accounts, sessions, posts, MovieVote would need separate models
    # with ForeignKey pointing to this User

    class Meta:
        db_table = "User" if USE_POSTGRES else "recommendations_user"
        managed = False if USE_POSTGRES else True

