import csv
from pathlib import Path
from django.core.management.base import BaseCommand
from recommendations.models import Movie, MovieVote
from decimal import Decimal

class Command(BaseCommand):
    help = "Import Movies and Votes from CSV files into the database"

    def add_arguments(self, parser):
        parser.add_argument(
            '--movies',
            type=str,
            required=True,
            help='Path to Movies.csv file'
        )
        parser.add_argument(
            '--votes',
            type=str,
            required=True,
            help='Path to Votes.csv file'
        )

    def handle(self, *args, **options):
        movies_path = Path(options['movies'])
        votes_path = Path(options['votes'])

        if not movies_path.exists():
            self.stderr.write(self.style.ERROR(f"Movies file not found: {movies_path}"))
            return
        if not votes_path.exists():
            self.stderr.write(self.style.ERROR(f"Votes file not found: {votes_path}"))
            return

        self.stdout.write(self.style.WARNING("Importing movies..."))
        with open(movies_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # self.stdout.write(self.style.NOTICE(f"Row data: {row}"))
                movie, created = Movie.objects.update_or_create(
                    movieId=int(row['movieId']),
                    defaults={
                        'title': row['title'],
                        'original_title': row['original_title'],
                        'genre': row['genre'],
                        'genres': row['genres'],
                        'release_date': row['release_date'],
                        'overview': row['overview'],
                        'vote_average': Decimal(0),
                        'vote_count': int(row['vote_count']),
                        'popularity': Decimal(0),
                        'adult': row['adult'].lower() in ['true', '1'],
                        'backdrop_path': row['backdrop_path'],
                        'original_language': row['original_language'],
                        'poster_path': row['poster_path'],
                        'video': row['video'].lower() in ['true', '1'],
                        'cast': row['cast'],
                        'direct': row['direct'],
                        'combined_features': row['combined_features'],
                    }
                )
                if created:
                    self.stdout.write(self.style.SUCCESS(f"Inserted movie: {movie.title}"))
                else:
                    self.stdout.write(self.style.NOTICE(f"Movie already exists: {movie.title}"))

        self.stdout.write(self.style.WARNING("Importing votes..."))
        with open(votes_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    movie = Movie.objects.get(movieId=int(row['movieId']))
                except Movie.DoesNotExist:
                    self.stderr.write(self.style.ERROR(f"MovieId {row['movieId']} not found, skipping vote."))
                    continue

                vote_obj, created = MovieVote.objects.get_or_create(
                    movie=movie,
                    createdBy=row['userId'],  # string, not FK
                    defaults={
                        'vote': int(row['rating']),
                    }
                )
                if created:
                    self.stdout.write(self.style.SUCCESS(f"Inserted vote for movieId {row['movieId']}"))
                else:
                    self.stdout.write(self.style.NOTICE(f"Vote already exists for movieId {row['movieId']} and user {row['userId']}"))

        self.stdout.write(self.style.SUCCESS("Import completed successfully!"))
