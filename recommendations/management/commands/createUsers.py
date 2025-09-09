from django.core.management.base import BaseCommand
from recommendations.models import  MovieVote, User


class Command(BaseCommand):
    help = 'Create sample users'

    def handle(self, *args, **kwargs):
        moviesVotes = MovieVote.objects.all()
        user_ids = set(mv.createdBy for mv in moviesVotes)
        for user_id in user_ids:
            User.objects.get_or_create(id=user_id, defaults={'name': "User " + str(user_id)})
        self.stdout.write(self.style.SUCCESS(f'Successfully created {len(user_ids)} users'))
        
        
        
        