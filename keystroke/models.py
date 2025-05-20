from django.db import models
from django.contrib.auth.models import User

class KeystrokeSample(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    typed_text = models.TextField()
    timing_data = models.JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.timestamp}"