from django.db import models
from django.contrib.auth.models import User

class KeystrokeSample(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    typed_text = models.TextField()
    timing_data = models.JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.timestamp}"
    
class KeystrokeProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    model_file = models.FileField(upload_to='models/')
    trained_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Model for {self.user.username}"
    
class KeystrokeLogin(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    typed_text = models.CharField(max_length=150)
    timing_data = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} login at {self.timestamp}"