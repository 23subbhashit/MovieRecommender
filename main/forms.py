from django import forms

class MovieRatingForm(forms.Form):
    movie_title = forms.CharField(label='Movie Title', max_length=100)
    rating = forms.IntegerField(label='Rating', min_value=1, max_value=5)