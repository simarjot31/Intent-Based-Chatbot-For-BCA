## Initial Setup & Method of Execution:

Create a virtual environment
$ python3 -m venv venv
$ venv/Scripts/activate.bat
```
Install dependencies
```
$ (venv) pip install Flask torch torchvision nltk
```
Install nltk package
```
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('punkt_tab')
```
Modify `intents.json` with different intents and responses for your Chatbot

Run
```
$ (venv) python train.py
```
This will dump data.pth file. And then run
the following command to test it in the console.
```
$ (venv) python chat.py
```

Now for deployment implement and execute `app.py` and `app.js`.


<!-- When the form is submitted the admin gets the mail about the details which has been submitted in the form. -->