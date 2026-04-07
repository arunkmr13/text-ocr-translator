from googletrans import Translator
translator = Translator()

def translate_text(text, target="en"):
    try:
        return translator.translate(text, dest=target).text
    except:
        return text
