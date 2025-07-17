import re


en_alphabets= "([A-Za-z])"
en_prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
en_suffixes = "(Inc|Ltd|Jr|Sr|Co)"
en_starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
en_acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me|ru|su|рф|рус)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

class Text:
    def __init__(self, text: str | None=None, path: str | None=None):
        self.text_length = 0
        self.text = ""
        self.sentencies = []
        self.words = []
        if text is not None and path is None:
            self.text_from_string(text=text)
        elif text is None and path is not None:
            self.text_from_file(path=path)
    
    
    def text_from_string(self, text: str):
        self.text += '\n' + text
        self.sentencies.extend(self.split_into_sentences(text=text))
        splitted = self.split_into_words(text)
        self.words.extend(splitted)
        self.text_length += len(splitted)
        
    def text_from_file(self, path: str):
        with open(path, 'r') as f:
            _text = str(f.read())
        self.text += '\n' + _text
        self.sentencies.extend(self.split_into_sentences(text=_text))
        splitted = self.split_into_words(text=_text)
        self.words.extend(splitted)
        self.text_length += len(splitted)
        
    
    def split_into_sentences(self, text: str) -> list[str]:
        """
        Split the text into sentences.

        If the text contains substrings "<prd>" or "<stop>", they would lead 
        to incorrect splitting because they are used as markers for splitting.

        :param text: text to be split into sentences
        :type text: str

        :return: list of sentences
        :rtype: list[str]
        """
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(en_prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
        text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + en_alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(en_acronyms+" "+en_starters,"\\1<stop> \\2",text)
        text = re.sub(en_alphabets + "[.]" + en_alphabets + "[.]" + en_alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(en_alphabets + "[.]" + en_alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+en_suffixes +"[.] "+en_starters," \\1<stop> \\2",text)
        text = re.sub(" "+en_suffixes +"[.]"," \\1<prd>",text)
        text = re.sub(" " + en_alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = [s.strip() for s in sentences]
        if sentences and not sentences[-1]: sentences = sentences[:-1]
        return sentences
    
    
    def split_into_words(self, text: str):
        text = re.sub(r"[^a-zA-Z@#' ]", '', text) 
        return re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text)