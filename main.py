import spacy, yaml
from numpy import random
from sklearn.naive_bayes import GaussianNB

tagger = spacy.load("es_core_news_sm")


class CorpusReader:
    def __init__(self, path):
        with open(path) as yamlfile:
            document = yaml.load(yamlfile, Loader=yaml.FullLoader)
            self.no_context = document.get("NO_CONTEXT_TAG", "no_context")
            self.contexts = document.get("contexts")
            self.responses = document.get("responses")

    def jitter(self, vector, response, sigma=2, jitter_count=10):
        x_jitters, y_jitters = [], []

        for _ in range(jitter_count):
            x_jitters.append(random.normal(vector, sigma))
            y_jitters.append(response)

        return  x_jitters, y_jitters 

    @property
    def dataset(self):
        x_train, y_train = [], []

        for label, values in self.contexts.items():
            for context in values:
                context = tagger(context.lower())
                x, y = self.jitter(context.vector, label)
                x_train.extend(x)
                y_train.extend(y)

        return x_train, y_train


class Chatbot:
    def __init__(self, corpus_path, classifier):
        self.corpus = CorpusReader(corpus_path)
        self.classifier = classifier()        
        self.last_label = ""
        self.train()

    def check_label(self, label):
        if label == self.corpus.no_context:
            self.last_label = ""
        
        if self.last_label and f"{self.last_label} {label}" in self.corpus.responses:
            label = f"{self.last_label} {label}"
        
        return label

    def train(self):
        self.classifier.fit(*self.corpus.dataset)
    
    def get_text_from_graph(self, label):
        label = self.check_label(label)
        label_values = self.corpus.responses[label]

        while True:
            response = response = random.choice(label_values) 

            if response.startswith("$"):
                label_values = self.corpus.responses[response[1:]]
                continue

            break

        return response

    def response(self, context):
        context = tagger(context.lower())
        label = self.classifier.predict([context.vector])[0]
        response = self.get_text_from_graph(label)
        self.last_label = label

        return response


chatbot = Chatbot("corpus.yaml", GaussianNB)

while True:
    context = input(">> ")
    print(f"BOT: {chatbot.response(context)}")