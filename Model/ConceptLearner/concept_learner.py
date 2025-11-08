class ConceptLearner:
    def __init__(self, concepts=None):
        if concepts is None:
            concepts = []
        self.concepts = concepts

    def add_concept(self, concept):
        self.concepts.append(concept)

    def get_concepts(self):
        return self.concepts