from dataclasses import dataclass


@dataclass
class AbstractRunResource():
    # setup
    prompt: str
    actual_value: str
    test_abstract: str
    abstract_id: str
    index_cao: str
    
    # final
    skipped: bool | None = None
    correct: bool | None = None
    predicted_value: str | None = None
    llm_answer: str | None = None
    error: list | None = None
    
    def finalize(self, skipped, correct, predicted_value, llm_answer, error):
        self.skipped = skipped
        self.correct = correct
        self.predicted_value = predicted_value
        self.llm_answer = llm_answer
        self.error = error
        if skipped:
            self.correct = False
            self.predicted_value = "fail"
            self.actual_value = "fail"