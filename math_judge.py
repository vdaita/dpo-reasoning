from trl.trl.trainer.judges import BaseRankJudge
from math_verify import parse, verify
from rapidfuzz.process import extractOne

class MathJudge(BaseRankJudge):
    def __init__(self, qa_dict: dict[str, str]):
        super(MathJudge, self).__init__()
        self.qa_dict = qa_dict
        self.questions = list(qa_dict.keys())

    def judge(self, prompts, completions):
        results = [0]
        for (prompt, (ans_a, ans_b)) in zip(prompts, completions):
            ground_truth = parse(self.qa_dict[extractOne(prompt, self.questions)[0]])
            ans_a_correct = verify(ground_truth, parse(ans_a))
            ans_b_correct = verify(ground_truth, parse(ans_b))
            
            if ans_a_correct and not ans_b_correct:
                results.append(1)
            elif ans_a_correct and ans_b_correct:
                results.append(0)
            else:
                results.append(-1)
        return results        
