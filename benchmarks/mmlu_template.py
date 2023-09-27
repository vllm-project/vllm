import pandas as pd
import json
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["question", "A", "B", "C", "D", "Answer"],
    template=
    """
USER: {question}
A. {A}
B. {B}
C. {C}
D. {D} ASSISTANT: Answer: {Answer}</s>
""",
)

template_with_analyse = PromptTemplate(
    input_variables=["question", "A", "B", "C", "D"],
    template=
    """
Q:{question}
(A) {A} (B) {B} (C) {C} (D) {D}
A: Let's think step by step.
""",
)


def gen_prompt(train_df, subject, k=1):
    prompt = "SYSTEM: The following are multiple choice questions (with answers) about {}," \
             "Please select the correct answer from the options.".format(subject.replace('_', ' '))

    for i in range(k):
        prompt += template.format(question=train_df.iloc[i, 0],
                                  A=train_df.iloc[i, 1],
                                  B=train_df.iloc[i, 2],
                                  C=train_df.iloc[i, 3],
                                  D=train_df.iloc[i, 4],
                                  Answer=train_df.iloc[i, 5]
                                  )[1:-1]
    return prompt


## add an abstract base class or common base class for generality
class MMLUTemplate():

    def __init__(self, subject, file_path, is_analyse):
        self.fiveShotTemplate = ""
        self.file_path = file_path
        self.subject = subject
        self.choices = ["A", "B", "C", "D"]
        self.is_analyse = is_analyse
        self.few_shot_template = ""
        if not is_analyse:
            self.getFewShotBaseTemplates()
        else:
            self.getFewShotBaseTemplateAnalyse()

    def getFewShotBaseTemplates(self, k=5):
        """few_shot模板不带分析"""
        dev_df = pd.read_csv(self.file_path, header=None)

        self.few_shot_template = gen_prompt(dev_df, self.subject, k)
        return self.few_shot_template

    def getFewShotBaseTemplateAnalyse(self):
        """few_shot模板带分析，更改json文件就行"""
        mmlu_prompt = json.load(open('templates/lib_prompt/mmlu-cot.json'))
        self.few_shot_template = mmlu_prompt[self.subject]
        return self.few_shot_template

    def getTemplate(self, test_df, i):
        """获得模板"""
        if self.is_analyse:
            templ = template_with_analyse.format(
                question=test_df.iloc[i, 0],
                A=test_df.iloc[i, 1],
                B=test_df.iloc[i, 2],
                C=test_df.iloc[i, 3],
                D=test_df.iloc[i, 4]
            )

            return self.few_shot_template + "\n" + templ

        else:
            prompt_end = template.format(
                question=test_df.iloc[i, 0],
                A=test_df.iloc[i, 1],
                B=test_df.iloc[i, 2],
                C=test_df.iloc[i, 3],
                D=test_df.iloc[i, 4],
                Answer='')[1:-5]
            return self.few_shot_template + prompt_end
    @staticmethod
    def findAnswer(res):
        """解析函数"""
        # print("模型输出为:", res)
        d = "NO"
        for d_ in res:
            if 65 <= ord(d_) <= 68:
                d = d_
                break
        # print("答案解析为:", d)
        return d

    @staticmethod
    def findAnwerUsingRule(res):
        # print("模型输出为:", res)
        result = "NO"
        pattern = 'the answer is ('
        try:
            pred = res.lower().split(pattern)[1][0]

            if 65 <= ord(pred.upper()) <= 68:
                result = pred.upper()
        except:
            pass

        # print("答案解析为:",result)
        return result
