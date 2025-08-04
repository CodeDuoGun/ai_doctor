from app.model.llm.deepseek import DeepSeekChat


class LLMFactory():
    """
    A factory class for creating instances of LLM (Language Model) classes.
    """

    def __init__(self):
        self.llm_classes = {
            'llama': 'Llama',
            'gpt-3.5-turbo': 'ChatOpenAI()',
            'gpt-4': 'ChatOpenAI()',
            'gpt-4-32k': 'ChatOpenAI()',
            "deepseek": DeepSeekChat(), 
            "coze_deepseek": DeepSeekChat(), 
            "coze_deepseek-r1": DeepSeekChat(), 
            'baichuan': 'Baichuan',
        }

    def create_llm(self, model_name: str, **kwargs):
        """
        Create an instance of the specified LLM class.

        :param model_name: The name of the LLM class to create.
        :param kwargs: Additional keyword arguments to pass to the LLM constructor.
        :return: An instance of the specified LLM class.
        """
        if model_name not in self.llm_classes:
            raise ValueError(f"Model '{model_name}' is not supported.")
        
        llm_class = self.llm_classes[model_name]
        # Assuming that the classes are defined and imported correctly
        return llm_class