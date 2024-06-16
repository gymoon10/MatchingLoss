
from models.ERFNet_Semantic_Original import ERFNet_Semantic_Original
from models.ERFNet_Semantic_Embedding import ERFNet_Semantic_Embedding
from models.ERFNet_Semantic2 import ERFNet_Semantic2
from models.ERFNet_Semantic_Mat import ERFNet_Semantic3
from models.ERFNet_Semantic3_1 import ERFNet_Semantic3_1
from models.ERFNet_Semantic4 import ERFNet_Semantic4
from models.DeepLabV3 import ResNet

from models.ERFNet_Semantic_Dual import ERFNet_Semantic_Dual
from models.ERFNet_Semantic_Dual2 import ERFNet_Semantic_Dual2
from models.ERFNet_Semantic_Dual3 import ERFNet_Semantic_Dual3

from models.BranchedERFNet_Semantic3 import Branched_ERFNet_Semantic3


def get_model(name, model_opts):
    if name == "ERFNet_Semantic_Original":
        model = ERFNet_Semantic_Original(**model_opts)
        return model

    elif name == "DeepLabV3":
        model = ResNet()
        return model

    elif name == "ERFNet_Semantic_Embedding":
        model = ERFNet_Semantic_Embedding(**model_opts)
        return model

    elif name == "ERFNet_Semantic2":
        model = ERFNet_Semantic2(**model_opts)
        return model

    elif name == "ERFNet_Semantic3":
        model = ERFNet_Semantic3(**model_opts)
        return model

    elif name == "ERFNet_Semantic3_1":
        model = ERFNet_Semantic3_1(**model_opts)
        return model

    elif name == "ERFNet_Semantic4":
        model = ERFNet_Semantic4(**model_opts)
        return model

    elif name == "ERFNet_Semantic_Dual":
        model = ERFNet_Semantic_Dual(**model_opts)
        return model

    elif name == "ERFNet_Semantic_Dual2":
        model = ERFNet_Semantic_Dual2(**model_opts)
        return model

    elif name == "ERFNet_Semantic_Dual3":
        model = ERFNet_Semantic_Dual3(**model_opts)
        return model

    elif name == "Branched_ERFNet_Semantic3":
        model = Branched_ERFNet_Semantic3(**model_opts)
        return model

    else:
        raise RuntimeError("model \"{}\" not available".format(name))
