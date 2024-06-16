
from datasets.CornDataset import CornDataset
# from datasets.CornDatasetCrop import CornDatasetCrop
# from datasets.CornDatasetTwoClass import CornDatasetTwoClass
from datasets.CityscapesDataset import CityscapesDataset

# from datasets.CornDatasetCropDistill import CornDatasetCropDistill
# from datasets.CornDatasetCropDual import CornDatasetCropDual


def get_dataset(name, dataset_opts):
    if name == "CornDataset":
        return CornDataset(**dataset_opts)

    elif name == "CornDatasetCrop":
        return CornDatasetCrop(**dataset_opts)

    elif name == "CityscapesDataset":
        return CityscapesDataset(**dataset_opts)

    elif name == "CornDatasetTwoClass":
        return CornDatasetTwoClass(**dataset_opts)

    elif name == "CornDatasetCropDistill":
        return CornDatasetCropDistill(**dataset_opts)

    elif name == "CornDatasetCropDual":
        return CornDatasetCropDual(**dataset_opts)

    else:
        raise RuntimeError("Dataset {} not available".format(name))
